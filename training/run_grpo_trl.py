# training/run_grpo_trl.py
import glob
import os
import math
import numpy as np
# ==== 在任何用到 transformers 之前打补丁 ====
import transformers
from transformers.utils import import_utils

def _disable_torch_load_check(*args, **kwargs):
    # 课程项目用的临时补丁：关闭 torch>=2.6 强制检查
    # 注意只加载来自 HuggingFace 官方或可信作者的权重
    return

# 1) 改 import_utils 里的实现
import_utils.check_torch_load_is_safe = _disable_torch_load_check

# 2) 同时改 modeling_utils 里拿到的别名
try:
    from transformers import modeling_utils
    if hasattr(modeling_utils, "check_torch_load_is_safe"):
        modeling_utils.check_torch_load_is_safe = _disable_torch_load_check
except Exception:
    # 万一不同版本导入方式不一样，这里就静默跳过
    pass

# 3) 关键：改 trainer 模块里的本地引用
try:
    import transformers.trainer as trainer_mod
    if hasattr(trainer_mod, "check_torch_load_is_safe"):
        trainer_mod.check_torch_load_is_safe = _disable_torch_load_check
except Exception:
    pass
# ==========================================
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

import torch
import torch.nn.functional as F
from peft import PeftModel
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

from data_utils.dataset_build import load_split_raw_data
from data_utils.prompts import generate_test_prompt

###############################################
# 1. 分类 helper：从 LLaMA 得到三分类概率
###############################################

LABEL_ORDER = ["negative", "neutral", "positive"]


def get_label_token_ids(tokenizer, label_order):
    """
    label_order: 比如 ["negative", "neutral", "positive"]
    返回: dict[label] -> token_ids (list[int])
    """
    label_token_ids = {}
    for label in label_order:
        ids = tokenizer.encode(label, add_special_tokens=False)
        label_token_ids[label] = ids
    return label_token_ids

@torch.no_grad()
def get_student_probs_from_prompts(model, tokenizer, prompts, label_order, max_length: int = 512):
    """
    给定一批 prompt，用当前学生模型算出对每个情感标签的概率。
    这里用的是「下一 token」的 logits，对每个 label 的 token 集合做 log-sum-exp 之后 softmax。
    """
    device = next(model.parameters()).device

    toks = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    ).to(device)

    out = model(**toks)
    # 每个样本最后一个非 padding token 的 index
    last_idx = toks["attention_mask"].sum(dim=1) - 1
    logits_last = out.logits[torch.arange(out.logits.size(0)), last_idx, :]

    label_token_ids = get_label_token_ids(tokenizer, label_order)
    label_scores = []
    for label in label_order:
        ids = label_token_ids[label]
        if len(ids) == 1:
            score = logits_last[:, ids[0]]
        else:
            # 多 token 的 label：对对应 token 的 logits 做 log-sum-exp
            score = torch.logsumexp(logits_last[:, ids], dim=1)
        label_scores.append(score)

    scores = torch.stack(label_scores, dim=1)  # [B, num_labels]
    probs = scores.softmax(dim=1)
    return probs

############################################################
# 2. Dataset 转换：X_train_raw(DataFrame) → HF Dataset
############################################################

def convert_to_hf_dataset(X_train_raw):
    """
    X_train_raw: DataFrame，包含列 ["text", "sentiment", "orig_text", "pert_text"]

    输出字段：
        - prompt:       原始文本 orig_text 生成的 prompt
        - pert_prompt:  扰动文本 pert_text 生成的 prompt（若无扰动，则为 None）
        - ground_truth: sentiment
        - orig_text, pert_text: 保留原文，调试/分析用
    """

    def process(row):
        label = row["sentiment"]
        orig = row.get("orig_text", row["text"])
        pert = row.get("pert_text", None)

        # 保证传给 generate_test_prompt 的 row 结构与你之前一样
        row_orig = dict(row)
        row_orig["text"] = orig
        row_orig["sentiment"] = label
        prompt = generate_test_prompt(row_orig)

        pert_prompt = None
        if pert is not None:
            row_pert = dict(row)
            row_pert["text"] = pert
            row_pert["sentiment"] = label
            pert_prompt = generate_test_prompt(row_pert)

        return {
            "prompt": prompt,
            "pert_prompt": pert_prompt,
            "ground_truth": label,
            "orig_text": orig,
            "pert_text": pert,
        }

    hf_ds = Dataset.from_pandas(X_train_raw.reset_index(drop=True))
    hf_ds = hf_ds.map(process)
    return hf_ds

##############################################
# 3. 基础 reward：格式 + 严格准确率
##############################################

def _extract_text_from_completion(completion):
    """兼容 GRPO 返回的多种 completion 结构."""
    # 可能是纯字符串
    if isinstance(completion, str):
        return completion
    # 可能是 list[{"role": "...", "content": "..."}]
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    # 兜底
    return str(completion)


def _extract_label_from_text(text: str, label_order=None):
    if label_order is None:
        label_order = LABEL_ORDER
    low = text.lower()
    for lab in label_order:
        if lab in low:
            return lab
    return None


######################################################################
# 4. 扰动一致性 reward：clean vs perturbed 对称 KL
######################################################################

def consistency_reward_base(
    prompts,
    model,
    tokenizer,
    label_order,
    pert_prompts_list,
):
    """
    对于每个有扰动版本的样本，计算：
        sym_kl = KL(p(x) || p(x~)) + KL(p(x~) || p(x))
    作为一致性惩罚的负号： reward = - sym_kl
    """
    pair_prompts = []
    pair_pert_prompts = []
    idx_map = []

    for i, (p, pp) in enumerate(zip(prompts, pert_prompts_list)):
        if pp is None or pp == "":
            continue
        pair_prompts.append(p)
        pair_pert_prompts.append(pp)
        idx_map.append(i)

    if not pair_prompts:
        return [0.0] * len(prompts)

    with torch.no_grad():
        p_x = get_student_probs_from_prompts(
            model, tokenizer, pair_prompts, label_order
        )
        p_xt = get_student_probs_from_prompts(
            model, tokenizer, pair_pert_prompts, label_order
        )

        kl1 = F.kl_div(
            p_x.clamp_min(1e-12).log(),
            p_xt.clamp_min(1e-12),
            reduction="none",
        ).sum(dim=-1)
        kl2 = F.kl_div(
            p_xt.clamp_min(1e-12).log(),
            p_x.clamp_min(1e-12),
            reduction="none",
        ).sum(dim=-1)

        sym_kl = kl1 + kl2
        vals = (-sym_kl).detach().cpu().tolist()

    rewards = [0.0] * len(prompts)
    for i, v in zip(idx_map, vals):
        rewards[i] = v
    return rewards

def build_finbert_teacher(finbert_model_name, label_order):
    """
    返回一个 teacher_probs_fn: texts -> [B, 3] 概率（顺序与 label_order 对齐）
    FinBERT 从默认 HF 缓存 (~/.cache/huggingface/hub) 加载，不再使用自定义 cache_dir。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔧 [FinBERT] 尝试从本地缓存加载: {finbert_model_name}")
    try:
        finbert_tokenizer = AutoTokenizer.from_pretrained(
            finbert_model_name,
            local_files_only=True,   # ⭐ 只用本地缓存，不联网
        )
        finbert_model = AutoModelForSequenceClassification.from_pretrained(
            finbert_model_name,
            local_files_only=True,
        ).to(device)
        finbert_model.eval()
    except Exception as e:
        print("❌ [FinBERT] 加载失败，将禁用 FinBERT reward。")
        print("   Error:", repr(e))
        return None

    # 根据 FinBERT 的 id2label 自动对齐到 ["negative", "neutral", "positive"]
    id2label = {int(k): v.lower() for k, v in finbert_model.config.id2label.items()}
    index_order = []
    for lab in label_order:
        matched = [i for i, name in id2label.items() if lab in name]
        if not matched:
            raise ValueError(f"Cannot find label containing '{lab}' in FinBERT id2label: {id2label}")
        index_order.append(matched[0])

    def teacher_probs_fn(texts):
        if isinstance(texts, str):
            texts_list = [texts]
        else:
            texts_list = list(texts)

        toks = finbert_tokenizer(
            texts_list,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = finbert_model(**toks).logits
            probs = logits.softmax(dim=-1)  # [B, num_labels(finbert)]
            probs = probs[:, index_order]  # 重新排序成 [neg, neu, pos]

        return probs

    print("✅ [FinBERT] 本地缓存加载成功。")
    return teacher_probs_fn

def finbert_reward_base(
    prompts,
    model,
    tokenizer,
    label_order,
    teacher_probs_fn,
    text_list,
):
    """
    用 FinBERT teacher 做分布对齐：
        reward = - [ KL(p_teacher || p_student) + KL(p_student || p_teacher) ]
    """
    if teacher_probs_fn is None:
        return [0.0] * len(prompts)

    with torch.no_grad():
        p_teacher = teacher_probs_fn(text_list)  # [B,3]
        p_student = get_student_probs_from_prompts(
            model, tokenizer, prompts, label_order
        )

        p_teacher = p_teacher.clamp_min(1e-12)
        p_student = p_student.clamp_min(1e-12)

        kl_ts = F.kl_div(
            p_teacher.log(), p_student, reduction="none"
        ).sum(dim=-1)
        kl_st = F.kl_div(
            p_student.log(), p_teacher, reduction="none"
        ).sum(dim=-1)

        sym_kl = kl_ts + kl_st
        rewards = (-sym_kl).detach().cpu().tolist()
    return rewards


def gt_logprob_reward_base(
    prompts,
    ground_truth,
    model,
    tokenizer,
    label_order,
):
    """
    Ground-truth log-prob reward:
        r_i = log p_theta(y_true | x_i)
    然后在一个 batch 内做标准化（减均值 / 除标准差）。

    prompts:      list[str]
    ground_truth: list[str] 或 list[int]，比如 "negative" / 0
    """
    # 1) 先用你已有的 helper 算出对三个 label 的分布
    with torch.no_grad():
        probs = get_student_probs_from_prompts(
            model, tokenizer, prompts, label_order
        )  # [B, 3]

    # 2) 构建 label -> index 映射
    label2idx = {lab: i for i, lab in enumerate(label_order)}

    base_r = []
    for p_vec, y in zip(probs, ground_truth):
        lab = str(y).strip().lower()

        # ground_truth 既可能是 "negative"，也可能是 0/1/2
        idx = None
        if lab.isdigit():
            idx_int = int(lab)
            if 0 <= idx_int < len(label_order):
                idx = idx_int
        else:
            idx = label2idx.get(lab, None)

        if idx is None:
            # 找不到就给个 0，当这个样本没贡献
            base_r.append(0.0)
            continue

        # 防止 log(0)
        p = float(p_vec[idx].clamp(min=1e-12))
        base_r.append(math.log(p))

    if len(base_r) == 0:
        return [0.0] * len(prompts)

    # 3) 在 batch 内做标准化
    mean = float(np.mean(base_r))
    std = float(np.std(base_r))
    if std < 1e-8:
        std = 1.0  # 避免除零，等价于只减均值

    normed = [(r - mean) / std for r in base_r]
    return normed


############################################################
# 6. GRPO 主入口：main 里直接调用这个
############################################################

def run_grpo_trl(
    data_path,
    sft_lora_path,
    base_model_path,
    cache_dir,
    output_dir="./outputs/grpo_output",
    perturb_data=True,
    use_finbert=True,
    finbert_model_name="ProsusAI/finbert",
    w_gt: float = 1.0,
    w_fin: float = 1.0,
    w_cons: float = 1.0,
    w_sft_kl: float = 0.1,
    resume: bool = False
):
    """
    data_path: CSV 路径，比如 "data/all-data.csv"
    sft_lora_path: 你 SFT 训练保存的 LoRA 目录
    base_model_path: LLAMA_MODEL_NAME
    cache_dir: CACHE_DIR
    output_dir: GRPO 输出目录
    perturb_data: 是否使用 clean+perturbed 成对数据
    use_finbert: 是否在内部构建 FinBERT teacher
    w_fin: FinBERT reward 权重
    w_cons: 一致性 reward 权重
    """
    # 1) 加载并转换数据
    print("📦 Loading & splitting raw data for GRPO ...")
    X_train_raw, X_test_raw, X_eval_raw = load_split_raw_data(
        data_path,
        perturb_data=perturb_data,
    )

    print("🔧 Converting data to HF dataset...")
    train_dataset = convert_to_hf_dataset(X_train_raw)

    # 2) 加载 tokenizer（必须跟 SFT 阶段一致）
    print("🔧 Loading tokenizer from SFT checkpoint...")
    tokenizer = AutoTokenizer.from_pretrained(
        sft_lora_path,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 3) 加载 base LLaMA + SFT LoRA （改成 4bit QLoRA 风格）
    print("🔧 Loading base LLaMA model in 4bit (QLoRA style)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # ✅ 统一用 fp16
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        local_files_only=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,  # ✅ 明确告诉 HF 用 fp16
        device_map="auto",
    )

    # 保证 vocab size 与 SFT/GRPO 使用的 tokenizer 一致（避免 32000 vs 32002 问题）
    if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        base_model.resize_token_embeddings(len(tokenizer))

    print("🔧 Loading SFT LoRA adapter...")
    model = PeftModel.from_pretrained(
        base_model,
        sft_lora_path,
        is_trainable=True,
    )

    # ================================
    # ⭐ 冻结的 SFT teacher，用于 KL 正则
    # ================================
    print("🔧 Loading frozen SFT teacher for KL regularization...")
    teacher_base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        local_files_only=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if teacher_base_model.get_input_embeddings().num_embeddings != len(tokenizer):
        teacher_base_model.resize_token_embeddings(len(tokenizer))

    sft_teacher_model = PeftModel.from_pretrained(
        teacher_base_model,
        sft_lora_path,
        is_trainable=False,  # 不训练
    )
    sft_teacher_model.eval()
    for p in sft_teacher_model.parameters():
        p.requires_grad_(False)

    label_order = LABEL_ORDER
    teacher_probs_fn = None

    if teacher_probs_fn is None and use_finbert:
        print(f"🔧 Loading FinBERT teacher model: {finbert_model_name}")
        teacher_probs_fn = build_finbert_teacher(
            finbert_model_name=finbert_model_name,
            label_order=label_order,
        )

    # 5) GRPO 配置
    # grpo_args = GRPOConfig(
    #     output_dir=output_dir,
    #     learning_rate=5e-6,
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=2,
    #     num_generations=4,
    #     max_prompt_length=512,
    #     max_completion_length=4,
    #     num_train_epochs=1,
    #     logging_steps=20,
    #     fp16=True,
    #     bf16=False,
    #     report_to="none",
    #     save_steps=200,     # 每 200 step 保存一次 checkpoint
    #     save_total_limit=3  # 只保留 3 个 checkpoint
    # )
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,  # 保留就行，反正梯度≈0
        per_device_train_batch_size=2,  # ↓ 调小，省显存
        gradient_accumulation_steps=2,  # ↓ 不用累积了，反正只是对照实验
        num_generations=4,  # ↓ 每个 prompt 只采样 1 个 completion，就够了
        max_prompt_length=256,  # ↓ prompt 截断短一点，加快 forward
        max_completion_length=4,  # 维持一个很小的 completion 长度即可
        num_train_epochs=1,  # 只跑 1 个 epoch
        logging_steps=50,  # 日志不用太频繁
        fp16=True,
        bf16=False,
        report_to="none",
        save_steps=10_000,  # 远大于总 step → 中途基本不会 save
        save_total_limit=1  # 只保留最后一个 checkpoint 就行
    )

    # 6) 定义最终用到的 reward 组合

    import numpy as np
    def gt_logprob_reward(prompts, completions, ground_truth, **kwargs):
        """
        使用 log p(y_true | x) 作为 reward，再乘以 w_gt。
        注意：不依赖 completions，只依赖当前策略的分布。
        """
        base_r = gt_logprob_reward_base(
            prompts=prompts,
            ground_truth=ground_truth,
            model=model,
            tokenizer=tokenizer,
            label_order=label_order,
        )
        return [w_gt * r for r in base_r]

    def finbert_reward(prompts, completions, ground_truth, orig_text=None, **kwargs):
        if teacher_probs_fn is None or w_fin is None or w_fin <= 0:
            return [0.0] * len(prompts)
        if orig_text is None:
            orig_text = prompts  # fallback

        base_r = finbert_reward_base(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            label_order=label_order,
            teacher_probs_fn=teacher_probs_fn,
            text_list=orig_text,
        )  # list[float], 通常是负的

        # ⭐ 每个 batch 内做一次标准化：mean=0, std=1
        m = float(np.mean(base_r))
        s = float(np.std(base_r)) + 1e-8
        normed = [(r - m) / s for r in base_r]

        return [w_fin * r for r in normed]

    def consistency_reward(prompts, completions, ground_truth, pert_prompt=None, **kwargs):
        if not perturb_data or pert_prompt is None or w_cons is None or w_cons <= 0:
            return [0.0] * len(prompts)

        base_r = consistency_reward_base(
            prompts=prompts,
            model=model,
            tokenizer=tokenizer,
            label_order=label_order,
            pert_prompts_list=pert_prompt,
        )  # list[float], 多半也是负的

        # 可能有一部分全 0（没有扰动），可以检查一下是否全相同
        if len(set(base_r)) <= 1:
            # 没有差异，就算了，直接当 0 贡献
            return [0.0] * len(prompts)

        m = float(np.mean(base_r))
        s = float(np.std(base_r)) + 1e-8
        normed = [(r - m) / s for r in base_r]

        return [w_cons * r for r in normed]

        # ==========================================
        # ⭐ 新增：SFT KL 正则 reward：- KL(student || SFT_teacher)
        # ==========================================

    def sft_kl_reward(prompts, completions, ground_truth, **kwargs):
        """
        KL-to-SFT:
            reward = - w_sft_kl * KL( p_student || p_sft_teacher )
        直观：不希望当前策略偏离 SFT 过远。
        """
        if w_sft_kl is None or w_sft_kl <= 0.0:
            return [0.0] * len(prompts)

        with torch.no_grad():
            p_student = get_student_probs_from_prompts(
                model, tokenizer, prompts, label_order
            )
            p_teacher = get_student_probs_from_prompts(
                sft_teacher_model, tokenizer, prompts, label_order
            )

            p_student = p_student.clamp_min(1e-12)
            p_teacher = p_teacher.clamp_min(1e-12)

            # KL(p_student || p_teacher)
            kl_st = F.kl_div(
                p_student.log(), p_teacher, reduction="none"
            ).sum(dim=-1)  # [B]

            rewards = (-w_sft_kl * kl_st).detach().cpu().tolist()
        return rewards

    reward_funcs = [
        gt_logprob_reward
    ]

    # 只要 use_finbert 且 w_fin>0 且 teacher 真的加载成功，就启用 FinBERT reward
    if use_finbert and (w_fin is not None and w_fin > 0) and teacher_probs_fn is not None:
        reward_funcs.append(finbert_reward)

    # 只要有扰动数据且 w_cons>0，就启用一致性 reward
    if perturb_data and (w_cons is not None and w_cons > 0):
        reward_funcs.append(consistency_reward)

    # ⭐ 新增：SFT KL 正则
    if w_sft_kl is not None and w_sft_kl > 0.0:
        reward_funcs.append(sft_kl_reward)

    # 7) 构建 GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=grpo_args,
        train_dataset=train_dataset,
    )

    print("🔥 Starting GRPO training...")

    if resume:
        ckpts = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        ckpts = sorted(ckpts, key=os.path.getmtime)
        if len(ckpts) > 0:
            last_ckpt = ckpts[-1]
            print(f"🔁 Found checkpoint: {last_ckpt}, resuming from it...")
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            print("⚠️ Asked to resume but no checkpoint found, training from scratch...")
            trainer.train()
    else:
        print("🆕 Forced fresh run (ignore checkpoints).")
        trainer.train()

    print("💾 Saving GRPO-tuned model...")
    trainer.save_model(output_dir)

    return trainer

