import warnings
warnings.filterwarnings("ignore")

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

import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel

from configs import peft_config, training_arguments, CACHE_DIR, LLAMA_MODEL_NAME, FINBERT_DIR
from data_utils.dataset_build import load_and_split_data, build_clean_and_perturbed_test
from data_utils.evaluation import evaluate, compute_flip_rate, compute_sym_kl
from models.load_llama import load_llama
from models.predict_llama import predict
from training.sft_trainer import run_sft
from data_utils.match_sft_path import find_latest_sft_dir
from training.run_grpo_trl import run_grpo_trl


def main():
    # ======== 1️⃣ 参数解析：SFT / GRPO / GRPO-EVAL / baseline ========
    parser = argparse.ArgumentParser(description="Run sentiment classification pipeline")
    parser.add_argument("--run_sft", action="store_true", help="Run supervised fine-tuning (LoRA)")
    parser.add_argument("--run_grpo", action="store_true", help="Run GRPO fine-tuning (policy optimization)")
    parser.add_argument("--resume", action="store_true", help="Resume GRPO from checkpoint")
    parser.add_argument("--eval_grpo", action="store_true", help="Load latest GRPO model and evaluate only")

    args = parser.parse_args()

    # ======== 2️⃣ baseline 用的数据（统一 clean）========
    # baseline 我们就用干净数据，简单清晰
    X_train, X_test, X_eval, y_true = load_and_split_data(
        "data/all-data.csv",
        perturb_data=False,   # ✅ baseline: clean only
    )
    print("Columns:", X_train.columns.tolist())
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    # ============================================================
    #   3️⃣ SFT（永远用 clean data 训练）
    # ============================================================
    if args.run_sft:
        print("\n================ SFT (LoRA) MODE ================\n")
        print("🧹 SFT training will use CLEAN training data only.\n")

        # SFT 总是用 clean
        X_train_clean, X_test_clean, X_eval_clean, y_true_clean = load_and_split_data(
            "data/all-data.csv",
            perturb_data=False,
        )
        print("Columns (SFT X_train):", X_train_clean.columns.tolist())
        train_data = Dataset.from_pandas(X_train_clean)
        eval_data = Dataset.from_pandas(X_eval_clean)

        # SFT checkpoint 命名统一用 "clean"
        sft_mode_tag = "clean"

        # 查找已有 CLEAN SFT checkpoint
        try:
            latest_sft_dir = find_latest_sft_dir(
                model_name=LLAMA_MODEL_NAME,
                mode_tag=sft_mode_tag,
                base_dir="./outputs/sft",
            )
            print(f"🧭 Found existing CLEAN SFT checkpoint: {latest_sft_dir}")
        except FileNotFoundError:
            latest_sft_dir = None
            print("🧭 No existing CLEAN SFT checkpoint found, will train a new one.\n")

        # === 3.1 训练或复用 SFT ===
        model, tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)

        if latest_sft_dir is not None:
            print(f"✅ Reusing CLEAN SFT checkpoint: {latest_sft_dir}")
            finetuned_model_dir = latest_sft_dir
        else:
            time_tag = datetime.now().strftime("%Y%m%d")
            sft_root = Path("./outputs/sft")
            sft_root.mkdir(parents=True, exist_ok=True)

            run_name = f"sft_{LLAMA_MODEL_NAME.split('/')[-1]}_{time_tag}_{sft_mode_tag}"
            sft_run_dir = sft_root / run_name

            training_arguments.output_dir = str(sft_run_dir)
            print(f"📁 SFT model will be saved to: {training_arguments.output_dir}\n")

            trainer = run_sft(
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                eval_data=eval_data,
                training_args=training_arguments,
                peft_config=peft_config
            )
            trainer.train()
            trainer.save_model()
            tokenizer.save_pretrained(sft_run_dir)
            print("✅ SFT training finished!\n")

            finetuned_model_dir = str(sft_run_dir)

        # === 3.2 加载 SFT LoRA 模型并 merge ===
        print("→ Loading fine-tuned LoRA model for evaluation...")
        compute_dtype = torch.float16
        print("微调模型位置：" + finetuned_model_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_MODEL_NAME,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=True,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            finetuned_model_dir,
            torch_dtype=compute_dtype,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        merged_model = model.merge_and_unload()

        time_tag = datetime.now().strftime("%Y%m%d")
        merged_root = Path("./outputs/merged")
        merged_root.mkdir(parents=True, exist_ok=True)
        merged_run_dir = merged_root / f"merged_{LLAMA_MODEL_NAME.split('/')[-1]}_{time_tag}_{sft_mode_tag}"

        merged_model.save_pretrained(
            merged_run_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
        tokenizer.save_pretrained(merged_run_dir)

        print(f"📁 Merged model saved to: {merged_run_dir}")

        # === 3.3 SFT：在 CLEAN test 上评估 ===
        print("\n→ [SFT] Evaluating on CLEAN test set ...")
        # preds_clean = predict(X_test_clean, merged_model, tokenizer)
        preds_clean, probs_clean = predict(X_test_clean_eval, grpo_model, grpo_tokenizer, return_probs=True)
        print("🔹 [SFT | CLEAN] Metrics:")
        evaluate(y_true_clean, preds_clean)

        # === 3.4 SFT：在 PERTURBED test 上评估 ===
        print("\n→ [SFT] Building CLEAN + PERTURBED test sets for robustness eval ...")
        X_test_clean2, y_true_clean2, X_test_pert, y_true_pert = build_clean_and_perturbed_test(
            "data/all-data.csv"
        )

        print("→ [SFT] Evaluating on PERTURBED test set ...")
        # preds_pert = predict(X_test_pert, merged_model, tokenizer)
        preds_pert, probs_pert = predict(X_test_clean_eval, grpo_model, grpo_tokenizer, return_probs=True)
        print("🔹 [SFT | PERTURBED] Metrics:")
        evaluate(y_true_pert, preds_pert)

        flip_rate = compute_flip_rate(preds_clean, preds_pert)
        sym_kl = compute_sym_kl(probs_clean, probs_pert)

        print(f"🔸 Flip Rate (clean vs perturbed): {flip_rate:.4f}")
        print(f"🔸 Symmetric KL (clean vs perturbed): {sym_kl:.4f}")

        return  # 结束 SFT 模式

    # ============================================================
    #   4️⃣ GRPO 训练（训练时必须用 perturb）
    # ============================================================
    if args.run_grpo:
        print("\n================ GRPO MODE ================\n")
        print("🧪 GRPO training will use PERTURBED data (plus clean) for robustness rewards.\n")

        # 1️⃣ 找 CLEAN SFT checkpoint（GRPO 的起点）
        sft_mode_tag = "clean"
        try:
            latest_sft_dir = find_latest_sft_dir(
                model_name=LLAMA_MODEL_NAME,
                mode_tag=sft_mode_tag,
                base_dir="./outputs/sft",
            )
            print(f"🧭 Using CLEAN SFT checkpoint for GRPO init: {latest_sft_dir}")
        except FileNotFoundError:
            raise RuntimeError(
                "No CLEAN SFT checkpoint found. Please run with --run_sft first."
            )

        # 2️⃣ GRPO 输出目录（按日期）
        time_tag = datetime.now().strftime("%Y%m%d")
        grpo_root = Path("./outputs/grpo")
        grpo_root.mkdir(parents=True, exist_ok=True)
        grpo_run_dir = grpo_root / f"grpo_{LLAMA_MODEL_NAME.split('/')[-1]}_{time_tag}"
        print(f"→ [GRPO] Output dir: {grpo_run_dir}")

        w_gt = 0.0
        w_fin = 0.2
        w_cons = 0.3
        w_sft_kl = 0.0


        # 3️⃣ 调用 GRPO 训练（内部处理 resume / save）
        print("→ [GRPO] Training with perturb_data=True (using clean+perturbed pairs)...")
        trainer = run_grpo_trl(
            data_path="data/all-data.csv",
            sft_lora_path=latest_sft_dir,  # 起点 = clean-SFT
            base_model_path=LLAMA_MODEL_NAME,
            cache_dir=CACHE_DIR,
            output_dir=str(grpo_run_dir),
            perturb_data=True,  # 干净 + 扰动 成对数据
            use_finbert=True,
            finbert_model_name="ProsusAI/finbert",
            w_gt=w_gt,
            w_fin=w_fin,
            w_cons=w_cons,
            w_sft_kl=w_sft_kl,  # 现在先全部 0，排除 reward 影响
            resume=args.resume,
        )

        print("\n================ GRPO Hyperparameters ================")
        print(f"  w_gt      = {w_gt}")
        print(f"  w_fin     = {w_fin}")
        print(f"  w_cons    = {w_cons}")
        print(f"  w_sft_kl  = {w_sft_kl}")
        print("======================================================\n")

        print(f"\n✅ GRPO fine-tuning done. Output saved to: {grpo_run_dir}\n")

        # 4️⃣ 训练用完就把 trainer/model 释放掉，防止显存 & 状态影响
        del trainer
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # 5️⃣ 像 SFT eval 一样重新加载 base LLaMA + tokenizer
        print("→ [GRPO] Reloading base model + tokenizer for eval (aligned with SFT eval)...")
        base_model, grpo_tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)
        # load_llama 里已经：
        # - 用 BitsAndBytes 4bit + nf4
        # - 设置 pad_token / padding_side
        # - setup_chat_format(model, tokenizer)

        # 6️⃣ 挂载 GRPO LoRA adapter
        print("→ [GRPO] Attaching GRPO LoRA adapter for eval ...")
        grpo_model = PeftModel.from_pretrained(
            base_model,
            grpo_run_dir,
            is_trainable=False,
        )
        grpo_model.eval()
        print("✅ GRPO eval model loaded.\n")

        # 7️⃣ CLEAN + PERTURBED 评估（完全复用 SFT 那套 predict / evaluate）
        print("\n→ [GRPO] Building CLEAN + PERTURBED test sets for robustness eval ...")
        X_test_clean_eval, y_true_clean_eval, X_test_pert_eval, y_true_pert_eval = build_clean_and_perturbed_test(
            "data/all-data.csv"
        )

        print("→ [GRPO-EVAL] Evaluating CLEAN test set ...")
        preds_clean, probs_clean = predict(X_test_clean_eval, grpo_model, grpo_tokenizer, return_probs=True)
        print("🔹 [GRPO-EVAL | CLEAN] Metrics:")
        evaluate(y_true_clean_eval, preds_clean)

        print("\n→ [GRPO-EVAL] Evaluating PERTURBED test set ...")
        preds_pert, probs_pert = predict(X_test_pert_eval, grpo_model, grpo_tokenizer, return_probs=True)
        print("🔹 [GRPO-EVAL | PERTURBED] Metrics:")
        evaluate(y_true_pert_eval, preds_pert)

        flip_rate = compute_flip_rate(preds_clean, preds_pert)
        sym_kl = compute_sym_kl(probs_clean, probs_pert)

        print(f"🔸 Flip Rate (clean vs perturbed): {flip_rate:.4f}")
        print(f"🔸 Symmetric KL (clean vs perturbed): {sym_kl:.4f}")

        return  # 结束 GRPO 训练模式

    # ============================================================
    #   5️⃣ 只评估 GRPO（不训练，用最新一次 GRPO）
    # ============================================================
    if args.eval_grpo:
        print("\n================ GRPO EVAL MODE ================\n")

        grpo_root = Path("./outputs/grpo")
        if not grpo_root.exists():
            raise RuntimeError("No GRPO outputs found in ./outputs/grpo. Please run --run_grpo first.")

        grpo_runs = sorted(grpo_root.glob("grpo_*"))
        if not grpo_runs:
            raise RuntimeError("No GRPO run directories found. Please run --run_grpo first.")

        # 最新一次实验
        latest_grpo_dir = grpo_runs[-1]
        grpo_run_dir = str(latest_grpo_dir)
        print(f"📂 Using latest GRPO dir: {grpo_run_dir}")

        # 1️⃣ 和 SFT / Baseline 完全一样：用 load_llama 重新加载 base + tokenizer
        print("→ [GRPO-EVAL] Reloading base model + tokenizer (aligned with SFT eval)...")
        base_model, tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)
        # load_llama 里面已经：
        #   - 用 BitsAndBytes 4bit + nf4
        #   - 设置 pad_token / padding_side
        #   - setup_chat_format(model, tokenizer)

        # 2️⃣ 挂载 GRPO LoRA adapter
        print("→ [GRPO-EVAL] Attaching GRPO LoRA adapter...")
        grpo_model = PeftModel.from_pretrained(
            base_model,
            grpo_run_dir,
            is_trainable=False,
        )
        grpo_model.eval()
        print("✅ GRPO eval model loaded.\n")

        # 3️⃣ CLEAN / PERTURBED 评估（复用和 SFT 一样的 pipeline）
        print("→ [GRPO-EVAL] Building CLEAN + PERTURBED test sets ...")
        X_test_clean_eval, y_true_clean_eval, X_test_pert_eval, y_true_pert_eval = build_clean_and_perturbed_test(
            "data/all-data.csv"
        )

        print("→ [GRPO-EVAL] Evaluating CLEAN test set ...")
        preds_clean, probs_clean = predict(X_test_clean_eval, grpo_model, tokenizer, return_probs=True)
        print("🔹 [GRPO-EVAL | CLEAN] Metrics:")
        evaluate(y_true_clean_eval, preds_clean)

        print("\n→ [GRPO-EVAL] Evaluating PERTURBED test set ...")
        preds_pert, probs_pert = predict(X_test_pert_eval, grpo_model, tokenizer, return_probs=True)
        print("🔹 [GRPO-EVAL | PERTURBED] Metrics:")
        evaluate(y_true_pert_eval, preds_pert)

        flip_rate = compute_flip_rate(preds_clean, preds_pert)
        sym_kl = compute_sym_kl(probs_clean, probs_pert)

        print(f"🔸 Flip Rate (clean vs perturbed): {flip_rate:.4f}")
        print(f"🔸 Symmetric KL (clean vs perturbed): {sym_kl:.4f}")

        print("\n🎉 GRPO Evaluation Finished.\n")
        return  # 结束 GRPO 评估模式

    # ============================================================
    #   6️⃣ Baseline（保持简单：clean 训练 + clean test）
    # ============================================================
    print("\n================ BASELINE MODE ================\n")
    print("🧹 Baseline uses CLEAN training data only.\n")
    model, tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)
    preds = predict(X_test, model, tokenizer)
    print("🔹 [Baseline | CLEAN] Metrics:")
    evaluate(y_true, preds)
    print("\n✅ Baseline evaluation complete.\n")



if __name__ == "__main__":
    main()
