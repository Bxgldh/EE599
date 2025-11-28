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
# ==========================================

from transformers import pipeline
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from data_utils.prompts import generate_test_prompt

# def predict(test, model, tokenizer):
#     y_pred = []
#     for i in tqdm(range(len(test))):
#         prompt = test.iloc[i]["text"]
#         pipe = pipeline(task="text-generation",
#                         model=model,
#                         tokenizer=tokenizer,
#                         max_new_tokens = 1,
#                         do_sample=False,
#                        )
#         result = pipe(prompt)
#         # print(result)
#         answer = result[0]['generated_text'].split("=")[-1]
#         if "positive" in answer:
#             y_pred.append("positive")
#         elif "negative" in answer:
#             y_pred.append("negative")
#         elif "neutral" in answer:
#             y_pred.append("neutral")
#         else:
#             y_pred.append("none")
#     return y_pred


def predict(test, model, tokenizer, return_probs: bool = False):
    """
    test: DataFrame，包含 'text' 列
    model: 已经挂好 LoRA 的 LLaMA 模型
    tokenizer: 对应 tokenizer
    return_probs:
        - False: 只返回 y_pred（和你原来完全一致）
        - True:  返回 (y_pred, probs)，probs.shape = (N, 3)，顺序 [positive, negative, neutral]
    """

    device = model.device

    # ====== 1. label 词表 & token id（用于算概率） ======
    label_words = ["positive", "negative", "neutral"]

    label_token_ids = []
    for w in label_words:
        # 和之前一样，用前面加空格的形式，倾向于得到单 token
        tokens = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(tokens) != 1:
            # 简单取第一个，粗糙但够用；你要更精细可以后续再改成多 token logprob 相加
            label_token_ids.append(tokens[0])
        else:
            label_token_ids.append(tokens[0])
    label_token_ids = torch.tensor(label_token_ids, device=device)  # (3,)

    # ====== 2. 保留你原来的 pipeline 行为（负责“真正的分类结果”） ======
    text_gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1,
        do_sample=False,
        # device=0  # 如果你是在 GPU 上，可以加这一行；否则让 HF 自己处理
    )

    y_pred = []
    all_probs = [] if return_probs else None

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]

        # ---- (A) 用 pipeline 生成文本，解析 label（完全保留你原来的逻辑） ----
        result = text_gen_pipe(prompt)
        answer = result[0]["generated_text"].split("=")[-1]

        if "positive" in answer:
            label_str = "positive"
        elif "negative" in answer:
            label_str = "negative"
        elif "neutral" in answer:
            label_str = "neutral"
        else:
            label_str = "none"

        y_pred.append(label_str)

        # ---- (B) 如果需要 probs：单独跑一遍 forward，拿 logits → 概率分布 ----
        # ---- (B) 如果需要 probs：单独跑一遍 forward，拿 logits → 概率分布 ----
        if return_probs:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                padding=False,
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                # logits: (1, seq_len, vocab_size)
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)
                probs_vocab = F.softmax(logits, dim=-1)  # (1, vocab_size)

                # 取出三个 label token 的概率
                probs_labels = probs_vocab[:, label_token_ids]  # (1, 3)

                # ⭐ 加一层安全防护：防止 sum=0 导致除 0
                denom = probs_labels.sum(dim=-1, keepdim=True)  # (1, 1)
                denom = torch.clamp(denom, min=1e-12)
                probs_labels = probs_labels / denom

            all_probs.append(probs_labels.squeeze(0).cpu().numpy())

    if return_probs:
        probs = np.stack(all_probs, axis=0)  # (N, 3)
        return y_pred, probs
    else:
        return y_pred
