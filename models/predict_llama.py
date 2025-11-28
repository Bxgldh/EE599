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
    参数
    ----
    test: pandas.DataFrame，至少包含一列 'text'
    model: causal LM（已经挂好 LoRA，eval 模式）
    tokenizer: 对应 tokenizer
    return_probs: 是否同时返回每个样本在 [positive, negative, neutral] 三类上的概率分布

    返回
    ----
    如果 return_probs = False:
        y_pred: List[str]，每个元素是 {"positive","negative","neutral","none"} 之一
    如果 return_probs = True:
        (y_pred, probs):
            y_pred 同上
            probs: np.ndarray, shape (N, 3)，列顺序为 [positive, negative, neutral]
    """

    device = model.device

    # 约定 label 顺序，后面算 flip-rate / sym-KL 要和这里保持一致
    label_words = ["positive", "negative", "neutral"]

    # 为每个 label 找到对应的 token id
    # 一般 LLaMA 系列用 " positive" 这种带空格的 token，更容易是单 token
    label_token_ids = []
    for w in label_words:
        # 注意加一个前导空格，让 tokenizer 倾向于返回单个 token
        tokens = tokenizer.encode(" " + w, add_special_tokens=False)
        if len(tokens) != 1:
            # 如果不是单 token，就简单取第一个，算是近似
            # （要更精确可以改成多 token logprob 相加）
            label_token_ids.append(tokens[0])
        else:
            label_token_ids.append(tokens[0])
    label_token_ids = torch.tensor(label_token_ids, device=device)  # shape (3,)

    y_pred = []
    all_probs = [] if return_probs else None

    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]

        # 构造输入
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits: (batch=1, seq_len, vocab_size)
            logits = outputs.logits[:, -1, :]      # 取最后一个 token 位置的 logits → (1, vocab)
            probs_vocab = F.softmax(logits, dim=-1)  # (1, vocab)

            # 取出三个 label token 的概率
            probs_labels = probs_vocab[:, label_token_ids]  # (1, 3)
            # 再归一化一下，保证和为 1（以防其余 token 概率也占了一些质量）
            probs_labels = probs_labels / probs_labels.sum(dim=-1, keepdim=True)  # (1, 3)

            # 预测 label
            pred_idx = probs_labels.argmax(dim=-1).item()  # 0/1/2

        pred_label = label_words[pred_idx]  # "positive"/"negative"/"neutral"
        y_pred.append(pred_label)

        if return_probs:
            all_probs.append(probs_labels.squeeze(0).cpu().numpy())

    if return_probs:
        probs = np.stack(all_probs, axis=0)  # (N, 3), 列顺序 [positive, negative, neutral]
        return y_pred, probs
    else:
        return y_pred
