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
from data_utils.prompts import generate_test_prompt

def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["text"]
        pipe = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_new_tokens = 1, 
                        do_sample=False,
                       )
        result = pipe(prompt)
        # print(result)
        answer = result[0]['generated_text'].split("=")[-1]
        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred
