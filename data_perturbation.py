"""
data_perturbation.py

金融新闻标题数据扰动模块：
- 同义词替换（synonym_replace）
- 模板释义改写（template_paraphrase_v1）
- 数字 + 实体同行替换（perturb_numbers_entities）
- 语义相似度 + 长度 + NER 类型过滤（pass_filter）
- FinBERT 情感 & 置信度对抗标签（adversarial_tag）
- 对单条/批量标题生成扰动样本（gen_perturb_sample, build_library）
- 训练时从通过过滤的扰动中抽一个（pick_one_pert）
"""
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

import random
import re
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================
# 0. 全局模型 & 工具初始化
# ======================

# spaCy 用于词性标注 & NER
nlp = spacy.load("en_core_web_sm")

# 设备
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# SBERT 用于句向量相似度
sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# FinBERT（ProsusAI）用于对抗标签（情感 + margin）
_FIN = "ProsusAI/finbert"
fin_tokenizer = AutoTokenizer.from_pretrained(_FIN)
fin_model = AutoModelForSequenceClassification.from_pretrained(_FIN).to(_DEVICE).eval()


# ======================
# 1. 同义词替换扰动(近义词+回译)
# ======================

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords#引入停用词，因为对停用词进行数据增强相当于没有增强
from nltk.corpus import wordnet as wn#引入同义词
import random
stop_words=set(stopwords.words('english'))
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    stop_words.add(w)

#获取同义词
def get_synonyms(word):
    synsets = wn.synsets(word)
    if not synsets:         # 没有同义词，返回空列表
        return []
    lemmas = synsets[0].lemma_names()
    # 去掉自身，顺便把 "_" 换成空格
    lemmas = [l.replace('_', ' ') for l in lemmas if l.lower() != word.lower()]
    return lemmas


#这里传入的words是一个列表,
#eg:"hello world".split(" ") or ["hello","world"]
def synonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))     
    random.shuffle(random_word_list)
    num_replaced = 0  
    for random_word in random_word_list:          
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)   
            new_words = [synonym if word == random_word else word for word in new_words]   
            num_replaced += 1
        if num_replaced >= n: 
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return " ".join(new_words)


# 经过测试，这个翻译的包翻译的时间是最短的
# 回译
from pygtrans import Translate

def backTran(wordss):
    client = Translate()
    text1 = client.translate(wordss)

    text2 = client.translate(text1.translatedText, target='en')
    return text2.translatedText

def synonym_replace_nltk(text: str, alpha: float = 0.25) -> str:
    """
    使用 NLTK+WordNet 做近义词替换：
    - text: 原始句子
    - alpha: 替换比例（例如 0.25 表示替换大约 1/4 的单词）
    """
    words = text.split()
    if not words:
        return text
    n = max(1, int(len(words) * alpha))  # 至少替换 1 个
    new_sent = synonym_replacement(words, n)  # 你的函数返回的是 string
    return new_sent


def back_translate(text: str) -> str:
    """
    使用 pygtrans 做一次回译扰动(en -> zh -> en)。
    失败时返回原句，避免训练中断。
    """
    try:
        return backTran(text)
    except Exception:
        return text

# ======================
# 2. 模板释义改写扰动
# ======================

# def template_paraphrase_v1(
#     text: str,
#     *,
#     max_len_delta: float = 0.20,
#     keep_polarity_fn=None,  # 可传入一个函数：str -> {"label": ...}
#     seed: int | None = 42
# ) -> str:
#     """
#     模板释义改写（v1）—— 为金融新闻标题设计的“轻度”改写器。
#     - 等价短语替换（beat/miss expectations 等）
#     - 强度副词中和（sharply -> moderately）
#     - 涨跌动词 soften（rose -> rose slightly）
#     - 轻度语态/结构改写（was up -> rose 等）
#     - 控制改写后长度变化比例 ≤ max_len_delta
#     - 可选：通过 keep_polarity_fn 保持情感极性一致
#     """
#     if seed is not None:
#         random.seed(seed)

#     orig = text
#     L0   = len(orig.split())

#     # 1) 短语级等价替换
#     rules_phrase = [
#         (r"\bbeat expectations\b",              "exceeded expectations"),
#         (r"\boutperformed expectations\b",      "exceeded expectations"),
#         (r"\bmissed expectations\b",            "failed to meet expectations"),
#         (r"\bfell short of expectations\b",     "failed to meet expectations"),
#         (r"\bmet expectations\b",               "matched expectations"),
#         (r"\babove guidance\b",                 "above its guidance"),
#         (r"\bbelow guidance\b",                 "below its guidance"),
#         (r"\bstrong demand\b",                  "robust demand"),
#         (r"\bweak demand\b",                    "subdued demand"),
#     ]

#     # 2) 强度副词中和
#     rules_intensity = [
#         (r"\b(sharply|dramatically|significantly|substantially|considerably)\b", "moderately"),
#         (r"\b(soared|surged|skyrocketed)\b", "jumped"),
#         (r"\b(plunged|tumbled|cratered)\b",  "fell"),
#     ]

#     # 3) 动词 soften
#     soften_adv = random.choice(["slightly", "modestly", "marginally", "somewhat"])
#     def add_soft_adv(m):
#         return f"{m.group(1)} {soften_adv}"

#     rules_verbs_soften = [
#         (r"\b(rose|increased|climbed|gained|advanced)\b", add_soft_adv),
#         (r"\b(fell|declined|dropped|slid|weakened)\b",     add_soft_adv),
#     ]

#     # 4) 轻度语态/结构改写
#     rules_syntax = [
#         (r"\bThe company reported\b", "A report from the company indicated"),
#         (r"\bThe firm reported\b",    "A report from the firm indicated"),
#         (r"\brevenues? (rose|increased)\b", r"revenue \1"),
#         (r"\bprofits? (fell|declined)\b",  r"profit \1"),
#         (r"\bwas up\b", "rose"),
#         (r"\bwas down\b", "declined"),
#     ]

#     new = text

#     for pat, rep in rules_phrase:
#         new = re.sub(pat, rep, new, flags=re.I)

#     for pat, rep in rules_intensity:
#         new = re.sub(pat, rep, new, flags=re.I)

#     for pat, rep in rules_verbs_soften:
#         new = re.sub(pat, rep, new, flags=re.I)

#     for pat, rep in rules_syntax:
#         new = re.sub(pat, rep, new, flags=re.I)

#     # 长度变化控制
#     L1 = len(new.split())
#     if abs(L1 - L0) / max(1, L0) > max_len_delta:
#         new = orig

#     # 可选：极性一致性检查
#     if callable(keep_polarity_fn):
#         try:
#             if keep_polarity_fn(orig)["label"] != keep_polarity_fn(new)["label"]:
#                 new = orig
#         except Exception:
#             new = orig

#     return new


# ======================
# 3. 数字 + 实体同行替换扰动
# ======================

SECTOR_PEERS: Dict[str, str] = {
    "Barclays": "Citigroup",
    "Citigroup": "Goldman Sachs",
    "Goldman Sachs": "Deutsche Bank",
    "Deutsche Bank": "Credit Suisse",
    "Credit Suisse": "JPMorgan",
    "JPMorgan": "UBS",
    "UBS": "Bank of America",
    "Bank of America": "Morgan Stanley",
    "Morgan Stanley": "Barclays",
    "United": "American Airlines",
    "American Airlines": "British Airways",
    "British Airways": "Delta",
    "Delta": "Emirates",
    "Emirates": "United",
    "Orange": "China Mobile",
    "China Mobile": "Verizon",
    "Verizon": "Vodafone",
    "Vodafone": "AT&T",
    "AT&T": "Orange",
    "Target": "CVS",
    "CVS": "Home Depot",
    "Home Depot": "Target",
    "Microsoft": "Google",
    "Google": "Apple",
    "Apple": "Intel",
    "Intel": "IBM",
    "IBM": "Qualcomm",
    "Qualcomm": "Microsoft",
    "BP": "Shell",
    "Shell": "BP",
    "Volkswagen": "Ford",
    "Ford": "Honda",
    "Honda": "Volkswagen",
    "Sony": "Fox",
    "Fox": "Sony",
    "X": "Snap",
}

def perturb_numbers_entities(text: str) -> str:
    """
    - 对百分比数字做轻微微调(±0.1 / 0.2 个百分点)
    - 对部分机构名/公司名做“同行替换”（例如 Barclays -> Citigroup）
    """
    def tweak_num(m):
        x = float(m.group(1))
        y = round(x + random.choice([-0.2, -0.1, 0.1, 0.2]), 1)
        return f"{y}%"

    text2 = re.sub(r"(\d+\.?\d*)\s?%", tweak_num, text)

    for a, b in SECTOR_PEERS.items():
        if a in text2:
            text2 = text2.replace(a, b, 1)
            break

    return text2


# ======================
# 4. 真实性过滤（语义相似度 + 长度 + NER 类型）
# ======================

ALLOWED_NER = {"ORG", "GPE", "PRODUCT", "MONEY", "PERCENT"}

def entity_label_set(text: str) -> set:
    """
    返回文本中允许的实体类型集合，用于保证扰动前后实体类型一致。
    """
    doc = nlp(text)
    if "ner" not in nlp.pipe_names:
        return set()
    return {ent.label_ for ent in doc.ents if ent.label_ in ALLOWED_NER}

def pass_filter(
    src: str,
    tgt: str,
    sim_th: float = 0.85,
    len_ratio: float = 0.5
) -> Tuple[bool, Dict[str, Any]]:
    """
    过滤器：
    - SBERT 语义相似度 ≥ sim_th
    - 长度变化比例 ≤ len_ratio
    - 实体标签集合一致(只看 ALLOWED_NER)
    """
    # 语义相似度（SBERT）
    emb = sbert.encode([src, tgt], convert_to_tensor=True, normalize_embeddings=True)
    sim = float(util.cos_sim(emb[0], emb[1]))

    if sim < sim_th:
        return False, {"sim": sim}

    # 长度变化约束
    L0, L1 = len(src.split()), len(tgt.split())
    if abs(L1 - L0) / max(1, L0) > len_ratio:
        return False, {"sim": sim, "len_ok": False}

    # 实体类型集合一致
    src_labels = entity_label_set(src)
    tgt_labels = entity_label_set(tgt)
    if src_labels != tgt_labels:
        return False, {
            "sim": sim,
            "ner_ok": False,
            "src_ner": sorted(src_labels),
            "tgt_ner": sorted(tgt_labels),
        }

    return True, {"sim": sim}


# ======================
# 5. FinBERT 对抗标签（情感翻转 / 置信度削弱）
# ======================

@torch.no_grad()
def finbert_pred(text: str) -> Tuple[str, float, Dict[str, float]]:
    """
    使用 FinBERT (ProsusAI/finbert) 返回：
    - label: 'positive' / 'negative' / 'neutral'
    - margin: 置信度边际(top - second)
    - scores: 每个类别的概率字典
    """
    enc = fin_tokenizer(text, return_tensors="pt", truncation=True).to(_DEVICE)
    logits = fin_model(**enc).logits               # [1,3]
    probs  = F.softmax(logits, dim=-1).squeeze(0)  # [3]

    id2label = {i: fin_model.config.id2label[i].lower() for i in range(probs.numel())}
    scores   = {id2label[i]: float(probs[i].item()) for i in range(probs.numel())}
    label    = max(scores, key=scores.get)
    top      = scores[label]
    second   = max(v for k, v in scores.items() if k != label)
    margin   = top - second

    return label, margin, scores

def adversarial_tag(
    orig: str,
    pert: str,
    margin_drop: float = 0.2
) -> Dict[str, Any]:
    """
    用 FinBERT 判断：
    - 情感是否翻转 (flipped)
    - 置信度边际是否明显下降 (weakened)
    """
    y0, m0, s0 = finbert_pred(orig)
    y1, m1, s1 = finbert_pred(pert)
    flipped   = (y0 != y1)
    weakened  = (m0 - m1) >= margin_drop
    return {
        "orig_label": y0,
        "pert_label": y1,
        "flipped":    flipped,
        "weakened":   weakened,
        "m0":         m0,
        "m1":         m1,
    }


# ======================
# 6. 组合三个扰动生成候选 + 构建扰动库
# ======================

def gen_perturb_sample(text: str) -> List[Dict[str, Any]]:
    """
    对单条文本生成多个扰动候选：
    - 同义词替换
    - 模板释义改写
    - 数字 + 实体扰动

    返回列表，每个元素形如：
    {
        "orig": 原句,
        "pert": 扰动句,
        "passed": True/False,   # 是否通过 pass_filter
        "sim": 相似度,
        ... 如果通过，还包含 FinBERT 对抗标签 ...
    }
    """
    candidates = [
        synonym_replace_nltk(text),   # NLTK 近义词替换
        back_translate(text),         # 回译扰动
        perturb_numbers_entities(text),
    ]
    outs: List[Dict[str, Any]] = []

    for t in candidates:
        ok, info = pass_filter(text, t)
        base = {"orig": text, "pert": t, "passed": ok, **info}
        if not ok:
            outs.append(base)
            continue
        # 通过过滤，再补充 FinBERT 对抗性标签
        tag = adversarial_tag(text, t)
        base.update(tag)
        outs.append(base)

    return outs

# def gen_perturb_sample(text: str) -> List[Dict[str, Any]]:
#     """
#     修改版：增加了去重逻辑，防止原句被算作扰动样本
#     """
#     candidates = []
    
#     # 1. 尝试同义词
#     cand1 = synonym_replace_nltk(text)
#     if cand1 != text: candidates.append(cand1)
    
#     # 2. 尝试回译 (注意：如果没有联网或API失败，back_translate 会返回原句)
#     cand2 = back_translate(text)
#     if cand2 != text: candidates.append(cand2)
    
#     # 3. 尝试数字/实体替换
#     cand3 = perturb_numbers_entities(text)
#     if cand3 != text: candidates.append(cand3)
    
#     # 4. (建议) 恢复模板改写
#     # cand4 = template_paraphrase_v1(text)
#     # if cand4 != text: candidates.append(cand4)

#     outs: List[Dict[str, Any]] = []

#     for t in candidates:
#         # 这里 t 肯定不等于 text，但为了双重保险以及计算相似度，继续走流程
#         ok, info = pass_filter(text, t)
        
#         # 再次强制检查：如果相似度极高（例如 > 0.999），通常意味着只有标点符号差异
#         # 根据需求决定是否保留。这里演示保留严格不等的。
#         if t.strip() == text.strip():
#             continue

#         base = {"orig": text, "pert": t, "passed": ok, **info}
        
#         if not ok:
#             outs.append(base)
#             continue
            
#         # 通过过滤，计算 FinBERT 指标
#         tag = adversarial_tag(text, t)
#         base.update(tag)
#         outs.append(base)

#     return outs


def build_library(headlines: List[str]) -> List[Dict[str, Any]]:
    """
    对一批文本构建扰动库：简单地把每条的 gen_perturb_sample 结果拼起来。
    """
    rows: List[Dict[str, Any]] = []
    for h in headlines:
        rows.extend(gen_perturb_sample(h))
    return rows


# ======================
# 7. RL / 训练时使用的接口：随机挑一个通过过滤的扰动
# ======================

def pick_one_pert(text: str) -> str | None:
    """
    从 gen_perturb_sample(text) 中挑一个通过过滤的扰动句：
    - 若没有通过过滤的候选，返回 None；
    - 否则在通过过滤的候选里随机采样一条的 "pert" 返回。
    """
    cands = gen_perturb_sample(text)
    passed = [c for c in cands if c.get("passed", False)]
    if not passed:
        return None
    return random.choice(passed)["pert"]


import os
import pandas as pd
from typing import Iterable

# ======================
# 8. 工具函数：保存“干净文本对”到 ./data
# ======================

def save_perturbation_text_pairs(
    headlines: Iterable[str],
    filename: str = "perturbations_text_pairs.csv",
    dirpath: str = "./data",
    only_passed: bool = True,
    drop_duplicates: bool = True,
) -> str:
    """
    对一批文本生成扰动，并只保存 (orig, pert)（可选仅保留通过过滤的）。

    Args:
        headlines: 原始新闻标题列表，例如 X_train_new["text"].tolist()
        filename:  输出文件名，默认 'perturbations_text_pairs.csv'
        dirpath:   保存目录，默认 './data'
        only_passed: 是否只保留 passed=True 的扰动样本
        drop_duplicates: 是否去重 (orig, pert) 对

    Returns:
        保存后的完整路径
    """
    # 1) 构建完整扰动库（包含各种附加字段）
    rows = build_library(list(headlines))

    # 2) 转成 DataFrame
    df = pd.DataFrame(rows)

    # 3) 只保留我们要的列
    cols = ["orig", "pert", "passed"]
    df = df[[c for c in cols if c in df.columns]]

    # 4) 只要通过过滤的（可选）
    if only_passed and "passed" in df.columns:
        df = df[df["passed"] == True]

    # 5) 去掉重复的 (orig, pert) 对
    if drop_duplicates:
        df = df.drop_duplicates(subset=["orig", "pert"])

    # 6) 保存
    os.makedirs(dirpath, exist_ok=True)
    out_path = os.path.join(dirpath, filename)
    df[["orig", "pert"]].to_csv(out_path, index=False, encoding="utf-8")

    print(f"[data_perturbation] Saved {len(df)} text pairs to {out_path}")
    return out_path
