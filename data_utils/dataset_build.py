import os
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from data_utils.prompts import generate_prompt, generate_test_prompt
from data_perturbation import save_perturbation_text_pairs


def load_split_raw_data(
    path,
    train_size=300,
    test_size=300,
    seed=42,
    perturb_data=True,
):
    """
    从 CSV 加载原始数据，并按类别划分 train/test/eval。

    返回：
        X_train: DataFrame，至少包含列：
                 - text       : 训练时用的文本（clean 或 perturb 后的）
                 - sentiment  : 标签
                 - orig_text  : 原始 clean 文本
                 - pert_text  : 对应的扰动文本（若未扰动则与 orig_text 相同）

        X_test:  DataFrame，列：
                 - sentiment
                 - text       （原始 clean 文本）

        X_eval:  DataFrame，列：
                 - sentiment
                 - text       （原始 clean 文本）
    """
    # 1. 读原始 CSV
    df = pd.read_csv(
        path,
        names=["sentiment", "text"],
        encoding="utf-8",
        encoding_errors="replace"
    )

    X_train = []
    X_test = []

    # 2. 按类别 stratified 采样 train/test
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(
            df[df.sentiment == sentiment],
            train_size=train_size,
            test_size=test_size,
            random_state=seed,
        )
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    # 3. 构造 eval 集：剩下的样本里，每类采样 50 条
    eval_idx = [idx for idx in df.index if idx not in list(X_train.index) + list(X_test.index)]
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = (
        X_eval.groupby("sentiment", group_keys=False)
        .apply(lambda x: x.sample(n=50, random_state=10, replace=True))
    )

    # 重置索引
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    X_eval = X_eval.reset_index(drop=True)

    # 4. 训练集：是否做扰动扩增
    if perturb_data:
        # 深拷贝，避免污染原 df
        X_train_new = copy.deepcopy(X_train)

        # ---- 原始干净样本 ----
        train_clean = X_train_new[["text", "sentiment"]].copy()
        train_clean["orig_text"] = train_clean["text"]
        train_clean["pert_text"] = train_clean["text"]

        # ---- 构建扰动对 ----
        headlines = X_train_new["text"].astype(str).tolist()

        if not os.path.exists("./data/train_perturbations_text_pairs.csv"):
            save_perturbation_text_pairs(
                headlines,
                filename="train_perturbations_text_pairs.csv"
            )

        pert_df = pd.read_csv("./data/train_perturbations_text_pairs.csv")
        # 期望列: ["orig", "pert"]

        # 用原始 X_train_new 的标签映射给 orig
        label_map = dict(zip(X_train_new["text"], X_train_new["sentiment"]))
        pert_df["sentiment"] = pert_df["orig"].map(label_map)

        # ---- 扰动样本部分 ----
        train_aug = pert_df.rename(columns={"pert": "text"})[
            ["text", "sentiment", "orig"]
        ].copy()
        train_aug = train_aug.rename(columns={"orig": "orig_text"})
        train_aug["pert_text"] = train_aug["text"]

        # ---- 合并干净样本 + 扰动样本 ----
        X_train = pd.concat([train_clean, train_aug], ignore_index=True)
    else:
        # 不做扰动扩增，但也补上 orig_text / pert_text，便于 GRPO 统一处理
        X_train = X_train.copy()
        X_train["orig_text"] = X_train["text"]
        X_train["pert_text"] = X_train["text"]

    return X_train, X_test, X_eval


def load_and_split_data(path, train_size=300, test_size=300, seed=42, perturb_data=True):
    X_train, X_test, X_eval = load_split_raw_data(
        path,
        train_size=train_size,
        test_size=test_size,
        seed=seed,
        perturb_data=perturb_data,
    )

    # ===== 下面是 SFT / baseline 专用的 prompt 化 =====
    X_train_prompt = pd.DataFrame(
        X_train.apply(generate_prompt, axis=1),
        columns=["text"]
    )
    X_eval_prompt = pd.DataFrame(
        X_eval.apply(generate_prompt, axis=1),
        columns=["text"]
    )

    y_true = X_test["sentiment"]
    X_test_prompt = pd.DataFrame(
        X_test.apply(generate_test_prompt, axis=1),
        columns=["text"]
    )

    return X_train_prompt, X_test_prompt, X_eval_prompt, y_true


def build_clean_and_perturbed_test(data_path: str):
    """
    返回：
        X_test_clean_prompt      : DataFrame，列 ["text"]（CLEAN prompt）
        y_true_clean             : list[str]
        X_test_perturbed_prompt  : DataFrame，列 ["text"]（PERTURBED prompt）
        y_true_perturbed         : list[str]
    """
    # 1️⃣ 先拿到 **raw** 的 clean test 集（不做任何扰动）
    X_train_raw, X_test_raw, X_eval_raw = load_split_raw_data(
        data_path,
        perturb_data=False,   # ✅ 强制 clean
    )
    # X_test_raw: 列 ["sentiment", "text"]

    # clean 部分：labels + prompt
    y_true_clean = X_test_raw["sentiment"].tolist()
    X_test_clean_prompt = pd.DataFrame(
        X_test_raw.apply(generate_test_prompt, axis=1),
        columns=["text"]
    )

    # 2️⃣ 构造对应的扰动版本（同一批原始句子）
    headlines = X_test_raw["text"].astype(str).tolist()

    pert_pairs_path = "./data/test_perturbations_text_pairs.csv"
    if not os.path.exists(pert_pairs_path):
        print("🧪 Generating test perturbation pairs ...")
        save_perturbation_text_pairs(
            headlines,
            filename="test_perturbations_text_pairs.csv"
        )

    pert_df = pd.read_csv(pert_pairs_path)  # 期望有列 ["orig", "pert"]

    # 用 clean test 的 label 映射扰动样本
    label_map = dict(zip(X_test_raw["text"], X_test_raw["sentiment"]))
    pert_df["sentiment"] = pert_df["orig"].map(label_map)

    # 构造和 raw test 相似的 DataFrame：["text","sentiment"]
    X_test_pert_raw = pert_df.rename(columns={"pert": "text"})[["text", "sentiment"]].copy()

    # 对扰动样本也做 prompt 化
    X_test_perturbed_prompt = pd.DataFrame(
        X_test_pert_raw.apply(generate_test_prompt, axis=1),
        columns=["text"]
    )
    y_true_perturbed = X_test_pert_raw["sentiment"].tolist()

    return X_test_clean_prompt, y_true_clean, X_test_perturbed_prompt, y_true_perturbed

def build_clean_and_perturbed_pairs(data_path: str):
    """
    构造 **一一成对** 的 CLEAN / PERTURBED eval 集，用于 Flip-Rate / Sym-KL。

    返回：
        X_clean_prompt_pairs : DataFrame，列 ["text"]（clean prompt，已 generate_test_prompt）
        X_pert_prompt_pairs  : DataFrame，列 ["text"]（perturbed prompt，已 generate_test_prompt）
        y_true_pairs         : list[str]，与 clean / perturbed 成对对齐的 label
    """
    # 1️⃣ 还是先拿 test split 的 clean raw 数据
    X_train_raw, X_test_raw, X_eval_raw = load_split_raw_data(
        data_path,
        perturb_data=False,   # ✅ 强制 clean
    )
    # X_test_raw: ["sentiment", "text"]

    # 2️⃣ 读入 test 对应的 perturb pairs
    pert_pairs_path = "./data/test_perturbations_text_pairs.csv"
    if not os.path.exists(pert_pairs_path):
        headlines = X_test_raw["text"].astype(str).tolist()
        print("🧪 Generating test perturbation pairs (for pair eval) ...")
        save_perturbation_text_pairs(
            headlines,
            filename="test_perturbations_text_pairs.csv"
        )

    pert_df = pd.read_csv(pert_pairs_path)  # 期望有列 ["orig", "pert"]

    # 只保留那些确实在当前 test set 里的 orig
    test_texts = set(X_test_raw["text"].astype(str).tolist())
    pert_df = pert_df[pert_df["orig"].astype(str).isin(test_texts)].copy()

    # 如果某个 orig 有多条 perturb，这里先简单拿第一条
    pert_df_unique = pert_df.drop_duplicates(subset=["orig"])

    # 3️⃣ 用 "text"/"orig" 做 inner join，获得一一配对的子集
    pairs = X_test_raw.merge(
        pert_df_unique,
        left_on="text",
        right_on="orig",
        how="inner",
    )
    # pairs 现在应该包含列：["text", "sentiment", "orig", "pert", ...]

    # 4️⃣ 构造 clean / perturbed 的 DataFrame：["text","sentiment"]
    clean_pairs_raw = pairs[["text", "sentiment"]].copy()
    pert_pairs_raw  = pairs[["pert", "sentiment"]].rename(columns={"pert": "text"}).copy()

    # 5️⃣ 同样做 prompt 化，保持和你原 eval 完全一致的 prompt 格式
    X_clean_prompt_pairs = pd.DataFrame(
        clean_pairs_raw.apply(generate_test_prompt, axis=1),
        columns=["text"]
    )
    X_pert_prompt_pairs = pd.DataFrame(
        pert_pairs_raw.apply(generate_test_prompt, axis=1),
        columns=["text"]
    )
    y_true_pairs = clean_pairs_raw["sentiment"].tolist()

    return X_clean_prompt_pairs, X_pert_prompt_pairs, y_true_pairs
