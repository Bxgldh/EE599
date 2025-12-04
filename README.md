# Robust Financial Sentiment Analysis with LLaMA-2, QLoRA, and GRPO

This project fine-tunes LLaMA-2 for financial sentiment classification
and further improves robustness through GRPO (Group Relative Policy Optimization)
using clean–perturbed data pairs.
We additionally integrate a FinBERT-based teacher model to provide
domain-aligned reward signals.

---

## 🔧 Getting started

### Setup dependencies

Please see `environment.yaml` for dependencies and adjust PyTorch CUDA version if needed. We can set them up with conda:

```cmd
conda env create -f environment.yaml
conda activate ftl
```

### Execution command

```bash
# Llama2
bash run.sh
# SFT
bash run.sh sft
# GRPO
bash run.sh grpo
```

## 📌 Project Overview

Financial news headlines often contain subtle linguistic cues that can shift market sentiment.
However, small paraphrases or lexical perturbations can easily flip predictions
of standard classifiers.

This project addresses two challenges:

1. **Domain Adaptation**  
   Align a general-purpose LLM (LLaMA-2) with financial sentiment labels.

2. **Robustness to Text Perturbations**  
   Ensure predictions remain stable under realistic input changes
   (synonym replacement, numeric jitter, entity paraphrase, etc.).

We combine:

- **SFT with QLoRA** (parameter-efficient supervised training)  
- **GRPO** for robustness-oriented RL fine-tuning  
- **FinBERT** as a teacher signal  
- **A perturbation pipeline** to generate robust training pairs

---

## 📊 Dataset

We use the publicly available financial headline dataset  
distributed on Kaggle as `all-data.csv`.

Sentiment labels:

- **negative (0)**
- **neutral (1)**
- **positive (2)**

### Balanced clean subset

We construct a balanced subset with:

| Label        | Count |
| ------------ | ----- |
| Negative (0) | 300   |
| Neutral (1)  | 300   |
| Positive (2) | 300   |

### Perturbation expansion

Using our perturbation pipeline, each clean sample generates one perturbed counterpart.
Final counts:

| Label    | Count |
| -------- | ----- |
| Negative | 503   |
| Neutral  | 561   |
| Positive | 451   |

---

## 🔧 Perturbation Pipeline

The pipeline generates a perturbed sample $\tilde{x}$ for each clean headline $x$.

Perturbation types include:

- **Synonym substitution**
- **Entity masking / replacement**
- **Numeric drift** (e.g., “10%” → “12%”)
- **Adverb / modifier rewriting**
- **Word-order shuffling (mild)**

This produces realistic text variations while preserving semantic sentiment.

---

## 🧠 Training Pipeline

### 1️⃣ Supervised Fine-Tuning (SFT) with QLoRA

- Base model: **LLaMA-2-7B**  
- Quantization: **4-bit NF4**  
- Trainable: **LoRA adapters only**  
- Objective: Teach model to output correct labels {0,1,2}  
- Verbalizer: Maps “negative / neutral / positive” to label IDs

This stage aligns the model to the financial domain.

---

### 2️⃣ Reinforcement Learning with GRPO

GRPO replaces PPO-style absolute reward optimization with group-relative ranking,
providing stable learning without a value network.

For each prompt:

1. Sample multiple model outputs  
2. Compute group-normalized advantages  
3. Adjust the policy toward relatively better predictions

### Reward components

- **FinBERT reward:**  
  Agreement between LLaMA output and a domain-specialized teacher model.

- **Consistency reward:**  
  Clean and perturbed texts should yield similar predictions.

- **Label accuracy reward:**  
  Prefer outputs matching the ground-truth sentiment.

- **Distributional KL penalty:**  
  Stays close to reference SFT policy.

---

## 📈 Evaluation Metrics

We evaluate clean accuracy and robustness:

### Per-class metrics (finance-aware)

- **Negative (0): Recall**  
  Missing bad-news headlines can cause real financial loss.

- **Neutral (1): F1-score**  
  Neutral headlines act as noise filters → balance matters.

- **Positive (2): Precision**  
  Avoid false positives that trigger unnecessary trades.

### Robustness metrics

- **Flip Rate:**  
  Fraction of pairs $(x,\tilde{x})$ where prediction changes.

- **Symmetric KL Divergence:**  
  Measures how much the probability distribution shifts.

Lower = more robust.