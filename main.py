import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM
import torch

import argparse
from datasets import Dataset
from datetime import datetime

from configs import peft_config, training_arguments, CACHE_DIR, LLAMA_MODEL_NAME
from data_utils.dataset_build import load_and_split_data
from data_utils.evaluation import evaluate
from models.load_llama import load_llama
from models.predict_llama import predict
from training.sft_trainer import run_sft
from training.grpo_trainer import GRPOTrainer


def main():
    # ======== 1️⃣ 参数解析 ========
    parser = argparse.ArgumentParser(description="Run sentiment classification pipeline")
    parser.add_argument("--run_sft", action="store_true", help="Run supervised fine-tuning (LoRA)")
    parser.add_argument("--run_grpo", action="store_true", help="Run GRPO fine-tuning (policy optimization)")
    args = parser.parse_args()

    # ======== 2️⃣ 加载数据 ========
    X_train, X_test, X_eval, y_true = load_and_split_data("data/all-data.csv")
    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)
    # breakpoint()
    # ======== 3️⃣ 模式选择 ========

    # === (1) SFT 微调 + 预测 + 评估 ===
    if args.run_sft:
        print("\n================ SFT (LoRA) MODE ================\n")

        model, tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)

        # 动态保存路径
        time_tag = datetime.now().strftime("%Y%m%d")
        training_arguments.output_dir = f"./outputs/sft_{LLAMA_MODEL_NAME.split('/')[-1]}_{time_tag}"
        # print(f"📁 Model will be saved to: {train_args.output_dir}\n")

        # === 1️⃣ 训练 ===
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
        tokenizer.save_pretrained(training_arguments.output_dir)
        print("✅ SFT training finished!\n")

        # === 2️⃣ 加载训练好的 LoRA 模型 ===
        print("→ Loading fine-tuned LoRA model for evaluation...")

        compute_dtype = getattr(torch, "float16")
        finetuned_model = training_arguments.output_dir
        print("微调模型位置：" + finetuned_model)

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            cache_dir=CACHE_DIR,
            local_files_only=True,   # 只用本地缓存
            use_fast=True,
            trust_remote_code=True,
        )

        model = AutoPeftModelForCausalLM.from_pretrained(
            finetuned_model,
            torch_dtype=compute_dtype,
            return_dict=True,
            low_cpu_mem_usage=True,
            device_map="auto" # 重要，不加会用cpu
        )

        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("./outputs/merged_model",safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained("./outputs/merged_model")


        # === 3️⃣ 预测与评估 ===
        print("→ Generating predictions on test set...")
        preds = predict(X_test, merged_model, tokenizer)  # ← 用 merged_model
        print("→ Evaluating...")
        evaluate(y_true, preds)

    # === (2) GRPO 优化 === 
    elif args.run_grpo:
        print("\n================ GRPO MODE ================\n")
        grpo_trainer = GRPOTrainer(peft_config, training_arguments, CACHE_DIR, LLAMA_MODEL_NAME)
        grpo_trainer.train(X_train)
        print("\n✅ GRPO fine-tuning done.\n")

    # === (3) Baseline 预测 === 完成
    else:
        print("\n================ BASELINE MODE ================\n")
        model, tokenizer = load_llama(LLAMA_MODEL_NAME, CACHE_DIR)
        preds = predict(X_test, model, tokenizer)
        evaluate(y_true, preds)
        print("\n✅ Baseline evaluation complete.\n")


if __name__ == "__main__":
    main()
