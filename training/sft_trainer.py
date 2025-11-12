from trl import SFTTrainer
from transformers import TrainingArguments, AutoModelForCausalLM
from peft import PeftModel
import os, torch

def run_sft(model, tokenizer, train_data, eval_data, training_args, peft_config):

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        max_seq_length=1024,
        packing=False,
        dataset_kwargs={
            "add_special_tokens": False,
            "append_concat_token": False,
        }
    )

    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)


