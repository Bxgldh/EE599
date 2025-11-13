from peft import LoraConfig
from transformers import TrainingArguments
from .paths import OUTPUT_DIR, LOG_DIR

training_arguments = TrainingArguments(
    output_dir=str(OUTPUT_DIR),                    # directory to save and repository id
    num_train_epochs=3,                       # number of training epochs
    per_device_train_batch_size=1,            # batch size per device during training
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,                         # log every 10 steps
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=True,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="tensorboard",                  # report metrics to tensorboard
    evaluation_strategy="epoch"               # save checkpoint every epoch
)

peft_config = LoraConfig(
        lora_alpha=16,  
        lora_dropout=0.1,
        r=64, # rank 表示LoRA规模
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
)

__all__ = ["peft_config", "training_arguments"]