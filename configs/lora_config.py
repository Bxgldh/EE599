from peft import LoraConfig
from trl import SFTConfig
from .paths import OUTPUT_DIR, LOG_DIR

# ====== SFT / Trainer 配置 ======
training_arguments = SFTConfig(
    output_dir=str(OUTPUT_DIR),           # 模型/Checkpoint 保存目录
    logging_dir=str(LOG_DIR),             # TensorBoard 日志目录

    num_train_epochs=3,                   # 训练 epoch 数
    per_device_train_batch_size=1,        # 每个 GPU 的 batch size
    gradient_accumulation_steps=8,        # 累积多少步再反向传播一次
    gradient_checkpointing=True,          # 开启梯度检查点以省显存

    optim="paged_adamw_32bit",            # QLoRA 里常用的优化器
    learning_rate=2e-4,                   # QLoRA 论文中的学习率
    weight_decay=0.001,

    fp16=False,
    bf16=False,

    max_grad_norm=0.3,                    # QLoRA 论文中的 max_grad_norm
    max_steps=-1,                         # 不手动限制 step 数

    warmup_ratio=0.03,                    # warmup 比例
    lr_scheduler_type="cosine",           # 余弦学习率调度

    group_by_length=True,                 # 按长度分组提高效率
    report_to=["tensorboard"],            # 上报到 tensorboard

    logging_steps=25,                     # 每 25 step 打一次 log

    # ⚠️ 这里用的是 eval_strategy（SFTConfig 的名字），不是 evaluation_strategy
    eval_strategy="epoch",                # 每个 epoch 做一次 evaluation
    save_strategy="epoch",                # 每个 epoch 存一次 checkpoint（如需）
    save_steps=0,                         # 只在 save_strategy='steps' 时起作用，这里无所谓
)

# ====== LoRA 配置 ======
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,                     # LoRA rank
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

__all__ = ["peft_config", "training_arguments"]
