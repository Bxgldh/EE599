# training/sft_trainer.py
from trl import SFTTrainer

def run_sft(
    model,
    tokenizer,
    train_data,
    eval_data,
    training_args,
    peft_config,
):
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # ====== 关键补丁：修掉 float vs half 的报错 ======
    def _cast_lm_head_input(module, inputs):
        """
        module: lm_head (nn.Linear)
        inputs[0]: hidden_states[..., d_model]
        把 hidden_states cast 成和 lm_head.weight 一样的 dtype
        """
        hidden = inputs[0].to(module.weight.dtype)
        # 其它位置基本不会有参数，一般只有一个 tensor
        return (hidden, *inputs[1:])

    # trainer.model 是 PeftModel，里面包着 LlamaForCausalLM
    base = trainer.model
    # 大部分版本是 base_model.lm_head
    if hasattr(base, "base_model") and hasattr(base.base_model, "lm_head"):
        base.base_model.lm_head.register_forward_pre_hook(_cast_lm_head_input)
    # 兜底：有的版本直接是 model.lm_head
    elif hasattr(base, "lm_head"):
        base.lm_head.register_forward_pre_hook(_cast_lm_head_input)
    # ====== 补丁结束 =================================

    trainer.train()
    trainer.save_model(training_args.output_dir)
    return trainer

# 35