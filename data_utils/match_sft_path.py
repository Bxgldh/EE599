from pathlib import Path


def find_latest_sft_dir(model_name: str, mode_tag: str, base_dir: str = "./outputs/sft") -> str:
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"{base_dir} does not exist")

    short_name = model_name.split("/")[-1]
    prefix = f"sft_{short_name}_"
    suffix = f"_{mode_tag}"

    def is_complete_sft_dir(p: Path) -> bool:
        """
        判定这个目录是不是一个“训练完成的 SFT checkpoint”：
        至少包含 LoRA adapter 的配置和权重文件。
        你可以根据自己实际的文件名再微调。
        """
        # 典型 PEFT LoRA 输出
        has_adapter_config = (p / "adapter_config.json").exists()
        has_adapter_model = any(
            (p / fname).exists()
            for fname in ["adapter_model.safetensors", "adapter_model.bin", "pytorch_model.bin"]
        )
        return has_adapter_config and has_adapter_model

    candidates = [
        p for p in base.glob("sft_*")
        if p.is_dir()
        and p.name.startswith(prefix)
        and p.name.endswith(suffix)
        and is_complete_sft_dir(p)          # ✅ 只保留“完整”的
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No COMPLETE SFT dir matching {prefix}*{suffix} in {base_dir}"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)
