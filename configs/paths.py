from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "all-data.csv"
CACHE_DIR = Path("/data3/zhenglon/huggingface/transformers")
# CACHE_DIR = Path.home() / ".cache" / "huggingface" / "transformers"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-hf"

# 1) HuggingFace cache 根目录:
#    若用户设置了 HF_HOME 就用 HF_HOME，否则默认 ~/.cache/huggingface
HF_CACHE_DIR = Path(
    os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
)

# 2) FinBERT 在 HF cache 里的目录
FINBERT_DIR = HF_CACHE_DIR / "hub" / "models--ProsusAI--finbert"

for d in [OUTPUT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)
__all__ = ["PROJECT_ROOT","DATA_PATH","CACHE_DIR","OUTPUT_DIR","LOG_DIR","LLAMA_MODEL_NAME","HF_CACHE_DIR","FINBERT_DIR"]
