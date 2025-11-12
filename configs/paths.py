from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "all-data.csv"
CACHE_DIR = Path("/data3/zhenglon/huggingface/transformers")
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"
LLAMA_MODEL_NAME = "meta-llama/Llama-2-7b-hf"
for d in [OUTPUT_DIR, LOG_DIR]: d.mkdir(parents=True, exist_ok=True)
__all__ = ["PROJECT_ROOT","DATA_PATH","CACHE_DIR","OUTPUT_DIR","LOG_DIR","LLAMA_MODEL_NAME"]
