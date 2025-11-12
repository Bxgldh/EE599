#!/bin/bash
# ==============================================
# 一键运行 sentiment pipeline (baseline / SFT / GRPO)
# ==============================================

# 默认环境
export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false

# 日志目录
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# 可选模式: baseline / sft / grpo
MODE=${1:-baseline}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${MODE}_${TIMESTAMP}.log"

echo "===================================================="
echo " 🧠 Running Mode: ${MODE}"
echo " 🕒 Start Time:  $(date)"
echo " 💾 Log File:   ${LOG_FILE}"
echo "===================================================="

# ==============================
# 主逻辑
# ==============================
if [ "${MODE}" = "baseline" ]; then
    echo "→ Running baseline inference..."
    python main.py 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "sft" ]; then
    echo "→ Running SFT (LoRA fine-tuning)..."
    python main.py --run_sft 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "grpo" ]; then
    echo "→ Running GRPO (reinforcement fine-tuning)..."
    python main.py --run_grpo 2>&1 | tee ${LOG_FILE}

else
    echo "❌ Unknown mode: ${MODE}"
    echo "Usage: bash run.sh [baseline|sft|grpo]"
    exit 1
fi

echo "===================================================="
echo " ✅ Finished Mode: ${MODE}"
echo " 🕓 End Time:     $(date)"
echo " 💾 Log saved to: ${LOG_FILE}"
echo "===================================================="
