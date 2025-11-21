#!/bin/bash
# ==============================================
# 一键运行 sentiment pipeline (baseline / SFT / GRPO)
# ==============================================

# 关掉所有 HF 离线模式
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# 默认环境
# export CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false


# 日志目录
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# 可选模式: baseline / sft / grpo
MODE=${1:-baseline}

# 第二个参数：是否使用扰动数据
# 用法示例：bash run.sh sft perturb
if [ "${2}" = "perturb" ]; then
    PERTURB_FLAG="--perturb_data"
    echo "📊 Data mode: USING perturbed data (train augmented)"
else
    PERTURB_FLAG=""
    echo "📊 Data mode: using ORIGINAL data only"
fi

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
    python main.py ${PERTURB_FLAG} 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "sft" ]; then
    echo "→ Running SFT (LoRA fine-tuning)..."
    python main.py --run_sft ${PERTURB_FLAG} 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "grpo" ]; then
    echo "→ Running GRPO (reinforcement fine-tuning)..."
    python main.py --run_grpo ${PERTURB_FLAG} 2>&1 | tee ${LOG_FILE}

else
    echo "❌ Unknown mode: ${MODE}"
    echo "Usage: bash run.sh [baseline|sft|grpo] [perturb]"
    exit 1
fi

echo "===================================================="
echo " ✅ Finished Mode: ${MODE}"
echo " 🕓 End Time:     $(date)"
echo " 💾 Log saved to: ${LOG_FILE}"
echo "===================================================="
