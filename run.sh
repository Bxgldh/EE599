#!/bin/bash
# ==============================================
# 一键运行 sentiment pipeline (baseline / SFT / GRPO)
# ==============================================

# 关掉所有 HF 离线模式
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# ⭐ 让 transformers 不要去 import torchvision
export TRANSFORMERS_NO_TORCHVISION=1

# 默认环境
export TOKENIZERS_PARALLELISM=false

# 日志目录
LOG_DIR="logs"
mkdir -p ${LOG_DIR}

# 第一个参数: baseline / sft / grpo
MODE=${1:-baseline}

# ✅ 不再在 run.sh 里控制扰动数据，是否使用 perturb_data 由 main.py 内部决定
echo "📊 Data mode: controlled INSIDE main.py (perturb logic not in run.sh)"

# 第二个参数：是否 resume
if [ "${2}" = "resume" ]; then
    RESUME_FLAG="--resume"
    echo "🔁 Resume mode: will try to resume from checkpoint"
else
    RESUME_FLAG=""
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
    python main.py ${RESUME_FLAG} 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "sft" ]; then
    echo "→ Running SFT (LoRA fine-tuning)..."
    python main.py --run_sft ${RESUME_FLAG} 2>&1 | tee ${LOG_FILE}

elif [ "${MODE}" = "grpo" ]; then
    echo "→ Running GRPO (reinforcement fine-tuning)..."
    python main.py --run_grpo ${RESUME_FLAG} 2>&1 | tee ${LOG_FILE}

else
    echo "❌ Unknown mode: ${MODE}"
    echo "Usage: bash run.sh [baseline|sft|grpo] [resume]"
    exit
fi

echo "===================================================="
echo " ✅ Finished Mode: ${MODE}"
echo " 🕓 End Time:     $(date)"
echo " 💾 Log saved to: ${LOG_FILE}"
echo "===================================================="
