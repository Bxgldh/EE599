#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grpo_%j.out
#SBATCH --error=logs/grpo_%j.err

# ⭐ 邮件通知：任务开始 / 结束 / 失败 都发邮件
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=honglihu@usc.edu

# ⭐ 允许被 requeue（重新排队）
#SBATCH --requeue

# ⭐ 在 time limit 前 5 分钟给任务发一个 SIGUSR1 信号
#SBATCH --signal=B:USR1@300

# ========== 1. 环境 ==========
source ~/.bashrc
conda activate ftl   # ← 你的环境名

cd /project2/rashidin_1753/EE599_hongli

echo "=========================================="
echo "SLURM_JOB_ID = ${SLURM_JOB_ID}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
echo "TMPDIR = ${TMPDIR}"
echo "=========================================="
echo "========== GRPO job started at $(date) on $(hostname) =========="

# ========== 2. 捕获快到时限的信号，自动 requeue ==========
_requeue_handler() {
    echo "⚠️  Time limit is near, requeuing job ${SLURM_JOB_ID} at $(date)..."
    scontrol requeue ${SLURM_JOB_ID}
    exit 0
}
trap _requeue_handler SIGUSR1

# ========== 3. 模式：fresh / auto ==========
#   sbatch grpo_job.sh fresh  -> 强制从头跑（忽略 checkpoint）
#   sbatch grpo_job.sh        -> auto 模式（智能判断：训 or eval）
MODE=${1:-auto}
echo "🎛  GRPO launch mode: ${MODE}"

# 根目录 & 前缀，要和 Python 里的一致
BASE_GRPO_ROOT="outputs/grpo"
MODEL_PREFIX="grpo_Llama-2-7b-hf_"

# 找到最新的实验目录（按修改时间排序）
LATEST_EXP_DIR=$(ls -dt ${BASE_GRPO_ROOT}/${MODEL_PREFIX}* 2>/dev/null | head -n 1 || true)

if [ "${MODE}" = "fresh" ]; then
    echo "🧼 Fresh run requested: ignoring any existing checkpoints."
    echo "🆕 Running: bash run.sh grpo"
    bash run.sh grpo
    echo "========== GRPO job finished at $(date) =========="
    exit 0
fi

# ========== 4. auto 模式：智能决定 训 / 续训 / 只 eval ==========
if [ -n "${LATEST_EXP_DIR}" ]; then
    echo "📂 Latest experiment dir: ${LATEST_EXP_DIR}"

    # 先找最新的 checkpoint-* 目录
    LATEST_CKPT_DIR=$(ls -dt ${LATEST_EXP_DIR}/checkpoint-* 2>/dev/null | head -n 1 || true)
    echo "🔎 Latest checkpoint dir (if any): ${LATEST_CKPT_DIR}"

    TRAINER_STATE=""

    # 优先使用最新 checkpoint 里的 trainer_state.json
    if [ -n "${LATEST_CKPT_DIR}" ] && [ -f "${LATEST_CKPT_DIR}/trainer_state.json" ]; then
        TRAINER_STATE="${LATEST_CKPT_DIR}/trainer_state.json"
    # 否则再看实验根目录下有没有
    elif [ -f "${LATEST_EXP_DIR}/trainer_state.json" ]; then
        TRAINER_STATE="${LATEST_EXP_DIR}/trainer_state.json"
    fi

    if [ -n "${TRAINER_STATE}" ]; then
        echo "🔍 Found trainer_state.json at: ${TRAINER_STATE}"
        echo "   (will check global_step vs max_steps)"

        TRAIN_STATUS=$(python - <<PY
import json, sys
path = sys.argv[1]
try:
    with open(path, "r") as f:
        s = json.load(f)
    gs = s.get("global_step", 0)
    ms = s.get("max_steps", None)
    print(f"DEBUG: global_step={gs}, max_steps={ms}")
    if ms is not None and gs >= ms:
        print("finished")
    else:
        print("not_finished")
except Exception as e:
    print("not_finished")
PY
"${TRAINER_STATE}"
)

        echo "🔎 Training status from trainer_state.json:"
        echo "${TRAIN_STATUS}"

        # TRAIN_STATUS 可能有多行（DEBUG + 状态），所以用 grep 检查是否包含 finished
        if echo "${TRAIN_STATUS}" | grep -q "finished"; then
            echo "✅ Latest GRPO run already finished (global_step >= max_steps)."
            echo "📊 Running evaluation only: python main.py --eval_grpo"
            python main.py --eval_grpo # Evaluate the finished model
            echo "========== GRPO job finished at $(date) =========="
            exit 0
        fi
    else
        echo "⚠️ No trainer_state.json found in ${LATEST_EXP_DIR} or its checkpoints, will treat as not finished."
    fi

    # 如果走到这里，说明：要么没训完，要么 trainer_state.json 不可用
    if ls "${LATEST_EXP_DIR}"/checkpoint-* 1> /dev/null 2>&1; then
        echo "🔁 Found checkpoint(s) in ${LATEST_EXP_DIR}:"
        ls -d "${LATEST_EXP_DIR}"/checkpoint-* || true
        echo "🔁 Resuming training (Python will pick the latest checkpoint)."
        echo "▶ bash run.sh grpo resume"
        bash run.sh grpo resume
    else
        echo "❓ No checkpoint-* found in ${LATEST_EXP_DIR}, starting new GRPO run ..."
        echo "▶ bash run.sh grpo"
        bash run.sh grpo
    fi
else
    echo "❓ No existing GRPO experiment dir found under ${BASE_GRPO_ROOT}."
    echo "🆕 Starting first GRPO run: bash run.sh grpo"
    bash run.sh grpo
fi

echo "========== GRPO job finished at $(date) =========="
