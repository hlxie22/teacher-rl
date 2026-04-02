#!/bin/bash
#SBATCH --job-name=rtt-inspect
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data outputs

source slurm/common.sh

# ------------------------------------------------------------
# Minimal fix #1: don't let USR1 kill the batch script early
# ------------------------------------------------------------
trap 'echo "[$(date)] Caught SIGUSR1 (pre-time-limit warning). Ignoring for inspect run." >&2' USR1

# =========================
# USER KNOBS (edit these)
# =========================
export RTT_INSPECT_CONFIG="configs/default.yml"

# student | teacher | both
export RTT_INSPECT_WHO="both"

# how many problems to inspect (TOTAL across all GPUs; will be sharded)
export RTT_INSPECT_N="8"
export RTT_INSPECT_BATCH="1"
export RTT_INSPECT_SEED="0"

# optional: pin specific examples (comma-separated ids)
export RTT_INSPECT_IDS=""

# -------- Dataset selection --------
export RTT_INSPECT_DATASET="math_level5"

# Optional HF loader knobs for DATASET=amc/math_level5/hf:...
export RTT_INSPECT_HF_SPLIT="train"
# export RTT_INSPECT_HF_NAME="default"
# export RTT_INSPECT_HF_ID="AI-MO/aimo-validation-math-level-5"

# -------- Generation overrides --------
export RTT_INSPECT_K="1"

# blank means "use config"
export RTT_INSPECT_MAX_NEW="16384"
export RTT_INSPECT_TEMP=""
export RTT_INSPECT_TOP_P=""

# -------- vLLM / inference overrides --------
export RTT_INSPECT_MAX_MODEL_LEN="16384"
export RTT_INSPECT_GPU_MEMORY_UTILIZATION="0.98"
export RTT_INSPECT_MAX_NUM_SEQS="1"

# Optional: merge per-rank JSONL shards into RTT_INSPECT_JSONL_OUT on rank0
export RTT_INSPECT_MERGE_JSONL="1"

# stop strings: marker | sequence | both
export RTT_INSPECT_STOP_MODE="marker"

# thinking: auto | on | off
# Requested behavior: force OFF
export RTT_INSPECT_THINKING_MODE="off"

# printing / filtering
export RTT_INSPECT_ONLY_TRUNCATED="0"
export RTT_INSPECT_FULL_OUTPUT="1"
export RTT_INSPECT_SHOW_QUESTION_CHARS="800"
export RTT_INSPECT_SHOW_OUTPUT_CHARS="2000"

# optional: write JSONL with full details (rank0 will merge rank shards into this file by default)
export RTT_INSPECT_JSONL_OUT="outputs/inspect_rollouts.jsonl"

# A stable run id shared by all ranks in this job (used for rendezvous temp dir)
export RTT_INSPECT_RUN_ID="${RTT_INSPECT_RUN_ID:-job${SLURM_JOB_ID}.$(date +%s)}"

echo "[$(date)] Running RTT rollout inspector with env:"
env | grep '^RTT_INSPECT_' | sort || true
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-} SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

srun --ntasks="${SLURM_NTASKS:-4}" --gpus-per-task=1 --gpu-bind=closest \
  --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" \
  --export=ALL \
  --kill-on-bad-exit=1 \
  -o "logs/%x-%j.rank%t.out" -e "logs/%x-%j.rank%t.err" \
  python -m src.inspect_rollouts