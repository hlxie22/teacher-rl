#!/bin/bash
set -euo pipefail

source slurm/lib.sh
rtt_job_prelude
rtt_use_train_env
source slurm/common.sh

if [[ -z "${CONFIG:-}" ]]; then
  echo "ERROR: CONFIG must be exported"
  exit 1
fi
if [[ -z "${WORLD_SIZE:-}" || -z "${MASTER_ADDR:-}" || -z "${MASTER_PORT:-}" || -z "${RTT_TRAINER_RANK_OFFSET:-}" ]]; then
  echo "ERROR: trainer launcher requires WORLD_SIZE, MASTER_ADDR, MASTER_PORT, RTT_TRAINER_RANK_OFFSET"
  exit 1
fi

LOCAL_PROCID="${SLURM_PROCID:-0}"
NODE_LOCAL_TASK_ID="${SLURM_LOCALID:-0}"

# Explicitly pin this task to one GPU. Do not rely on Slurm to rewrite CVD.
rtt_pin_single_gpu_task_cvd "$NODE_LOCAL_TASK_ID"

export RANK="$(( RTT_TRAINER_RANK_OFFSET + LOCAL_PROCID ))"
export WORLD_SIZE
export MASTER_ADDR
export MASTER_PORT

# After pinning CVD to a single device, LOCAL_RANK should be 0 inside the task.
export RTT_NODE_LOCAL_TASK_ID="$NODE_LOCAL_TASK_ID"
export LOCAL_RANK=0

NODE_HOST="$(rtt_require_resolved_hostname)"
echo "[$(date)] trainer-rank node=$NODE_HOST step_rank=${SLURM_PROCID:-0} global_rank=$RANK node_local_task=$RTT_NODE_LOCAL_TASK_ID local_rank=$LOCAL_RANK world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} CVD=${CUDA_VISIBLE_DEVICES:-}"

exec python -m src.learn_persistent --config "$CONFIG"