#!/bin/bash
set -euo pipefail

source slurm/lib.sh
rtt_job_prelude
rtt_use_infer_env
source slurm/common.sh

TASK_INDEX="${SLURM_LOCALID:-0}"
REPLICA_ID="$(rtt_resolve_replica_id "$TASK_INDEX" "${RTT_STUDENT_REPLICA_ID:-}" "${RTT_STUDENT_REPLICA_IDS:-}")"
rtt_pin_single_gpu_task_cvd "$TASK_INDEX"

PORT=$(( ${RTT_STUDENT_BASE_PORT:?} + REPLICA_ID ))
NODE_HOST="$(rtt_require_resolved_hostname)"
echo "[$(date)] student-vllm node=$NODE_HOST task_index=$TASK_INDEX replica=$REPLICA_ID port=$PORT CVD=${CUDA_VISIBLE_DEVICES:-}"

ARGS=()
rtt_append_common_vllm_args \
  ARGS \
  "$RTT_STUDENT_MODEL_ID" \
  "$RTT_STUDENT_DTYPE" \
  "$RTT_STUDENT_MAX_MODEL_LEN" \
  "$RTT_STUDENT_GPU_MEM_UTIL" \
  "$PORT"

exec vllm serve "$RTT_STUDENT_MODEL_ID" "${ARGS[@]}"