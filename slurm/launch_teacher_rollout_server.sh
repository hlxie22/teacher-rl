#!/bin/bash
set -euo pipefail

source slurm/lib.sh
rtt_job_prelude
rtt_use_infer_env
source slurm/common.sh

export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_PLUGINS=lora_filesystem_resolver
export VLLM_LORA_RESOLVER_CACHE_DIR="${RTT_TEACHER_LIVE_ADAPTER_ROOT:?}"

TASK_INDEX="${SLURM_LOCALID:-${SLURM_PROCID:-0}}"
REPLICA_ID="$(rtt_resolve_replica_id "$TASK_INDEX" "${RTT_TEACHER_REPLICA_ID:-}" "${RTT_TEACHER_REPLICA_IDS:-}")"

PORT=$(( ${RTT_TEACHER_BASE_PORT:?} + REPLICA_ID ))
NODE_HOST="$(rtt_require_resolved_hostname)"
echo "[$(date)] teacher-rollout-vllm node=$NODE_HOST task_index=$TASK_INDEX replica=$REPLICA_ID port=$PORT live_root=$RTT_TEACHER_LIVE_ADAPTER_ROOT CVD=${CUDA_VISIBLE_DEVICES:-}"

ARGS=()
rtt_append_common_vllm_args \
  ARGS \
  "$RTT_TEACHER_MODEL_ID" \
  "$RTT_TEACHER_DTYPE" \
  "$RTT_TEACHER_MAX_MODEL_LEN" \
  "$RTT_TEACHER_GPU_MEM_UTIL" \
  "$PORT"
ARGS+=(
  --tensor-parallel-size "$RTT_TEACHER_TP_SIZE"
  --enable-lora
  --max-loras "$RTT_INFER_MAX_LORAS"
  --max-lora-rank "$RTT_INFER_MAX_LORA_RANK"
  --max-logprobs "$RTT_TEACHER_REQUEST_MAX_LOGPROBS"
)

if [[ -n "${RTT_TEACHER_MAX_NUM_SEQS:-}" && "${RTT_TEACHER_MAX_NUM_SEQS}" != "0" ]]; then
  ARGS+=(--max-num-seqs "$RTT_TEACHER_MAX_NUM_SEQS")
fi

exec vllm serve "$RTT_TEACHER_MODEL_ID" "${ARGS[@]}"