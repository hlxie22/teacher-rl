#!/bin/bash

rtt_job_prelude() {
  local submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
  cd "$submit_dir"
  if (( $# > 0 )); then
    mkdir -p "$@"
  fi
}

rtt_init_config() {
  CONFIG="${CONFIG:-configs/default.yml}"
  export CONFIG
}

rtt_use_default_env() {
  local root="${SLURM_SUBMIT_DIR:-$(pwd)}"
  export VENV_DIR="$root/.venv"
  export REQ_FILE="$root/requirements.txt"
  export REQ_PROFILE="default"
}

rtt_use_infer_env() {
  local root="${SLURM_SUBMIT_DIR:-$(pwd)}"
  export VENV_DIR="$root/.venv_infer"
  export REQ_FILE="$root/requirements.qwen35-infer.txt"
  export REQ_PROFILE="infer"
}

rtt_use_train_env() {
  local root="${SLURM_SUBMIT_DIR:-$(pwd)}"
  export VENV_DIR="$root/.venv_train"
  export REQ_FILE="$root/requirements.qwen35-train.txt"
  export REQ_PROFILE="train"
}

rtt_resolved_hostname() {
  local out=""

  if command -v hostname >/dev/null 2>&1; then
    out="$(hostname -f 2>/dev/null || hostname 2>/dev/null || true)"
  fi

  if [[ -z "$out" && -x /bin/hostname ]]; then
    out="$(/bin/hostname -f 2>/dev/null || /bin/hostname 2>/dev/null || true)"
  fi

  if [[ -z "$out" && -x /usr/bin/hostname ]]; then
    out="$(/usr/bin/hostname -f 2>/dev/null || /usr/bin/hostname 2>/dev/null || true)"
  fi

  if [[ -z "$out" ]]; then
    out="$(uname -n 2>/dev/null || true)"
  fi

  out="$(printf '%s' "$out" | head -n1 | tr -d '\r')"
  if [[ -z "$out" ]]; then
    return 1
  fi

  printf '%s\n' "$out"
}

rtt_require_resolved_hostname() {
  local out=""
  if ! out="$(rtt_resolved_hostname)"; then
    echo "ERROR: could not resolve hostname on this node" >&2
    return 1
  fi
  printf '%s\n' "$out"
}

rtt_stop_process_group_if_alive() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- -"${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  fi
}

rtt_stop_pid_array_if_alive() {
  local arr_name="$1"
  local -n arr_ref="$arr_name"
  local pid
  for pid in "${arr_ref[@]:-}"; do
    rtt_stop_process_group_if_alive "$pid"
  done
}

RTT_REQUEUE_SENT=0
RTT_REQUEUE_MAIN_PID_VAR="MAIN_PID"
RTT_REQUEUE_CHILD_ARRAY_NAME="CHILD_PIDS"

rtt_usr1_requeue_simple() {
  if (( RTT_REQUEUE_SENT == 1 )); then
    return 0
  fi
  RTT_REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue..."
  scontrol requeue "$SLURM_JOB_ID" || true
  exit 0
}

rtt_usr1_requeue_main_pid() {
  if (( RTT_REQUEUE_SENT == 1 )); then
    return 0
  fi
  RTT_REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue..."
  scontrol requeue "$SLURM_JOB_ID" || true
  rtt_stop_process_group_if_alive "${!RTT_REQUEUE_MAIN_PID_VAR:-}"
  exit 0
}

rtt_usr1_requeue_pid_array() {
  if (( RTT_REQUEUE_SENT == 1 )); then
    return 0
  fi
  RTT_REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue and stopping child steps..."
  rtt_stop_pid_array_if_alive "$RTT_REQUEUE_CHILD_ARRAY_NAME"
  scontrol requeue "$SLURM_JOB_ID" || true
  exit 0
}

rtt_term_stop_pid_array() {
  echo "[$(date)] termination received: stopping child steps..."
  rtt_stop_pid_array_if_alive "$RTT_REQUEUE_CHILD_ARRAY_NAME"
  exit 1
}

rtt_install_requeue_trap_simple() {
  RTT_REQUEUE_SENT=0
  trap rtt_usr1_requeue_simple USR1
}

rtt_install_requeue_trap_main_pid() {
  RTT_REQUEUE_SENT=0
  RTT_REQUEUE_MAIN_PID_VAR="${1:-MAIN_PID}"
  trap rtt_usr1_requeue_main_pid USR1
}

rtt_install_requeue_traps_pid_array() {
  RTT_REQUEUE_SENT=0
  RTT_REQUEUE_CHILD_ARRAY_NAME="${1:-CHILD_PIDS}"
  trap rtt_usr1_requeue_pid_array USR1
  trap rtt_term_stop_pid_array TERM INT
}

rtt_resolve_replica_id() {
  local task_index="${1:-0}"
  local explicit_id="${2:-}"
  local ids_csv="${3:-}"
  local replica_id="$explicit_id"

  if [[ -n "$ids_csv" ]]; then
    local -a ids_arr=()
    IFS=',' read -r -a ids_arr <<< "$ids_csv"
    replica_id="${ids_arr[$task_index]:-$replica_id}"
  fi

  if [[ -z "$replica_id" ]]; then
    replica_id="$task_index"
  fi
  printf '%s\n' "$replica_id"
}

rtt_pin_single_gpu_task_cvd() {
  local task_index="${1:-0}"
  local cvd_raw="${CUDA_VISIBLE_DEVICES:-}"

  if [[ "$cvd_raw" == *,* ]]; then
    local -a cvd_arr=()
    IFS=',' read -r -a cvd_arr <<< "$cvd_raw"
    export CUDA_VISIBLE_DEVICES="${cvd_arr[$task_index]:-${cvd_arr[0]}}"
  elif [[ -z "$cvd_raw" ]]; then
    export CUDA_VISIBLE_DEVICES="$task_index"
  fi
}

rtt_append_common_vllm_args() {
  local out_name="$1"
  local served_model_name="$2"
  local dtype="$3"
  local max_model_len="$4"
  local gpu_mem_util="$5"
  local port="$6"

  local -n out_ref="$out_name"
  out_ref+=(
    --host 0.0.0.0
    --port "$port"
    --served-model-name "$served_model_name"
    --dtype "$dtype"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$gpu_mem_util"
    --generation-config vllm
    --trust-remote-code
  )

  if [[ "${RTT_INFER_LANGUAGE_MODEL_ONLY:-1}" == "1" ]]; then
    out_ref+=(--language-model-only)
  fi
  if [[ "${RTT_INFER_CPU_OFFLOAD_GB:-0}" != "0" ]]; then
    out_ref+=(--cpu-offload-gb "$RTT_INFER_CPU_OFFLOAD_GB")
  fi
  if [[ "${RTT_INFER_ENFORCE_EAGER:-0}" == "1" ]]; then
    out_ref+=(--enforce-eager)
  fi
  if [[ "${RTT_INFER_DISABLE_LOG_STATS:-1}" == "1" ]]; then
    out_ref+=(--disable-log-stats)
  fi
  if [[ -n "${RTT_INFER_ATTN_BACKEND:-}" ]]; then
    out_ref+=(--attention-backend "$RTT_INFER_ATTN_BACKEND")
  fi
}