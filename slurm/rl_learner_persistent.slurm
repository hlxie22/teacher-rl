#!/bin/bash
#SBATCH --job-name=rtt-learner
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs

CHILD_PIDS=()
rtt_install_requeue_traps_pid_array CHILD_PIDS

rtt_use_default_env
rtt_init_config
source slurm/common.sh

PLACEMENT_MANIFEST="${PLACEMENT_MANIFEST:?PLACEMENT_MANIFEST must be exported}"

eval "$(python -m src.config_helpers shell --config "$CONFIG")"
eval "$(python -m src.placement runtime-shell --manifest "$PLACEMENT_MANIFEST")"

if [[ "$RTT_PL_MODE" != "packed" && "$RTT_PL_MODE" != "legacy" && "$RTT_PL_MODE" != "trl_colocate" ]]; then
  echo "ERROR: unsupported runtime placement mode: $RTT_PL_MODE"
  exit 1
fi

component_host() {
  local cid="$1"
  local submit_dir="${SLURM_SUBMIT_DIR:-$(pwd)}"
  srun --het-group="$cid" --nodes=1 --ntasks=1 --exact \
       --export=ALL,RTT_HOST_HELPER_DIR="$submit_dir" \
       /bin/bash -lc 'cd "$RTT_HOST_HELPER_DIR" && source slurm/lib.sh && rtt_require_resolved_hostname' | tail -n1
}

split_csv_to_array() {
  local raw="$1"
  local -n out_arr_ref="$2"
  out_arr_ref=()
  if [[ -n "$raw" ]]; then
    IFS=',' read -r -a out_arr_ref <<< "$raw"
  fi
}

join_by_comma() {
  local IFS=','
  echo "$*"
}

poll_url_ready() {
  local url="$1"
  python - "$url" <<'PY'
import sys, urllib.request
url = sys.argv[1].rstrip('/') + '/v1/models'
try:
    with urllib.request.urlopen(url, timeout=5) as resp:
        ok = 200 <= int(getattr(resp, 'status', 200)) < 300
    print('1' if ok else '0')
except Exception:
    print('0')
PY
}

wait_for_urls() {
  local label="$1"
  local timeout_s="$2"
  shift 2
  local urls=("$@")
  if (( ${#urls[@]} == 0 )); then
    return 0
  fi

  local start now all_ok url ready
  start="$(date +%s)"
  while true; do
    all_ok=1
    for url in "${urls[@]}"; do
      ready="$(poll_url_ready "$url")"
      if [[ "$ready" != "1" ]]; then
        all_ok=0
        break
      fi
    done

    if (( all_ok == 1 )); then
      echo "[$(date)] all $label endpoints are ready"
      return 0
    fi

    now="$(date +%s)"
    if (( now - start >= timeout_s )); then
      echo "ERROR: timed out waiting for $label endpoints: ${urls[*]}"
      return 1
    fi
    sleep 5
  done
}

get_coord_state() {
python - <<PY
import json
st_path = "$RTT_CFG_COORD_DIR/state.json"
try:
    st=json.load(open(st_path))
except Exception:
    st={}
r = int(st.get("round", 0) or 0)
phase = str(st.get("phase","train") or "train")
adapter = st.get("teacher_adapter") or ""
last = int(st.get("last_teacher_eval_round", -1) or -1)
infr = st.get("eval_inflight_round", None)
infj = st.get("eval_inflight_job_id", None)
inft = st.get("eval_inflight_ts", None)
def fmt(x):
    if x is None:
        return ""
    return str(x)
print(r, phase, adapter, last, fmt(infr), fmt(infj), fmt(inft), sep="\t")
PY
}

start_internal_eval_watch() {
  if [[ "${RTT_CFG_TEACHER_EVAL_SCHEDULE:-online}" != "online" ]]; then
    return 0
  fi

  if [[ "${RTT_PL_EVAL_COLOCATE:-0}" != "1" ]]; then
    return 0
  fi

  local poll="${POLL:-600}"
  local eval_cid="${RTT_PL_EVAL_COMPONENT_ID:?missing eval component id}"
  local eval_gpus="${RTT_PL_EVAL_GPUS:?missing colocated eval gpu count}"
  local outfile=""
  local outdir="$RTT_CFG_EVAL_OUTDIR"
  local enabled="$RTT_CFG_TEACHER_EVAL_ENABLED"
  local every="$RTT_CFG_TEACHER_EVAL_EVERY_ROUNDS"

  mkdir -p "$outdir"

  (
    echo "[$(date)] internal eval watcher starting: component_id=$eval_cid eval_gpus=$eval_gpus poll=$poll enabled=$enabled every_rounds=$every"

    if [[ "$enabled" != "1" ]]; then
      echo "[$(date)] teacher_eval disabled in config; internal watcher exiting."
      exit 0
    fi

    while true; do
      IFS=$'\t' read -r R PHASE ADAPTER LAST INFR INFJ INFT <<<"$(get_coord_state)"

      DUE=0
      if [[ -n "$ADAPTER" ]] && [[ "$R" != "$LAST" ]] && (( R % EVERY == 0 )); then
        DUE=1
      fi

      if (( DUE == 1 )); then
        SHOULD_RUN=1
        if [[ -n "$INFR" ]] && [[ "$INFR" == "$R" ]]; then
          SHOULD_RUN=0
        fi

        if (( SHOULD_RUN == 1 )); then
          OUTFILE="$outdir/teacher_eval.round_${R}.json"
          echo "[$(date)] launching colocated eval for round=$R adapter=$ADAPTER on component=$eval_cid gpus=$eval_gpus"
          python - <<PY
import time
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    eval_inflight_round=int("$R"),
    eval_inflight_job_id="inalloc:${SLURM_JOB_ID}:round${R}",
    eval_inflight_ts=float(time.time()),
)
print(f"[internal_eval_watch] set inflight round={int('$R')} job=inalloc:{'$SLURM_JOB_ID'}:round{int('$R')}")
PY
          if ! srun --het-group="$eval_cid" --nodes=1 --ntasks="$eval_gpus" --gpus-per-task=1 --gpu-bind=closest \
               --cpus-per-task="$RTT_CFG_CPUS_PER_GPU" --exact --kill-on-bad-exit=1 \
               --export=ALL,CONFIG="$CONFIG",ROUND="$R",ADAPTER="$ADAPTER",OUTFILE="$OUTFILE",EVAL_GPUS="$eval_gpus" \
             /bin/bash slurm/teacher_eval_in_alloc.sh; then
            echo "[$(date)] colocated eval failed for round=$R; clearing inflight so the watcher can retry later"
            python - <<PY
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(eval_inflight_round=None, eval_inflight_job_id=None, eval_inflight_ts=None)
PY
          fi
        fi
      fi

      if [[ "${PHASE,,}" == "done" ]] && (( DUE == 0 )); then
        echo "[$(date)] coord phase=done and no teacher eval is due; internal watcher exiting."
        exit 0
      fi

      sleep "$poll"
    done
  ) &

  EVAL_WATCH_PID="$!"
  CHILD_PIDS+=("$EVAL_WATCH_PID")
}

MASTER_PORT="${MASTER_PORT:-29500}"
TRAINER_WORLD_SIZE="$RTT_PL_TRAINER_WORLD_SIZE"
NEXT_TRAINER_OFFSET=0
FIRST_TRAINER_HOST=""
TRAINER_COMPONENTS=()
TRAINER_COUNTS=()
TRAINER_OFFSETS=()
EVAL_WATCH_PID=""

RTT_TEACHER_LIVE_ADAPTER_ROOT="$RTT_CFG_PERSISTENT_LIVE_ADAPTER_ROOT"
mkdir -p "$RTT_TEACHER_LIVE_ADAPTER_ROOT"

STUDENT_URLS=()
TEACHER_URLS=()

for cid in $RTT_PL_COMPONENT_IDS; do
  host="$(component_host "$cid")"

  trainer_count_var="RTT_PL_COMP_${cid}_TRAINER_COUNT"
  trainer_ids_var="RTT_PL_COMP_${cid}_TRAINER_IDS"
  student_count_var="RTT_PL_COMP_${cid}_STUDENT_COUNT"
  student_ids_var="RTT_PL_COMP_${cid}_STUDENT_IDS"
  teacher_count_var="RTT_PL_COMP_${cid}_TEACHER_COUNT"
  teacher_ids_var="RTT_PL_COMP_${cid}_TEACHER_IDS"

  trainer_count="${!trainer_count_var}"
  trainer_ids_raw="${!trainer_ids_var}"
  student_count="${!student_count_var}"
  student_ids_raw="${!student_ids_var}"
  teacher_count="${!teacher_count_var}"
  teacher_ids_raw="${!teacher_ids_var}"

  echo "[$(date)] component $cid host=$host trainer_count=$trainer_count student_count=$student_count teacher_count=$teacher_count"

  if (( trainer_count > 0 )); then
    if [[ -z "$FIRST_TRAINER_HOST" ]]; then
      FIRST_TRAINER_HOST="$host"
    fi
    TRAINER_COMPONENTS+=("$cid")
    TRAINER_COUNTS+=("$trainer_count")
    TRAINER_OFFSETS+=("$NEXT_TRAINER_OFFSET")
    NEXT_TRAINER_OFFSET=$(( NEXT_TRAINER_OFFSET + trainer_count ))
  fi

  if (( student_count > 0 )); then
    split_csv_to_array "$student_ids_raw" student_ids_arr
    port_base="$RTT_CFG_STUDENT_BASE_PORT"
    for rid in "${student_ids_arr[@]}"; do
      STUDENT_URLS+=("http://${host}:$(( port_base + rid ))")
    done
    (
      srun --het-group="$cid" --nodes=1 --ntasks="$student_count" --gpus-per-task=1 --gpu-bind=closest \
           --cpus-per-task="$RTT_CFG_STUDENT_CPUS_PER_REPLICA" --exact --kill-on-bad-exit=1 \
           --export=ALL,CONFIG="$CONFIG",RTT_STUDENT_REPLICA_IDS="$student_ids_raw",RTT_STUDENT_BASE_PORT="$RTT_CFG_STUDENT_BASE_PORT",RTT_STUDENT_MODEL_ID="$RTT_CFG_STUDENT_MODEL_ID",RTT_STUDENT_DTYPE="$RTT_CFG_STUDENT_DTYPE",RTT_STUDENT_MAX_MODEL_LEN="$RTT_CFG_STUDENT_MAX_MODEL_LEN",RTT_STUDENT_GPU_MEM_UTIL="$RTT_CFG_STUDENT_GPU_MEM_UTIL",RTT_INFER_LANGUAGE_MODEL_ONLY="$RTT_CFG_INFER_LANGUAGE_MODEL_ONLY",RTT_INFER_CPU_OFFLOAD_GB="$RTT_CFG_INFER_CPU_OFFLOAD_GB",RTT_INFER_ENFORCE_EAGER="$RTT_CFG_INFER_ENFORCE_EAGER",RTT_INFER_DISABLE_LOG_STATS="$RTT_CFG_INFER_DISABLE_LOG_STATS",RTT_INFER_ATTN_BACKEND="$RTT_CFG_INFER_ATTN_BACKEND" \
        /bin/bash slurm/launch_student_server_replica.sh
    ) &
    CHILD_PIDS+=("$!")
  fi

  if (( teacher_count > 0 )); then
    split_csv_to_array "$teacher_ids_raw" teacher_ids_arr
    port_base="$RTT_CFG_TEACHER_ROLLOUT_BASE_PORT"
    for rid in "${teacher_ids_arr[@]}"; do
      TEACHER_URLS+=("http://${host}:$(( port_base + rid ))")
    done
    (
      srun --het-group="$cid" --nodes=1 --ntasks="$teacher_count" --gpus-per-task="$RTT_CFG_TEACHER_ROLLOUT_TP_SIZE" --gpu-bind=closest \
           --cpus-per-task="$RTT_CFG_TEACHER_ROLLOUT_CPUS_PER_REPLICA" --exact --kill-on-bad-exit=1 \
           --export=ALL,CONFIG="$CONFIG",RTT_TEACHER_REPLICA_IDS="$teacher_ids_raw",RTT_TEACHER_BASE_PORT="$RTT_CFG_TEACHER_ROLLOUT_BASE_PORT",RTT_TEACHER_MODEL_ID="$RTT_CFG_TEACHER_MODEL_ID",RTT_TEACHER_DTYPE="$RTT_CFG_TEACHER_ROLLOUT_DTYPE",RTT_TEACHER_TP_SIZE="$RTT_CFG_TEACHER_ROLLOUT_TP_SIZE",RTT_TEACHER_MAX_MODEL_LEN="$RTT_CFG_TEACHER_ROLLOUT_MAX_MODEL_LEN",RTT_TEACHER_GPU_MEM_UTIL="$RTT_CFG_TEACHER_ROLLOUT_GPU_MEM_UTIL",RTT_TEACHER_REQUEST_MAX_LOGPROBS="$RTT_CFG_TEACHER_ROLLOUT_MAX_LOGPROBS",RTT_TEACHER_MAX_NUM_SEQS="$RTT_CFG_TEACHER_ROLLOUT_MAX_NUM_SEQS",RTT_TEACHER_LIVE_ADAPTER_ROOT="$RTT_TEACHER_LIVE_ADAPTER_ROOT",RTT_INFER_LANGUAGE_MODEL_ONLY="$RTT_CFG_INFER_LANGUAGE_MODEL_ONLY",RTT_INFER_CPU_OFFLOAD_GB="$RTT_CFG_INFER_CPU_OFFLOAD_GB",RTT_INFER_ENFORCE_EAGER="$RTT_CFG_INFER_ENFORCE_EAGER",RTT_INFER_DISABLE_LOG_STATS="$RTT_CFG_INFER_DISABLE_LOG_STATS",RTT_INFER_ATTN_BACKEND="$RTT_CFG_INFER_ATTN_BACKEND",RTT_INFER_MAX_LORAS="$RTT_CFG_INFER_MAX_LORAS",RTT_INFER_MAX_LORA_RANK="$RTT_CFG_INFER_MAX_LORA_RANK" \
        /bin/bash slurm/launch_teacher_rollout_server.sh
    ) &
    CHILD_PIDS+=("$!")
  fi
done

if [[ -z "$FIRST_TRAINER_HOST" ]]; then
  echo "ERROR: no trainer components found in placement manifest"
  exit 1
fi

if (( NEXT_TRAINER_OFFSET != TRAINER_WORLD_SIZE )); then
  echo "ERROR: trainer world size mismatch: manifest says $TRAINER_WORLD_SIZE but packed launcher found $NEXT_TRAINER_OFFSET ranks"
  exit 1
fi

wait_for_urls student "$RTT_CFG_STUDENT_STARTUP_TIMEOUT_S" "${STUDENT_URLS[@]}"
wait_for_urls teacher "$RTT_CFG_TEACHER_ROLLOUT_STARTUP_TIMEOUT_S" "${TEACHER_URLS[@]}"

STUDENT_URLS_CSV="$(join_by_comma "${STUDENT_URLS[@]}")"
TEACHER_URLS_CSV="$(join_by_comma "${TEACHER_URLS[@]}")"

echo "[$(date)] placement runtime summary:"
echo "  mode=$RTT_PL_MODE"
echo "  packing_policy=$RTT_PL_PACKING_POLICY"
echo "  trainer_world_size=$TRAINER_WORLD_SIZE"
echo "  master=${FIRST_TRAINER_HOST}:${MASTER_PORT}"
echo "  student_urls=$STUDENT_URLS_CSV"
echo "  teacher_urls=$TEACHER_URLS_CSV"
echo "  eval_colocate=${RTT_PL_EVAL_COLOCATE:-0}"
echo "  eval_component_id=${RTT_PL_EVAL_COMPONENT_ID:-}"
echo "  eval_gpus=${RTT_PL_EVAL_GPUS:-0}"

start_internal_eval_watch

TRAINER_STEP_PIDS=()
for idx in "${!TRAINER_COMPONENTS[@]}"; do
  cid="${TRAINER_COMPONENTS[$idx]}"
  tcount="${TRAINER_COUNTS[$idx]}"
  toffset="${TRAINER_OFFSETS[$idx]}"
  (
    srun --het-group="$cid" --nodes=1 --ntasks="$tcount" --gpus-per-task=1 --gpu-bind=closest \
         --cpus-per-task="$RTT_CFG_CPUS_PER_GPU" --exact --kill-on-bad-exit=1 \
         --export=ALL,CONFIG="$CONFIG",WORLD_SIZE="$TRAINER_WORLD_SIZE",MASTER_ADDR="$FIRST_TRAINER_HOST",MASTER_PORT="$MASTER_PORT",RTT_TRAINER_RANK_OFFSET="$toffset",RTT_STUDENT_VLLM_BASE_URLS="$STUDENT_URLS_CSV",RTT_STUDENT_VLLM_MODEL_NAME="$RTT_CFG_STUDENT_MODEL_ID",RTT_TEACHER_VLLM_BASE_URLS="$TEACHER_URLS_CSV",RTT_TEACHER_VLLM_TIMEOUT_S="$RTT_CFG_TEACHER_ROLLOUT_REQUEST_TIMEOUT_S",RTT_TEACHER_VLLM_MAX_RETRIES="$RTT_CFG_TEACHER_ROLLOUT_REQUEST_MAX_RETRIES",RTT_TEACHER_VLLM_RETRY_BACKOFF_S="$RTT_CFG_TEACHER_ROLLOUT_REQUEST_RETRY_BACKOFF_S",RTT_TEACHER_VLLM_MAX_LOGPROBS="$RTT_CFG_TEACHER_ROLLOUT_MAX_LOGPROBS",RTT_TEACHER_VLLM_EXTRA_BODY="$RTT_CFG_TEACHER_ROLLOUT_EXTRA_BODY_JSON" \
      /bin/bash slurm/launch_trainer_rank.sh
  ) &
  TRAINER_STEP_PIDS+=("$!")
  CHILD_PIDS+=("$!")
done

trainer_rc=0
for pid in "${TRAINER_STEP_PIDS[@]}"; do
  if ! wait "$pid"; then
    trainer_rc=1
  fi
done

eval_watch_rc=0
if [[ -n "$EVAL_WATCH_PID" ]]; then
  if ! wait "$EVAL_WATCH_PID"; then
    eval_watch_rc=1
  fi
fi

rtt_stop_pid_array_if_alive CHILD_PIDS
wait || true

if (( trainer_rc != 0 )); then
  echo "ERROR: one or more trainer steps failed"
  exit "$trainer_rc"
fi

if (( eval_watch_rc != 0 )); then
  echo "ERROR: internal eval watcher failed"
  exit "$eval_watch_rc"
fi

echo "[$(date)] rl_learner_persistent completed successfully"
