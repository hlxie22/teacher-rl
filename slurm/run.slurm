#!/bin/bash
#SBATCH --job-name=rtt-grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@900
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# -----------------------------
# User-configurable knobs
# -----------------------------
CONFIG="${CONFIG:-configs/default.yml}"
export CONFIG

CONDA_ENV="${CONDA_ENV:-rtt-grpo}"
MASTER_PORT="${MASTER_PORT:-29500}"

ROUNDS="${ROUNDS:-}"
CLEAN_COORD="${CLEAN_COORD:-0}"
RUN_POST_EVAL="${RUN_POST_EVAL:-1}"
SKIP_TRAIN_IF_DONE="${SKIP_TRAIN_IF_DONE:-1}"
POST_EVAL_FORCE="${POST_EVAL_FORCE:-0}"
EVAL_SWEEP_OVERRIDE="${EVAL_SWEEP_OVERRIDE:-}"

# Optional email notifications.
MAIL_TO="${MAIL_TO:-}"
MAIL_SUBJECT_PREFIX="${MAIL_SUBJECT_PREFIX:-slurm}"

# This standalone script only supports the colocated GRPO path.
REQUIRED_PLACEMENT_MODE="${REQUIRED_PLACEMENT_MODE:-trl_colocate}"

# -----------------------------
# Requeue handling (USR1)
# -----------------------------
REQUEUE_SENT=0
MAIN_PID=""
handle_usr1() {
  if (( REQUEUE_SENT == 1 )); then return 0; fi
  REQUEUE_SENT=1
  echo "[$(date)] USR1 received: requesting requeue..."
  scontrol requeue "$SLURM_JOB_ID" || true
  if [[ -n "${MAIN_PID}" ]] && kill -0 "${MAIN_PID}" 2>/dev/null; then
    kill -TERM -- -"${MAIN_PID}" 2>/dev/null || kill -TERM "${MAIN_PID}" 2>/dev/null || true
  fi
  exit 0
}
trap handle_usr1 USR1

# -----------------------------
# Email notifications (manual; skips requeue exits)
# -----------------------------
SECONDS=0
START_TS="$(date -Iseconds)"

send_mail() {
  local subject="$1"
  local body="$2"

  [[ -n "$MAIL_TO" ]] || return 0

  if command -v mail >/dev/null 2>&1; then
    printf "%s\n" "$body" | mail -s "$subject" "$MAIL_TO"
  elif command -v mailx >/dev/null 2>&1; then
    printf "%s\n" "$body" | mailx -s "$subject" "$MAIL_TO"
  elif command -v sendmail >/dev/null 2>&1; then
    {
      printf "To: %s\n" "$MAIL_TO"
      printf "Subject: %s\n" "$subject"
      printf "\n%s\n" "$body"
    } | sendmail -t
  else
    echo "[$(date)] WARNING: no mail/mailx/sendmail found; cannot send email." >&2
    return 1
  fi
}

notify_on_exit() {
  local rc=$?

  if (( REQUEUE_SENT == 1 )); then
    return 0
  fi

  set +e

  local end_ts dur out_log err_log status subject body
  end_ts="$(date -Iseconds)"
  dur="$SECONDS"
  out_log="logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-unknown}.out"
  err_log="logs/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-unknown}.err"

  if [[ "$rc" -eq 0 ]]; then
    status="COMPLETED"
  else
    status="FAILED (rc=$rc)"
  fi

  subject="${MAIL_SUBJECT_PREFIX}: ${SLURM_JOB_NAME:-job} ${SLURM_JOB_ID:-} ${status}"

  body=$(
    cat <<EOF2
Job notification: $status

JobName:   ${SLURM_JOB_NAME:-}
JobID:     ${SLURM_JOB_ID:-}
User:      ${SLURM_JOB_USER:-$USER}
Cluster:   ${SLURM_CLUSTER_NAME:-}
Partition: ${SLURM_JOB_PARTITION:-}
Nodes:     ${SLURM_JOB_NODELIST:-}
Workdir:   ${SLURM_SUBMIT_DIR:-$(pwd)}

Start:     $START_TS
End:       $end_ts
Duration:  ${dur}s

CONFIG:    ${CONFIG:-}
Stdout:    $out_log
Stderr:    $err_log
EOF2
  )

  send_mail "$subject" "$body" >/dev/null 2>&1 || true
  return 0
}
trap notify_on_exit EXIT

# -----------------------------
# Conda activation (cluster-specific)
# -----------------------------
module -q load miniforge/24.3.0-0

CONDA_BASE="${CONDA_BASE:-/orcd/software/core/001/pkg/miniforge/24.3.0-0}"
if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found at: $CONDA_BASE/etc/profile.d/conda.sh" >&2
  echo "Edit CONDA_BASE/module load lines to match your cluster." >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "[$(date)] Activating conda env: $CONDA_ENV"
conda activate "$CONDA_ENV"

echo "[$(date)] Python: $(command -v python)"
python -V

# -----------------------------
# Shared caches / misc knobs
# -----------------------------
JOB_ID="${SLURM_JOB_ID:-$$}"
LOCAL_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
SHARED_SCRATCH="${SCRATCH:-$HOME}"

export HF_HOME="${HF_HOME:-${LOCAL_SCRATCH}/hf/${USER}/${JOB_ID}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SHARED_SCRATCH}/hf/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"

export TRITON_CACHE_DIR_BASE="${TRITON_CACHE_DIR_BASE:-${LOCAL_SCRATCH}/triton_cache/${USER}/${JOB_ID}}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_SCRATCH}/torch_extensions/${USER}/${JOB_ID}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${LOCAL_SCRATCH}/vllm/${USER}/${JOB_ID}}"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$HF_ASSETS_CACHE" \
  "$TRITON_CACHE_DIR_BASE" \
  "$TORCH_EXTENSIONS_DIR" \
  "$VLLM_CACHE_ROOT"

export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

# -----------------------------
# Helper functions
# -----------------------------
pin_one_gpu_task() {
  local task_index="${SLURM_LOCALID:-0}"
  local cvd_raw="${CUDA_VISIBLE_DEVICES:-}"

  if [[ "$cvd_raw" == *,* ]]; then
    local -a cvd_arr=()
    IFS=',' read -r -a cvd_arr <<< "$cvd_raw"
    export CUDA_VISIBLE_DEVICES="${cvd_arr[$task_index]:-${cvd_arr[0]}}"
  elif [[ -z "$cvd_raw" ]]; then
    export CUDA_VISIBLE_DEVICES="$task_index"
  fi
}

launch_train_rank() {
  set -euo pipefail
  cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

  pin_one_gpu_task

  export RANK="${SLURM_PROCID:-0}"
  export WORLD_SIZE="${SLURM_NTASKS:-1}"
  export MASTER_ADDR
  export MASTER_PORT
  export LOCAL_RANK=0
  export RTT_NODE_LOCAL_TASK_ID="${SLURM_LOCALID:-0}"

  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_BASE}/task${SLURM_PROCID:-0}"
  mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

  echo "[$(date)] trainer-rank node=$(hostname -f 2>/dev/null || hostname) step_rank=${SLURM_PROCID:-0} global_rank=$RANK local_task=${SLURM_LOCALID:-0} world_size=$WORLD_SIZE master=${MASTER_ADDR}:${MASTER_PORT} CVD=${CUDA_VISIBLE_DEVICES:-}"
  exec python -m src.learn_persistent --config "$CONFIG"
}
export -f pin_one_gpu_task
export -f launch_train_rank

run_post_eval() {
  local eval_sweep="$1"
  local eval_total_gpus="$2"
  local eval_gpus_per_node="$3"
  local eval_nodes="$4"

  mkdir -p "$RTT_CFG_EVAL_OUTDIR"

  mapfile -t targets < <(python - <<'PY'
import json
import os
from pathlib import Path
from src.utils import load_config

cfg = load_config(os.environ["CONFIG"])
sweep = str(os.environ.get("EVAL_SWEEP", "final_only")).strip().lower()
outdir = Path(os.environ["RTT_CFG_EVAL_OUTDIR"])
coord_path = Path(cfg["paths"]["coord_dir"]) / "state.json"

st = {}
if coord_path.exists():
    try:
        st = json.loads(coord_path.read_text(encoding="utf-8"))
    except Exception:
        st = {}

targets = []

if sweep == "final_only":
    adapter = st.get("teacher_adapter") or ""
    if adapter:
        try:
            round_id = int(st.get("round", 0) or 0)
        except Exception:
            round_id = 0
        targets.append((round_id, str(adapter), str(outdir / f"teacher_eval.round_{round_id}.json")))

elif sweep == "all_rounds":
    adapter_root = Path(cfg.get("persistent", {}).get("adapter_root", "outputs/teacher_lora_persistent"))
    seen = set()

    for p in sorted(adapter_root.glob("round_*/final"), key=lambda x: x.parent.name):
        try:
            round_id = int(p.parent.name.split("_", 1)[1])
        except Exception:
            continue
        seen.add(round_id)
        targets.append((round_id, str(p), str(outdir / f"teacher_eval.round_{round_id}.json")))

    adapter = st.get("teacher_adapter") or ""
    if adapter:
        rp = Path(str(adapter))
        round_id = None
        try:
            if rp.parent.name.startswith("round_"):
                round_id = int(rp.parent.name.split("_", 1)[1])
        except Exception:
            round_id = None
        if round_id is None:
            try:
                round_id = int(st.get("round", 0) or 0)
            except Exception:
                round_id = 0
        if round_id not in seen and rp.exists():
            targets.append((round_id, str(rp), str(outdir / f"teacher_eval.round_{round_id}.json")))

    targets.sort(key=lambda x: x[0])

else:
    raise SystemExit(f"Unsupported EVAL_SWEEP={sweep!r}; expected final_only or all_rounds")

for round_id, adapter, outfile in targets:
    print(f"{round_id}\t{adapter}\t{outfile}")
PY
  )

  if (( ${#targets[@]} == 0 )); then
    echo "[$(date)] No post-train eval targets found for EVAL_SWEEP=$eval_sweep; skipping."
    return 0
  fi

  local cpus_per_gpu="$RTT_CFG_CPUS_PER_GPU"
  local target round adapter outfile

  for target in "${targets[@]}"; do
    IFS=$'\t' read -r round adapter outfile <<< "$target"

    if [[ -z "$adapter" ]]; then
      continue
    fi

    if [[ -f "$outfile" && "$POST_EVAL_FORCE" != "1" ]]; then
      echo "[$(date)] Post-eval output already exists; skipping round=$round outfile=$outfile"
      python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    last_teacher_eval_round=int("$round"),
    last_teacher_eval_path=str("$outfile"),
    last_teacher_eval_ts=float(time.time()),
    eval_inflight_round=None,
    eval_inflight_job_id=None,
    eval_inflight_ts=None,
)
PY
      continue
    fi

    echo "[$(date)] Running post-train eval for round=$round adapter=$adapter"

    python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    eval_inflight_round=int("$round"),
    eval_inflight_job_id="injob:${SLURM_JOB_ID}:round${round}",
    eval_inflight_ts=float(time.time()),
)
print(f"[grpo_after_only] set inflight round={int('$round')} job=injob:{'$SLURM_JOB_ID'}:round{int('$round')}")
PY

    export RTT_EVAL_RUN_ID="job${SLURM_JOB_ID}.round${round}.$(date +%s)"

    set +e
    srun --export=ALL,EVAL_SWEEP="$eval_sweep" \
         --nodes="$eval_nodes" \
         --ntasks="$eval_total_gpus" \
         --ntasks-per-node="$eval_gpus_per_node" \
         --gpus-per-task=1 \
         --gpu-bind=closest \
         --cpus-per-task="$cpus_per_gpu" \
         --kill-on-bad-exit=1 \
      python -m src.teacher_eval \
        --config "$CONFIG" \
        --adapter "$adapter" \
        --round "$round" \
        --parallelism slurm \
        --out "$outfile" &
    MAIN_PID=$!
    wait "$MAIN_PID"
    rc=$?
    MAIN_PID=""
    set -e

    if [[ "$rc" -ne 0 ]]; then
      echo "[$(date)] Post-eval failed for round=$round with rc=$rc"
      exit "$rc"
    fi

    python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    last_teacher_eval_round=int("$round"),
    last_teacher_eval_path=str("$outfile"),
    last_teacher_eval_ts=float(time.time()),
    eval_inflight_round=None,
    eval_inflight_job_id=None,
    eval_inflight_ts=None,
)
print(f"[grpo_after_only] updated coord last_teacher_eval_round={int('$round')}")
PY
  done
}

# -----------------------------
# Validate config + resolve runtime settings
# -----------------------------
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: CONFIG not found: $CONFIG" >&2
  exit 1
fi

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

read -r REPLAY_BUFFER_PATH COORD_STATE_PATH TRAIN_PHASE TRAIN_ROUND TRAIN_MAX_ROUNDS < <(
  python - <<'PY'
import json
import os
from pathlib import Path
from src.utils import load_config

cfg = load_config(os.environ["CONFIG"])
replay = Path(cfg["paths"]["replay_buffer_path"])
state = Path(cfg["paths"]["coord_dir"]) / "state.json"
phase = ""
round_id = ""
max_rounds = ""
if state.exists():
    try:
        st = json.loads(state.read_text(encoding="utf-8"))
        phase = str(st.get("phase", "") or "")
        round_id = str(int(st.get("round", 0) or 0))
        max_rounds = str(int(st.get("max_rounds", 0) or 0))
    except Exception:
        phase = ""
        round_id = ""
        max_rounds = ""
print(str(replay), str(state), phase, round_id, max_rounds, sep='\t')
PY
)

if [[ "$RTT_CFG_PLACEMENT_MODE" != "$REQUIRED_PLACEMENT_MODE" ]]; then
  echo "ERROR: this standalone script only supports resources.placement_mode=$REQUIRED_PLACEMENT_MODE" >&2
  echo "Found resources.placement_mode=$RTT_CFG_PLACEMENT_MODE in $CONFIG" >&2
  exit 1
fi

if [[ ! -s "$REPLAY_BUFFER_PATH" ]]; then
  echo "ERROR: replay buffer missing or empty: $REPLAY_BUFFER_PATH" >&2
  echo "Run prep/cache/merge/mine on shared storage before using this GRPO-only script." >&2
  exit 1
fi

if [[ "$CLEAN_COORD" == "1" || ! -f "$COORD_STATE_PATH" ]]; then
  echo "[$(date)] Initializing coordinator state..."
  INIT_ARGS=(--config "$CONFIG")
  if [[ "$CLEAN_COORD" == "1" ]]; then
    INIT_ARGS+=(--clean)
  fi
  if [[ -n "$ROUNDS" ]]; then
    INIT_ARGS+=(--rounds "$ROUNDS")
  fi
  python -m src.init_coord "${INIT_ARGS[@]}"
fi

read -r TRAIN_PHASE TRAIN_ROUND TRAIN_MAX_ROUNDS < <(
  python - <<'PY'
import json
import os
from pathlib import Path
from src.utils import load_config

cfg = load_config(os.environ["CONFIG"])
state = Path(cfg["paths"]["coord_dir"]) / "state.json"
st = json.loads(state.read_text(encoding="utf-8"))
print(
    str(st.get("phase", "")),
    str(int(st.get("round", 0) or 0)),
    str(int(st.get("max_rounds", 0) or 0)),
    sep='\t',
)
PY
)

TRAIN_DONE=0
if [[ "$TRAIN_PHASE" == "done" ]] || [[ "$TRAIN_MAX_ROUNDS" != "" && "$TRAIN_ROUND" != "" && "$TRAIN_ROUND" -ge "$TRAIN_MAX_ROUNDS" ]]; then
  TRAIN_DONE=1
fi

TRAIN_GPUS="${TRAIN_GPUS:-$RTT_CFG_TOTAL_GPUS}"
CPUS_PER_GPU="${CPUS_PER_GPU:-$RTT_CFG_CPUS_PER_GPU}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR MASTER_PORT

EVAL_ENABLED="$RTT_CFG_TEACHER_EVAL_ENABLED"
EVAL_SCHEDULE="$RTT_CFG_TEACHER_EVAL_SCHEDULE"
EVAL_SWEEP="${EVAL_SWEEP_OVERRIDE:-$RTT_CFG_TEACHER_EVAL_SWEEP}"
EVAL_NODES="${EVAL_NODES:-1}"
EVAL_GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-$TRAIN_GPUS}"
EVAL_TOTAL_GPUS="${EVAL_TOTAL_GPUS:-$(( EVAL_NODES * EVAL_GPUS_PER_NODE ))}"

echo "[$(date)] CONFIG=$CONFIG"
echo "[$(date)] REPLAY_BUFFER_PATH=$REPLAY_BUFFER_PATH"
echo "[$(date)] RTT_CFG_PLACEMENT_MODE=$RTT_CFG_PLACEMENT_MODE"
echo "[$(date)] TRAIN_PHASE=$TRAIN_PHASE TRAIN_ROUND=$TRAIN_ROUND TRAIN_MAX_ROUNDS=$TRAIN_MAX_ROUNDS TRAIN_DONE=$TRAIN_DONE"
echo "[$(date)] TRAIN_GPUS=$TRAIN_GPUS CPUS_PER_GPU=$CPUS_PER_GPU"
echo "[$(date)] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "[$(date)] RUN_POST_EVAL=$RUN_POST_EVAL EVAL_ENABLED=$EVAL_ENABLED EVAL_SCHEDULE=$EVAL_SCHEDULE EVAL_SWEEP=$EVAL_SWEEP"

# -----------------------------
# Training
# -----------------------------
if (( TRAIN_DONE == 1 )) && (( SKIP_TRAIN_IF_DONE == 1 )); then
  echo "[$(date)] Training already complete according to coord state; skipping training."
else
  echo "[$(date)] Starting GRPO training..."
  set +e
  srun --export=ALL \
       --ntasks="$TRAIN_GPUS" \
       --ntasks-per-node="$TRAIN_GPUS" \
       --gpus-per-task=1 \
       --gpu-bind=closest \
       --cpus-per-task="$CPUS_PER_GPU" \
       --kill-on-bad-exit=1 \
       /bin/bash -lc 'launch_train_rank' &
  MAIN_PID=$!
  wait "$MAIN_PID"
  RC=$?
  MAIN_PID=""
  set -e

  if [[ "$RC" -ne 0 ]]; then
    echo "[$(date)] GRPO training exited with RC=$RC"
    exit "$RC"
  fi

  echo "[$(date)] GRPO training finished OK."
fi

# -----------------------------
# Post-train eval (optional)
# -----------------------------
if (( RUN_POST_EVAL == 1 )) && [[ "$EVAL_ENABLED" == "1" ]] && [[ "$EVAL_SCHEDULE" == "post_train" ]]; then
  echo "[$(date)] Running post-train eval sweep..."
  run_post_eval "$EVAL_SWEEP" "$EVAL_TOTAL_GPUS" "$EVAL_GPUS_PER_NODE" "$EVAL_NODES"
elif (( RUN_POST_EVAL == 1 )) && [[ "$EVAL_ENABLED" != "1" ]]; then
  echo "[$(date)] teacher_eval is disabled in config; skipping post-train eval."
elif (( RUN_POST_EVAL == 1 )) && [[ "$EVAL_SCHEDULE" != "post_train" ]]; then
  echo "[$(date)] teacher_eval schedule=$EVAL_SCHEDULE; standalone script only runs post-train eval, so skipping."
else
  echo "[$(date)] RUN_POST_EVAL=0; skipping post-train eval."
fi

echo "[$(date)] Done."
exit 0
