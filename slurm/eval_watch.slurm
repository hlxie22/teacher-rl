#!/bin/bash
#SBATCH --job-name=rtt-evalwatch
#SBATCH --nodes=1
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs
rtt_install_requeue_trap_simple

rtt_use_default_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

POLL="${POLL:-600}"
RESUBMIT_TTL="${RESUBMIT_TTL:-7200}"
EVAL_GPUS="${EVAL_GPUS:-$RTT_CFG_TEACHER_EVAL_GPUS}"
CPUS_PER_GPU="${CPUS_PER_GPU:-$RTT_CFG_CPUS_PER_GPU}"
MEM_PER_NODE_GB="${MEM_PER_NODE_GB:-$RTT_CFG_MEM_PER_NODE_GB}"

EVAL_CPUS_PER_TASK=$(( EVAL_GPUS * CPUS_PER_GPU ))

# Read relevant state fields in one shot
get_state() {
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

# Print tab-separated (shell-friendly)
def fmt(x):
    if x is None:
        return ""
    return str(x)
print(r, phase, adapter, last, fmt(infr), fmt(infj), fmt(inft), sep="\t")
PY
}

job_alive() {
  local JOBID="$1"
  if [[ -z "$JOBID" ]]; then
    return 1
  fi
  [[ -n "$(squeue -j "$JOBID" -h -o '%A' 2>/dev/null)" ]]
}

ENABLED="$RTT_CFG_TEACHER_EVAL_ENABLED"
EVERY="$RTT_CFG_TEACHER_EVAL_EVERY_ROUNDS"
OUTDIR="$RTT_CFG_EVAL_OUTDIR"
mkdir -p "$OUTDIR"

echo "[$(date)] eval_watch (CPU) starting:"
echo "  CONFIG=$CONFIG"
echo "  POLL=$POLL"
echo "  RESUBMIT_TTL=$RESUBMIT_TTL"
echo "  EVAL_GPUS=$EVAL_GPUS"
echo "  CPUS_PER_GPU=$CPUS_PER_GPU"
echo "  EVAL_CPUS_PER_TASK=$EVAL_CPUS_PER_TASK"
echo "  MEM_PER_NODE_GB=$MEM_PER_NODE_GB"
echo "  enabled=$ENABLED"
echo "  every_rounds=$EVERY"
echo "  outdir=$OUTDIR"

if [[ "$ENABLED" != "1" ]]; then
  echo "[$(date)] teacher_eval disabled in config; exiting."
  exit 0
fi

if [[ "${RTT_CFG_TEACHER_EVAL_SCHEDULE:-online}" != "online" ]]; then
  echo "[$(date)] teacher_eval schedule=${RTT_CFG_TEACHER_EVAL_SCHEDULE}; online watcher not needed. Exiting."
  exit 0
fi

while true; do
  IFS=$'\t' read -r R PHASE ADAPTER LAST INFR INFJ INFT <<<"$(get_state)"

  DUE=0
  if [[ -n "$ADAPTER" ]] && [[ "$R" != "$LAST" ]] && (( R % EVERY == 0 )); then
    DUE=1
  fi

  if (( DUE == 1 )); then
    # If inflight exists, check if it's still alive; if dead or too old, allow resubmit.
    SHOULD_SUBMIT=1
    if [[ -n "$INFR" ]] && [[ "$INFR" == "$R" ]]; then
      # inflight for this same round
      if job_alive "$INFJ"; then
        SHOULD_SUBMIT=0
      else
        # If we know when it was set, enforce TTL
        NOW="$(date +%s)"
        if [[ -n "$INFT" ]]; then
          AGE=$(( NOW - ${INFT%.*} ))
          if (( AGE < RESUBMIT_TTL )); then
            SHOULD_SUBMIT=0
          fi
        fi
      fi
    fi

    if (( SHOULD_SUBMIT == 1 )); then
      OUTFILE="$OUTDIR/teacher_eval.round_${R}.json"

      echo "[$(date)] Submitting GPU eval for round=$R adapter=$ADAPTER"
      JOBID=$(sbatch --parsable \
        --gres=gpu:"$EVAL_GPUS" \
        --cpus-per-task="$EVAL_CPUS_PER_TASK" \
        --mem="${MEM_PER_NODE_GB}G" \
        --export=ALL,CONFIG="$CONFIG",ROUND="$R",ADAPTER="$ADAPTER",OUTFILE="$OUTFILE",EVAL_GPUS="$EVAL_GPUS" \
        slurm/teacher_eval_once.slurm)

      echo "[$(date)] Submitted eval job: $JOBID"
      python - <<PY
import time
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    eval_inflight_round=int("$R"),
    eval_inflight_job_id=str("$JOBID"),
    eval_inflight_ts=float(time.time()),
)
print(f"[eval_watch] set inflight round={int('$R')} job={str('$JOBID')}")
PY
    fi
  fi

  # Once training is marked done, keep watching until any due eval for the
  # final round has been submitted and completed (or no eval is due at all).
  if [[ "${PHASE,,}" == "done" ]] && (( DUE == 0 )); then
    echo "[$(date)] coord phase=done and no teacher eval is due; exiting watcher."
    exit 0
  fi

  sleep "$POLL"
done