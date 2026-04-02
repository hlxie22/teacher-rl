#!/bin/bash
#SBATCH --job-name=rtt-teval
#SBATCH --nodes=1
#SBATCH --partition=gpu
# Fallback resource shape only; eval_watch.slurm should override via sbatch CLI.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs

MAIN_PID=""
rtt_install_requeue_trap_main_pid MAIN_PID

rtt_use_infer_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

ROUND="${ROUND:?Must set ROUND (int)}"
ADAPTER="${ADAPTER:-}"
OUTFILE="${OUTFILE:-}"

EVAL_GPUS="${EVAL_GPUS:-$RTT_CFG_TEACHER_EVAL_GPUS}"
OUTDIR="$RTT_CFG_EVAL_OUTDIR"

mkdir -p "$OUTDIR"
if [[ -z "$OUTFILE" ]]; then
  OUTFILE="$OUTDIR/teacher_eval.round_${ROUND}.json"
fi

echo "[$(date)] teacher_eval_once starting:"
echo "  CONFIG=$CONFIG"
echo "  ROUND=$ROUND"
echo "  ADAPTER=$ADAPTER"
echo "  EVAL_GPUS=$EVAL_GPUS"
echo "  OUTFILE=$OUTFILE"

# Fast skip: if coord already says this round was evaluated, exit cleanly.
ALREADY_DONE="$(python - <<PY
import json
st_path = "$RTT_CFG_COORD_DIR/state.json"
try:
    st=json.load(open(st_path))
except Exception:
    st={}
last = int(st.get("last_teacher_eval_round", -1) or -1)
r = int("$ROUND")
print(1 if last >= r else 0)
PY
)"

if [[ "$ALREADY_DONE" == "1" ]]; then
  echo "[teacher_eval_once] already evaluated round=$ROUND; clearing inflight (if any) and exiting."
  python - <<PY
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
st=coord.read_state()
if int(st.get("eval_inflight_round", -999) or -999) == int("$ROUND"):
    coord.update_state(eval_inflight_round=None, eval_inflight_job_id=None, eval_inflight_ts=None)
PY
  exit 0
fi

# Split CPUs across GPU tasks
CPUS_PER_GPU=$(( (${SLURM_CPUS_PER_TASK:-16} + EVAL_GPUS - 1) / EVAL_GPUS ))
if (( CPUS_PER_GPU < 1 )); then CPUS_PER_GPU=1; fi

# Unique run id so vLLM partial dirs never collide
export RTT_EVAL_RUN_ID="job${SLURM_JOB_ID}.round${ROUND}.$(date +%s)"

# Run eval (task-parallel, 1 GPU per task)
srun --ntasks="$EVAL_GPUS" --gpus-per-task=1 --gpu-bind=closest \
     --cpus-per-task="$CPUS_PER_GPU" --export=ALL \
  python -m src.teacher_eval \
    --config "$CONFIG" \
    --adapter "$ADAPTER" \
    --round "$ROUND" \
    --parallelism slurm \
    --out "$OUTFILE" &

MAIN_PID=$!
wait "$MAIN_PID"
MAIN_PID=""

# If we got here, eval succeeded: update coord state.
python - <<PY
import time
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    last_teacher_eval_round=int("$ROUND"),
    last_teacher_eval_path=str("$OUTFILE"),
    last_teacher_eval_ts=float(time.time()),
    eval_inflight_round=None,
    eval_inflight_job_id=None,
    eval_inflight_ts=None,
)
print(f"[teacher_eval_once] updated coord last_teacher_eval_round={int('$ROUND')}")
PY

echo "[$(date)] teacher_eval_once done."