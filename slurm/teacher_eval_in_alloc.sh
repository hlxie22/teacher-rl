#!/bin/bash
set -euo pipefail

source slurm/lib.sh
rtt_job_prelude logs

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

echo "[$(date)] teacher_eval_in_alloc starting:"
echo "  CONFIG=$CONFIG"
echo "  ROUND=$ROUND"
echo "  ADAPTER=$ADAPTER"
echo "  EVAL_GPUS=$EVAL_GPUS"
echo "  OUTFILE=$OUTFILE"

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
  echo "[teacher_eval_in_alloc] already evaluated round=$ROUND; clearing inflight (if any) and exiting."
  if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
    python - <<PY
from src.coord import Coordinator
coord=Coordinator("$RTT_CFG_COORD_DIR")
st=coord.read_state()
if int(st.get("eval_inflight_round", -999) or -999) == int("$ROUND"):
    coord.update_state(eval_inflight_round=None, eval_inflight_job_id=None, eval_inflight_ts=None)
PY
  fi
  exit 0
fi

export RTT_EVAL_RUN_ID="job${SLURM_JOB_ID}.round${ROUND}.$(date +%s)"

python -m src.teacher_eval \
  --config "$CONFIG" \
  --adapter "$ADAPTER" \
  --round "$ROUND" \
  --parallelism slurm \
  --out "$OUTFILE"

if [[ "${SLURM_PROCID:-0}" == "0" ]]; then
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
print(f"[teacher_eval_in_alloc] updated coord last_teacher_eval_round={int('$ROUND')}")
PY
fi

echo "[$(date)] teacher_eval_in_alloc done."
