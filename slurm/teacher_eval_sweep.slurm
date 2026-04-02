#!/bin/bash
#SBATCH --job-name=rtt-teval-sweep
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

MAIN_PID=""
rtt_install_requeue_trap_main_pid MAIN_PID

rtt_use_infer_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

if [[ "$RTT_CFG_TEACHER_EVAL_ENABLED" != "1" ]]; then
  echo "[$(date)] teacher_eval disabled in config; exiting."
  exit 0
fi

EVAL_SWEEP="${EVAL_SWEEP:-$RTT_CFG_TEACHER_EVAL_SWEEP}"
EVAL_NODES="${EVAL_NODES:-${SLURM_JOB_NUM_NODES:-1}}"
EVAL_GPUS_PER_NODE="${EVAL_GPUS_PER_NODE:-$RTT_CFG_GPUS_PER_NODE}"
EVAL_TOTAL_GPUS="${EVAL_TOTAL_GPUS:-$(( EVAL_NODES * EVAL_GPUS_PER_NODE ))}"
CPUS_PER_GPU="${RTT_CFG_CPUS_PER_GPU}"
OUTDIR="$RTT_CFG_EVAL_OUTDIR"

mkdir -p "$OUTDIR"

echo "[$(date)] teacher_eval_sweep starting:"
echo "  CONFIG=$CONFIG"
echo "  EVAL_SWEEP=$EVAL_SWEEP"
echo "  EVAL_NODES=$EVAL_NODES"
echo "  EVAL_GPUS_PER_NODE=$EVAL_GPUS_PER_NODE"
echo "  EVAL_TOTAL_GPUS=$EVAL_TOTAL_GPUS"
echo "  CPUS_PER_GPU=$CPUS_PER_GPU"
echo "  OUTDIR=$OUTDIR"

mapfile -t TARGETS < <(python - <<'PY'
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

if (( ${#TARGETS[@]} == 0 )); then
  echo "[$(date)] no teacher adapters found for EVAL_SWEEP=$EVAL_SWEEP; exiting."
  exit 0
fi

for target in "${TARGETS[@]}"; do
  IFS=$'\t' read -r ROUND ADAPTER OUTFILE <<<"$target"

  if [[ -z "$ADAPTER" ]]; then
    continue
  fi

  echo "[$(date)] eval target: round=$ROUND adapter=$ADAPTER outfile=$OUTFILE"

  if [[ -f "$OUTFILE" ]]; then
    echo "[$(date)] output already exists; skipping run and refreshing coord state."
    python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    last_teacher_eval_round=int("$ROUND"),
    last_teacher_eval_path=str("$OUTFILE"),
    last_teacher_eval_ts=float(time.time()),
    eval_inflight_round=None,
    eval_inflight_job_id=None,
    eval_inflight_ts=None,
)
PY
    continue
  fi

  python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    eval_inflight_round=int("$ROUND"),
    eval_inflight_job_id="posttrain:${SLURM_JOB_ID}:round${ROUND}",
    eval_inflight_ts=float(time.time()),
)
print(f"[teacher_eval_sweep] set inflight round={int('$ROUND')} job=posttrain:{'$SLURM_JOB_ID'}:round{int('$ROUND')}")
PY

  export RTT_EVAL_RUN_ID="job${SLURM_JOB_ID}.round${ROUND}.$(date +%s)"

  srun --nodes="$EVAL_NODES" \
       --ntasks="$EVAL_TOTAL_GPUS" \
       --ntasks-per-node="$EVAL_GPUS_PER_NODE" \
       --gpus-per-task=1 \
       --gpu-bind=closest \
       --cpus-per-task="$CPUS_PER_GPU" \
       --kill-on-bad-exit=1 \
       --export=ALL \
    python -m src.teacher_eval \
      --config "$CONFIG" \
      --adapter "$ADAPTER" \
      --round "$ROUND" \
      --parallelism slurm \
      --out "$OUTFILE" &

  MAIN_PID=$!
  wait "$MAIN_PID"
  MAIN_PID=""

  python - <<PY
import time
from src.coord import Coordinator
coord = Coordinator("$RTT_CFG_COORD_DIR")
coord.update_state(
    last_teacher_eval_round=int("$ROUND"),
    last_teacher_eval_path=str("$OUTFILE"),
    last_teacher_eval_ts=float(time.time()),
    eval_inflight_round=None,
    eval_inflight_job_id=None,
    eval_inflight_ts=None,
)
print(f"[teacher_eval_sweep] updated coord last_teacher_eval_round={int('$ROUND')}")
PY
done

echo "[$(date)] teacher_eval_sweep done."