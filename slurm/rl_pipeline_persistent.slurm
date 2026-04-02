#!/bin/bash
#SBATCH --job-name=rtt-rlpipe
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs

rtt_use_default_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"
eval "$(python -m src.placement shell --config "$CONFIG")"

ROUNDS="${ROUNDS:-$RTT_CFG_PERSISTENT_ROUNDS}"
CLEAN="${CLEAN:-$RTT_CFG_PERSISTENT_CLEAN_COORD}"

LEARN_PARTITION="${LEARN_PARTITION:-gpu}"
LEARN_WALLTIME="${LEARN_WALLTIME:-06:00:00}"
LEARN_RETRIES="${LEARN_RETRIES:-10}"
LEARN_RETRY_SLEEP="${LEARN_RETRY_SLEEP:-60}"

EVAL_WALLTIME="${EVAL_WALLTIME:-06:00:00}"
EVAL_SCHEDULE="${EVAL_SCHEDULE:-$RTT_CFG_TEACHER_EVAL_SCHEDULE}"
EVAL_SWEEP="${EVAL_SWEEP:-$RTT_CFG_TEACHER_EVAL_SWEEP}"

IFS=',' read -r -a RL_COMPONENT_GPU_COUNTS <<< "$RTT_PLACEMENT_COMPONENT_GPU_COUNTS"
if (( ${#RL_COMPONENT_GPU_COUNTS[@]} < 1 )); then
  echo "ERROR: placement planner returned no learner components"
  exit 1
fi

echo "[$(date)] rl_pipeline_persistent starting:"
echo "  CONFIG=$CONFIG"
echo "  placement_mode=$RTT_CFG_PLACEMENT_MODE"
echo "  packing_policy=$RTT_CFG_PACKING_POLICY"
echo "  placement_manifest=$RTT_PLACEMENT_MANIFEST"
echo "  component_gpu_counts=$RTT_PLACEMENT_COMPONENT_GPU_COUNTS"
echo "  eval_colocate=$RTT_PLACEMENT_EVAL_COLOCATE"
echo "  eval_component_id=${RTT_PLACEMENT_EVAL_COMPONENT_ID:-}"
echo "  eval_gpus=${RTT_PLACEMENT_EVAL_GPUS:-0}"
echo "  eval_reason=${RTT_PLACEMENT_EVAL_REASON:-}"
echo "  cpus_per_node=12"
echo "  mem_per_node_gb=$RTT_PLACEMENT_MEM_PER_NODE_GB"
echo "  learn_partition=$LEARN_PARTITION"
echo "  learn_walltime=$LEARN_WALLTIME"
echo "  teacher_eval_enabled=$RTT_CFG_TEACHER_EVAL_ENABLED"
echo "  teacher_eval_schedule=$EVAL_SCHEDULE"
echo "  teacher_eval_sweep=$EVAL_SWEEP"

INIT_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG",ROUNDS="$ROUNDS",CLEAN="$CLEAN" slurm/init_coord.slurm)
echo "init job: $INIT_JOB"

SBATCH_CMD=(
  sbatch --parsable
  --dependency=afterok:"$INIT_JOB"
  --export=ALL,CONFIG="$CONFIG",PLACEMENT_MANIFEST="$RTT_PLACEMENT_MANIFEST"
)

for idx in "${!RL_COMPONENT_GPU_COUNTS[@]}"; do
  gpu_count="${RL_COMPONENT_GPU_COUNTS[$idx]}"
  if [[ -z "$gpu_count" ]]; then
    echo "ERROR: empty GPU count in placement component list"
    exit 1
  fi

  SBATCH_CMD+=(
    --partition="$LEARN_PARTITION"
    --time="$LEARN_WALLTIME"
    --nodes=1
    --gres=gpu:"$gpu_count"
    --cpus-per-task=12
    --mem="${RTT_PLACEMENT_MEM_PER_NODE_GB}G"
  )

  if (( idx + 1 < ${#RL_COMPONENT_GPU_COUNTS[@]} )); then
    SBATCH_CMD+=(:)
  fi
done

SBATCH_CMD+=(slurm/rl_learner_persistent.slurm)

LEARN_JOB=""
LEARN_LAST_ERR=""

echo "[$(date)] learner submit retries: attempts=$LEARN_RETRIES sleep_s=$LEARN_RETRY_SLEEP"
printf '[%s] learner sbatch command:' "$(date)"
printf ' %q' "${SBATCH_CMD[@]}"
echo

submit_learner_job() {
  local out rc
  set +e
  out="$(${SBATCH_CMD[@]} 2>&1)"
  rc=$?
  set -e
  printf '%s' "$out"
  return "$rc"
}

for (( attempt=1; attempt<=LEARN_RETRIES; attempt++ )); do
  echo "[$(date)] learner submit attempt $attempt/$LEARN_RETRIES"

  set +e
  LEARN_SUBMIT_OUT="$(submit_learner_job)"
  RC=$?
  set -e

  if (( RC == 0 )); then
    LEARN_JOB="$LEARN_SUBMIT_OUT"
    break
  fi

  LEARN_LAST_ERR="$LEARN_SUBMIT_OUT"
  echo "[$(date)] learner sbatch failed (attempt $attempt/$LEARN_RETRIES, rc=$RC):"
  echo "$LEARN_LAST_ERR"

  if (( attempt < LEARN_RETRIES )); then
    echo "[$(date)] sleeping ${LEARN_RETRY_SLEEP}s before retry..."
    sleep "$LEARN_RETRY_SLEEP"
  fi
done

if [[ -z "$LEARN_JOB" ]]; then
  echo "ERROR: learner sbatch failed after $LEARN_RETRIES attempts"
  echo "$LEARN_LAST_ERR"
  exit 1
fi

echo "learner job: $LEARN_JOB (placement_mode=$RTT_CFG_PLACEMENT_MODE, packing_policy=$RTT_CFG_PACKING_POLICY, components=$RTT_PLACEMENT_COMPONENT_GPU_COUNTS)"

if [[ "$RTT_CFG_TEACHER_EVAL_ENABLED" != "1" ]]; then
  echo "teacher eval: disabled"
elif [[ "$EVAL_SCHEDULE" == "online" ]]; then
  if [[ "$RTT_PLACEMENT_EVAL_COLOCATE" == "1" ]]; then
    echo "eval watcher: colocated inside learner allocation (component_id=$RTT_PLACEMENT_EVAL_COMPONENT_ID, eval_gpus=$RTT_PLACEMENT_EVAL_GPUS)"
  else
    EVAL_JOB=$(sbatch --parsable \
      --dependency=afterok:"$INIT_JOB" \
      --export=ALL,CONFIG="$CONFIG",GPUS_PER_JOB="$RTT_CFG_GPUS_PER_JOB" \
      slurm/eval_watch.slurm)
    echo "eval watcher job: $EVAL_JOB"
  fi
elif [[ "$EVAL_SCHEDULE" == "post_train" ]]; then
  # Reuse the same node count as training, but request full GPU count on each node.
  # This keeps peak nodes low and gives eval the whole post-train footprint.
  POST_EVAL_NODES="${POST_EVAL_NODES:-$RTT_CFG_TOTAL_NODES}"
  POST_EVAL_GPUS_PER_NODE="${POST_EVAL_GPUS_PER_NODE:-$RTT_CFG_GPUS_PER_NODE}"
  POST_EVAL_TOTAL_GPUS="${POST_EVAL_TOTAL_GPUS:-$(( POST_EVAL_NODES * POST_EVAL_GPUS_PER_NODE ))}"
  POST_EVAL_CPUS_PER_TASK=$(( POST_EVAL_GPUS_PER_NODE * RTT_CFG_CPUS_PER_GPU ))

  EVAL_JOB=$(sbatch --parsable \
    --dependency=afterok:"$LEARN_JOB" \
    --partition="$LEARN_PARTITION" \
    --time="$EVAL_WALLTIME" \
    --nodes="$POST_EVAL_NODES" \
    --gres=gpu:"$POST_EVAL_GPUS_PER_NODE" \
    --cpus-per-task="$POST_EVAL_CPUS_PER_TASK" \
    --mem="${RTT_CFG_MEM_PER_NODE_GB}G" \
    --export=ALL,CONFIG="$CONFIG",EVAL_NODES="$POST_EVAL_NODES",EVAL_GPUS_PER_NODE="$POST_EVAL_GPUS_PER_NODE",EVAL_TOTAL_GPUS="$POST_EVAL_TOTAL_GPUS",EVAL_SWEEP="$EVAL_SWEEP" \
    slurm/teacher_eval_sweep.slurm)

  echo "post-train eval job: $EVAL_JOB (nodes=$POST_EVAL_NODES, gpus_per_node=$POST_EVAL_GPUS_PER_NODE, total_gpus=$POST_EVAL_TOTAL_GPUS, sweep=$EVAL_SWEEP)"
else
  echo "ERROR: unsupported teacher eval schedule: $EVAL_SCHEDULE"
  exit 1
fi

echo "GRPO training launched. Coordination: data/coord_rl/state.json"