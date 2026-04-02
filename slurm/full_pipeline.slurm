#!/bin/bash
#SBATCH --job-name=rtt-full
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs

rtt_use_default_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

CACHE_S_JOBS="${CACHE_S_JOBS:-$RTT_CFG_CACHE_STUDENT_JOBS}"
CACHE_T_JOBS="${CACHE_T_JOBS:-$RTT_CFG_CACHE_TEACHER_JOBS}"
GPUS_PER_JOB="${GPUS_PER_JOB:-$RTT_CFG_GPUS_PER_JOB}"
CPUS_PER_GPU="${CPUS_PER_GPU:-$RTT_CFG_CPUS_PER_GPU}"
MEM_PER_NODE_GB="${MEM_PER_NODE_GB:-$RTT_CFG_MEM_PER_NODE_GB}"

CACHE_CPUS_PER_TASK=$(( GPUS_PER_JOB * CPUS_PER_GPU ))

PREP_JOB=$(sbatch --parsable --export=ALL,CONFIG="$CONFIG" slurm/prep.slurm)
echo "prep job: $PREP_JOB"

CACHE_S=$(sbatch --parsable \
  --job-name=rtt-cacheS \
  --gres=gpu:"$GPUS_PER_JOB" \
  --cpus-per-task="$CACHE_CPUS_PER_TASK" \
  --mem="${MEM_PER_NODE_GB}G" \
  --dependency=afterok:"$PREP_JOB" \
  --array=0-$((CACHE_S_JOBS-1)) \
  --export=ALL,CONFIG="$CONFIG",GPUS_PER_JOB="$GPUS_PER_JOB",WHO=student \
  slurm/cache_array.slurm)
echo "cache student: $CACHE_S (jobs=$CACHE_S_JOBS, gpus_per_job=$GPUS_PER_JOB)"

CACHE_T=$(sbatch --parsable \
  --job-name=rtt-cacheT \
  --gres=gpu:"$GPUS_PER_JOB" \
  --cpus-per-task="$CACHE_CPUS_PER_TASK" \
  --mem="${MEM_PER_NODE_GB}G" \
  --dependency=afterok:"$PREP_JOB" \
  --array=0-$((CACHE_T_JOBS-1)) \
  --export=ALL,CONFIG="$CONFIG",GPUS_PER_JOB="$GPUS_PER_JOB",WHO=teacher \
  slurm/cache_array.slurm)
echo "cache teacher: $CACHE_T (jobs=$CACHE_T_JOBS, gpus_per_job=$GPUS_PER_JOB)"

MERGE_JOB=$(sbatch --parsable --dependency=afterok:"$CACHE_S":"$CACHE_T" --export=ALL,CONFIG="$CONFIG" slurm/merge_cache.slurm)
echo "merge job: $MERGE_JOB"

MINE_JOB=$(sbatch --parsable --dependency=afterok:"$MERGE_JOB" --export=ALL,CONFIG="$CONFIG" slurm/mine_buffer.slurm)
echo "mine job: $MINE_JOB"

RL_JOB=$(sbatch --parsable --dependency=afterok:"$MINE_JOB" --export=ALL,CONFIG="$CONFIG" slurm/rl_pipeline_persistent.slurm)
echo "rl pipeline (GRPO-only): $RL_JOB"
