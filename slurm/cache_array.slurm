#!/bin/bash
#SBATCH --job-name=rtt-cache
#SBATCH --nodes=1
#SBATCH --partition=gpu
# Fallback resource shape only; full_pipeline.slurm should override via sbatch CLI.
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@900

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs data outputs

MAIN_PID=""
rtt_install_requeue_trap_main_pid MAIN_PID

rtt_use_infer_env
rtt_init_config
source slurm/common.sh

eval "$(python -m src.config_helpers shell --config "$CONFIG")"

WHO="${WHO:?WHO must be student or teacher}"
if [[ "$WHO" != "student" && "$WHO" != "teacher" ]]; then
  echo "ERROR: WHO must be student or teacher, got: $WHO"
  exit 1
fi

GPUS_PER_JOB="${GPUS_PER_JOB:-$RTT_CFG_GPUS_PER_JOB}"
PARALLELISM="${PARALLELISM:-$RTT_CFG_CACHE_PARALLELISM}"

echo "[$(date)] cache_array:"
echo "  WHO=$WHO"
echo "  CONFIG=$CONFIG"
echo "  GPUS_PER_JOB=$GPUS_PER_JOB"
echo "  PARALLELISM=$PARALLELISM"
echo "  SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "  SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

CPUS_PER_GPU=$(( (${SLURM_CPUS_PER_TASK:-16} + GPUS_PER_JOB - 1) / GPUS_PER_JOB ))
if (( CPUS_PER_GPU < 1 )); then CPUS_PER_GPU=1; fi

if [[ "$PARALLELISM" == "tp" ]]; then
  srun --ntasks=1 --gres=gpu:"$GPUS_PER_JOB" --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" --export=ALL \
    python -m src.cache_rollouts --config "$CONFIG" --who "$WHO" --parallelism tp &
else
  srun --ntasks="$GPUS_PER_JOB" --gpus-per-task=1 --gpu-bind=closest \
       --cpus-per-task="$CPUS_PER_GPU" --export=ALL \
    python -m src.cache_rollouts --config "$CONFIG" --who "$WHO" --parallelism slurm &
fi

MAIN_PID=$!
wait "$MAIN_PID"
