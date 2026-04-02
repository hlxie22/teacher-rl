#!/bin/bash
#SBATCH --job-name=rtt-cacheS
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
export WHO=student
exec /bin/bash slurm/cache_array.slurm