#!/bin/bash
#SBATCH --job-name=rtt-merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs data outputs

rtt_use_default_env
rtt_init_config
source slurm/common.sh

python -m src.merge_shards --config "$CONFIG" --what student_cache
python -m src.merge_shards --config "$CONFIG" --what teacher_cache
