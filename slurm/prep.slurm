#!/bin/bash
#SBATCH --job-name=rtt-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs data outputs

rtt_use_default_env
rtt_init_config
source slurm/common.sh

python -m src.prep_data --config "$CONFIG"
