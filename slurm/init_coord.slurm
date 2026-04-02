#!/bin/bash
#SBATCH --job-name=rtt-init
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source slurm/lib.sh
rtt_job_prelude logs data outputs

rtt_use_default_env
rtt_init_config
source slurm/common.sh

ROUNDS="${ROUNDS:-}"
CLEAN="${CLEAN:-0}"

ARGS=(--config "$CONFIG")
if [[ "$CLEAN" == "1" ]]; then
  ARGS+=(--clean)
fi
if [[ -n "$ROUNDS" ]]; then
  ARGS+=(--rounds "$ROUNDS")
fi

python -m src.init_coord "${ARGS[@]}"
