#!/bin/bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-rtt-grpo}"
CONDA_BASE="${CONDA_BASE:-/orcd/software/core/001/pkg/miniforge/24.3.0-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_YML="$SCRIPT_DIR/environment.yml"

if [[ ! -f "$ENV_YML" ]]; then
  echo "ERROR: environment.yml not found at $ENV_YML" >&2
  exit 1
fi

module -q load miniforge/24.3.0-0
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Create/update from configs/environment.yml
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda env create -n "$CONDA_ENV" -f "$ENV_YML"
else
  conda env update -n "$CONDA_ENV" -f "$ENV_YML" --prune
fi

conda activate "$CONDA_ENV"

echo "Using python: $(command -v python)"
python -V
echo "Repo root: $REPO_ROOT"

python -m pip install --upgrade pip

# environment.yml already provides most of the base stack
uv pip install -U \
  vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly

uv pip install -U \
  "transformers @ git+https://github.com/huggingface/transformers.git@main" \
  "trl==1.0.0" \
  torchvision

python - <<'PY'
import json
import os
import pathlib
import platform
import sys

manifest = {
    "python": sys.version.split()[0],
    "python_executable": sys.executable,
    "platform": platform.platform(),
}

for name in ["torch", "torchvision", "transformers", "trl", "vllm"]:
    try:
        m = __import__(name)
        manifest[name] = getattr(m, "__version__", "unknown")
    except Exception as e:
        manifest[name] = f"not-importable: {e}"

try:
    import torch
    manifest["torch_cuda"] = torch.version.cuda
except Exception:
    manifest["torch_cuda"] = None

prefix = pathlib.Path(os.environ["CONDA_PREFIX"])
out = prefix / "rtt_grpo_versions.json"
out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

print("Wrote:", out)
print(json.dumps(manifest, indent=2))
PY

python -m pip freeze | sort > "${CONDA_PREFIX}/rtt_grpo_pip_freeze.txt"

echo
echo "Environment is ready."
echo "Manifest: ${CONDA_PREFIX}/rtt_grpo_versions.json"
echo "Freeze:   ${CONDA_PREFIX}/rtt_grpo_pip_freeze.txt"