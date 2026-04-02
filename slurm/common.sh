#!/bin/bash
# Common SLURM bootstrap: venv + deps + caches
# Intended to be: source slurm/common.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# ---- choose python (require 3.12 explicitly) ----
PYTHON_VERSION="${PYTHON_VERSION:-}"
PYTHON="${PYTHON:-}"

if [[ -z "${PYTHON}" && -n "${PYTHON_VERSION}" ]]; then
  PYTHON="$(command -v "python${PYTHON_VERSION}" || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  PYTHON="$(command -v python3.12 || true)"
fi
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: Python 3.12 is required but python3.12 was not found on PATH."
  exit 1
fi

SELECTED_PY_MM="$("$PYTHON" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

if [[ "$SELECTED_PY_MM" != "3.12" ]]; then
  echo "ERROR: Selected PYTHON=$PYTHON is Python $SELECTED_PY_MM, but Python 3.12 is required."
  exit 1
fi

echo "[$(date)] Using PYTHON=$PYTHON"
"$PYTHON" -V

# ---- profile-aware defaults ----
REQ_PROFILE="${REQ_PROFILE:-default}"

case "$REQ_PROFILE" in
  infer)
    : "${VENV_DIR:=$SLURM_SUBMIT_DIR/.venv_infer}"
    : "${REQ_FILE:=$SLURM_SUBMIT_DIR/requirements.qwen35-infer.txt}"
    ;;
  train)
    : "${VENV_DIR:=$SLURM_SUBMIT_DIR/.venv_train}"
    : "${REQ_FILE:=$SLURM_SUBMIT_DIR/requirements.qwen35-train.txt}"
    ;;
  *)
    : "${VENV_DIR:=$SLURM_SUBMIT_DIR/.venv}"
    : "${REQ_FILE:=$SLURM_SUBMIT_DIR/requirements.txt}"
    ;;
esac

if [[ ! -f "$REQ_FILE" ]]; then
  echo "ERROR: requirements file not found: $REQ_FILE"
  exit 1
fi

echo "[$(date)] Bootstrap settings:"
echo "  REQ_PROFILE=$REQ_PROFILE"
echo "  VENV_DIR=$VENV_DIR"
echo "  REQ_FILE=$REQ_FILE"

# ---- install knobs that must participate in dependency hashing ----
VLLM_SPEC="${VLLM_SPEC:-vllm}"
VLLM_EXTRA_INDEX_URL="${VLLM_EXTRA_INDEX_URL:-https://wheels.vllm.ai/nightly}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

# Used only as a targeted fallback when transformers import fails with:
#   ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub'
HF_HUB_REPAIR_SPEC="${HF_HUB_REPAIR_SPEC:-huggingface-hub @ git+https://github.com/huggingface/huggingface_hub.git}"

# ---- dependency sentinel ----
REQ_HASH="$("$PYTHON" - <<PY
import hashlib, pathlib
parts = []
parts.append(pathlib.Path("$REQ_FILE").read_bytes())
parts.append("\nREQ_PROFILE=$REQ_PROFILE\n".encode("utf-8"))
parts.append("\nVLLM_SPEC=$VLLM_SPEC\n".encode("utf-8"))
parts.append("\nVLLM_EXTRA_INDEX_URL=$VLLM_EXTRA_INDEX_URL\n".encode("utf-8"))
parts.append("\nUV_TORCH_BACKEND=$UV_TORCH_BACKEND\n".encode("utf-8"))
parts.append("\nHF_HUB_REPAIR_SPEC=$HF_HUB_REPAIR_SPEC\n".encode("utf-8"))
parts.append(b"\nCOMMON_SH_REV=17\n")
print(hashlib.sha256(b"".join(parts)).hexdigest())
PY
)"
DEPS_SENTINEL="$VENV_DIR/.deps.${REQ_HASH}"
BOOT_SENTINEL="$VENV_DIR/.bootstrap.done"

LOCK_SUFFIX="$("$PYTHON" - <<PY
import hashlib
print(hashlib.sha256("$VENV_DIR".encode("utf-8")).hexdigest()[:12])
PY
)"
LOCKDIR=""
LOCKFILE=""
LOCK_FD=""

cleanup_common_lock() {
  if [[ -n "${LOCK_FD:-}" ]]; then
    flock -u "$LOCK_FD" 2>/dev/null || true
    eval "exec ${LOCK_FD}>&-" 2>/dev/null || true
    LOCK_FD=""
  fi
  if [[ -n "${LOCKDIR:-}" ]]; then
    rmdir "$LOCKDIR" 2>/dev/null || true
    LOCKDIR=""
  fi
}

# Serialize venv mutations across array tasks without clobbering caller traps.
if command -v flock >/dev/null 2>&1; then
  LOCKFILE="$SLURM_SUBMIT_DIR/.venv_install.${LOCK_SUFFIX}.lock"
  echo "[$(date)] Acquiring venv install lock: $LOCKFILE"
  exec {LOCK_FD}> "$LOCKFILE"
  flock "$LOCK_FD"
else
  LOCKDIR="$SLURM_SUBMIT_DIR/.venv_install.${LOCK_SUFFIX}.lockdir"
  while ! mkdir "$LOCKDIR" 2>/dev/null; do
    echo "[$(date)] Another task is setting up this venv; waiting..."
    sleep 5
  done
fi
trap cleanup_common_lock EXIT

# Recreate the venv if it exists but is not Python 3.12
if [[ -x "$VENV_DIR/bin/python" ]]; then
  VENV_PY_MM="$("$VENV_DIR/bin/python" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  if [[ "$VENV_PY_MM" != "3.12" ]]; then
    echo "[$(date)] Existing venv uses Python $VENV_PY_MM; recreating with Python 3.12"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  rm -rf "$VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[$(date)] Activated venv: $(which python)"
python -V

ACTIVE_PY_MM="$(python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
if [[ "$ACTIVE_PY_MM" != "3.12" ]]; then
  echo "ERROR: Active venv python is $ACTIVE_PY_MM, expected 3.12"
  exit 1
fi

bootstrap_python_tools() {
  echo "[$(date)] Bootstrapping pip/setuptools/wheel/uv ..."
  python -m pip install -U pip wheel setuptools uv
}

install_sqlite_shim() {
  echo "[$(date)] Installing sqlite shim ..."
  uv pip install -U pysqlite3-binary

  SITE_PKGS="$(python - <<'PY'
import site
paths = site.getsitepackages()
print(paths[0])
PY
)"

  cat > "${SITE_PKGS}/sitecustomize.py" <<'PY'
import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
PY
}

verify_transformers_import() {
  python - <<'PY'
import huggingface_hub
import transformers
print("transformers =", transformers.__version__)
print("huggingface_hub =", huggingface_hub.__version__)
PY
}

repair_hf_stack_if_needed() {
  if [[ "$REQ_PROFILE" != "train" && "$REQ_PROFILE" != "infer" ]]; then
    return 0
  fi

  echo "[$(date)] Verifying Transformers/HF Hub import compatibility ..."
  set +e
  IMPORT_OUT="$({
    python - <<'PY' 2>&1
import huggingface_hub
import transformers
print("transformers =", transformers.__version__)
print("huggingface_hub =", huggingface_hub.__version__)
PY
  })"
  RC=$?
  set -e

  if [[ "$RC" == "0" ]]; then
    echo "$IMPORT_OUT"
    return 0
  fi

  echo "$IMPORT_OUT"

  if grep -q "cannot import name 'is_offline_mode' from 'huggingface_hub'" <<<"$IMPORT_OUT"; then
    echo "[$(date)] Detected HF Hub / Transformers import mismatch."
    echo "[$(date)] Repairing by upgrading huggingface-hub from git main ..."
    uv pip install -U "$HF_HUB_REPAIR_SPEC"

    echo "[$(date)] Re-checking Transformers/HF Hub import after repair ..."
    verify_transformers_import
    return 0
  fi

  echo "[$(date)] ERROR: Transformers import failed for an unexpected reason."
  return "$RC"
}

verify_transformers_qwen35() {
  if [[ "$REQ_PROFILE" != "train" && "$REQ_PROFILE" != "infer" ]]; then
    return 0
  fi

  echo "[$(date)] Verifying installed Transformers supports qwen3_5 ..."
  python - <<'PY'
from transformers import __version__
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

print("transformers =", __version__)
if "qwen3_5" not in CONFIG_MAPPING:
    raise SystemExit(
        "Installed transformers does not support model_type=qwen3_5. "
        "Bootstrap would fail later when loading Qwen3.5 checkpoints."
    )
print("qwen3_5 is supported")
PY
}

install_profile_deps() {
  local profile="$1"

  if [[ "$profile" == "infer" || "$profile" == "train" ]]; then
    export UV_TORCH_BACKEND
    echo "[$(date)] vLLM package spec: $VLLM_SPEC"
    echo "[$(date)] VLLM_EXTRA_INDEX_URL=$VLLM_EXTRA_INDEX_URL"
    echo "[$(date)] UV_TORCH_BACKEND=$UV_TORCH_BACKEND"
    echo "[$(date)] HF_HUB_REPAIR_SPEC=$HF_HUB_REPAIR_SPEC"
  fi

  if [[ "$profile" == "infer" ]]; then
    echo "[$(date)] Installing infer-profile requested Python deps ..."
    uv pip install -U -r "$REQ_FILE"

    echo "[$(date)] Installing vLLM for infer profile ..."
    if [[ -n "$VLLM_EXTRA_INDEX_URL" ]]; then
      uv pip install -U "$VLLM_SPEC" \
        --torch-backend="$UV_TORCH_BACKEND" \
        --extra-index-url "$VLLM_EXTRA_INDEX_URL"
    else
      uv pip install -U "$VLLM_SPEC" \
        --torch-backend="$UV_TORCH_BACKEND"
    fi

    echo "[$(date)] Re-applying infer-profile requested stack with deps ..."
    uv pip install -U -r "$REQ_FILE"

    repair_hf_stack_if_needed
    verify_transformers_qwen35
    install_sqlite_shim

  elif [[ "$profile" == "train" ]]; then
    echo "[$(date)] Installing train-profile requested Python deps ..."
    uv pip install -U -r "$REQ_FILE"

    repair_hf_stack_if_needed
    verify_transformers_qwen35
    install_sqlite_shim

  else
    echo "[$(date)] Installing default-profile Python deps ..."
    python -m pip install -r "$REQ_FILE"
  fi
}

if [[ ! -f "$BOOT_SENTINEL" ]]; then
  bootstrap_python_tools
  touch "$BOOT_SENTINEL"
fi

if [[ ! -f "$DEPS_SENTINEL" ]]; then
  echo "[$(date)] Installing deps from $REQ_FILE ..."
  install_profile_deps "$REQ_PROFILE"
  touch "$DEPS_SENTINEL"
else
  echo "[$(date)] Deps already installed for requirements hash; skipping install."
  repair_hf_stack_if_needed
  verify_transformers_qwen35
fi

# ensure imports
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

cleanup_common_lock
trap - EXIT

# ---- caches ----
JOB_ID="${SLURM_JOB_ID:-$$}"
LOCAL_SCRATCH="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
SHARED_SCRATCH="${SCRATCH:-$HOME}"

# For training, prefer node-local Hugging Face hub cache to avoid concurrent
# mutation of a shared cache across many distributed ranks/nodes.
# Keep datasets shared by default so prep/eval jobs do not re-download them.
if [[ "$REQ_PROFILE" == "train" ]]; then
  export HF_HOME="${HF_HOME:-${LOCAL_SCRATCH}/hf/${USER}/${JOB_ID}}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SHARED_SCRATCH}/hf/datasets}"
else
  export HF_HOME="${HF_HOME:-${SHARED_SCRATCH}/hf}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
fi

HUB_CACHE_DEFAULT="$HF_HOME/hub"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HUB_CACHE_DEFAULT}}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"

# Do not export deprecated TRANSFORMERS_CACHE; modern Transformers uses HF_HOME/HF_HUB_CACHE.
unset TRANSFORMERS_CACHE || true
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HUGGINGFACE_HUB_CACHE" "$HF_ASSETS_CACHE"

# ---- Triton / torch extensions / vLLM caches (avoid NFS races & stale handles) ----
TASK_ID="${SLURM_PROCID:-${SLURM_LOCALID:-0}}"

export TRITON_CACHE_DIR_BASE="${LOCAL_SCRATCH}/triton_cache/${USER}/${JOB_ID}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR_BASE}/task${TASK_ID}"
export TORCH_EXTENSIONS_DIR="${LOCAL_SCRATCH}/torch_extensions/${USER}/${JOB_ID}"
export VLLM_CACHE_ROOT="${LOCAL_SCRATCH}/vllm/${USER}/${JOB_ID}"

mkdir -p "$TRITON_CACHE_DIR_BASE" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

unset SLURM_MEM_PER_CPU SLURM_MEM_PER_GPU SLURM_MEM_PER_NODE

echo "[$(date)] Cache settings:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"
echo "  HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-<unset>}"
echo "  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
echo "  TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "  VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT"