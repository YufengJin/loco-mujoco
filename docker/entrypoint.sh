#!/usr/bin/env bash
set -e

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/opt/venv}"

cd /workspace

# Determine loco-mujoco project root (may be /workspace/loco-mujoco when parent is mounted)
LOCO_ROOT=""
if [ -f /workspace/loco-mujoco/setup.py ]; then
  LOCO_ROOT="/workspace/loco-mujoco"
elif [ -f /workspace/setup.py ]; then
  LOCO_ROOT="/workspace"
fi

if [ -n "${LOCO_ROOT}" ]; then
  echo "Installing loco-mujoco from ${LOCO_ROOT} (editable)..."
  uv pip install -e "${LOCO_ROOT}"

  # Trigger first-import so loco_mujoco/__init__.py writes LOCOMUJOCO_VARIABLES.yaml
  # and ~/.loco-mujoco-caches/models is created. This is non-interactive.
  python -c "import loco_mujoco" >/dev/null 2>&1 || true

  # MyoSkeleton model is opt-in (license acceptance required).
  # Set LOCO_AUTO_MYO=1 to auto-accept and fetch; otherwise just print a hint.
  if [ "${LOCO_AUTO_MYO:-0}" = "1" ]; then
    echo "LOCO_AUTO_MYO=1 set — running loco-mujoco-myomodel-init..."
    yes | loco-mujoco-myomodel-init || true
  fi

  # Dataset presence hints (do not force-download)
  HF_CACHE="${HF_HOME:-/root/.cache/huggingface}"
  if [ ! -d "${HF_CACHE}/datasets" ] || [ -z "$(ls -A "${HF_CACHE}/datasets" 2>/dev/null)" ]; then
    echo "[INFO] No HuggingFace datasets cached yet at ${HF_CACHE}/datasets."
    echo "       Run one of: loco-mujoco-download / loco-mujoco-download-real / loco-mujoco-download-perfect"
    echo "       Or let ImitationFactory.make(...) pull on first use."
  fi
fi

if [ $# -eq 0 ]; then
  exec bash
else
  exec "$@"
fi
