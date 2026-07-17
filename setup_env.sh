#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${NERO_ENV_NAME:-nero}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

echo "Setting up conda environment: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.yml" --prune
else
  conda env create -f "${ROOT_DIR}/environment.yml"
fi

echo "Installing nero_collection in editable mode"
conda run -n "${ENV_NAME}" python -m pip install -e "${ROOT_DIR}"

echo "Installing AgileX pyAgxArm SDK"
conda run -n "${ENV_NAME}" python -m pip install "python-can>=3.3.4"
if ! conda run -n "${ENV_NAME}" python -c "import pyAgxArm" >/dev/null 2>&1; then
  conda run -n "${ENV_NAME}" python -m pip install "git+https://github.com/agilexrobotics/pyAgxArm.git"
fi

echo
echo "Environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
echo
echo "Optional system tools for hardware debugging:"
if ! command -v candump >/dev/null 2>&1 || ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "  sudo apt-get install -y can-utils v4l-utils"
else
  echo "  can-utils and v4l-utils look available."
fi
