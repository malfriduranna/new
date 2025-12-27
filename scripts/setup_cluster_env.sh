#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_PATH:-${REPO_ROOT}/.venv}"
SENTINEL="${VENV}/.deps_installed"

echo "[setup] repo: ${REPO_ROOT}"
echo "[setup] venv: ${VENV}"

if [ ! -d "${VENV}" ]; then
  echo "[setup] creating virtualenv"
  python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"

echo "[setup] upgrading pip"
pip install ${PIP_UPGRADE_OPTS:---no-cache-dir} --upgrade pip

echo "[setup] installing requirements"
pip install ${PIP_INSTALL_OPTS:---no-cache-dir} -r "${REPO_ROOT}/requirements.txt"

python - <<'PY'
import importlib, pkgutil
import numpy
import sys
if pkgutil.find_loader("numpy._utils") is None:
    raise SystemExit("numpy installation incomplete (missing numpy._utils)")
print(f"[setup] verified numpy {numpy.__version__}")
PY

touch "${SENTINEL}"

echo "[setup] done. Activate with: source ${VENV}/bin/activate"
