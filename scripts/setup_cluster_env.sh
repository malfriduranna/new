#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
SENTINEL="${VENV}/.deps_installed"

echo "[setup] repo: ${REPO_ROOT}"
echo "[setup] venv: ${VENV}"

if [ ! -d "${VENV}" ]; then
  echo "[setup] creating virtualenv"
  python3 -m venv "${VENV}"
fi

source "${VENV}/bin/activate"

echo "[setup] upgrading pip"
pip install --upgrade pip

echo "[setup] installing requirements"
pip install -r "${REPO_ROOT}/requirements.txt"

touch "${SENTINEL}"

echo "[setup] done. Activate with: source ${VENV}/bin/activate"
