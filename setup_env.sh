#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ENV_NAME="${ENV_NAME:-flash3d}"

echo "============================================"
echo "  Flash3D Environment Setup"
echo "============================================"
echo ""

if command -v conda &>/dev/null; then
    echo "[1/4] Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    eval "$(conda shell.bash hook)"
    conda activate "${ENV_NAME}"
else
    echo "[1/4] Creating venv: ${ENV_NAME}"
    python3 -m venv "${ENV_NAME}"
    source "${ENV_NAME}/bin/activate"
fi

echo "[2/4] Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "[3/4] Installing Flash3D with all dependencies..."
pip install -e ".[dev,full]"

echo "[4/4] Installing pre-commit hooks..."
if command -v pre-commit &>/dev/null; then
    pre-commit install
fi

echo ""
echo "============================================"
echo "  Flash3D environment ready!"
echo "  Activate with: conda activate ${ENV_NAME}"
echo "  Or: source ${ENV_NAME}/bin/activate"
echo "============================================"
