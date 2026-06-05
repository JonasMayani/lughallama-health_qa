#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python src/training/train_lugha.py --config "$CONFIG"
