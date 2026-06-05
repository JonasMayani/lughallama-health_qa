#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

bash scripts/train.sh "$CONFIG"
bash scripts/evaluate_and_submit.sh "$CONFIG" best
