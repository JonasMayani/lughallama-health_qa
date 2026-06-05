#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
BATCH_SIZE="${2:-4}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

bash scripts/train_3xa40.sh "$CONFIG"
bash scripts/evaluate_and_submit.sh "$CONFIG" best "$BATCH_SIZE"
