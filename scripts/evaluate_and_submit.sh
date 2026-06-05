#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
CHECKPOINT="${2:-best}"
BATCH_SIZE="${3:-4}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python src/evaluation/eval_lugha.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --batch-size "$BATCH_SIZE" \
  --make-submission
