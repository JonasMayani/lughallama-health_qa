#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
CHECKPOINT="${2:-best}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python src/evaluation/generate_lugha.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --output-file submissions/test_predictions.csv \
  --make-submission
