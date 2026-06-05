#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
CHECKPOINT="${2:-best}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python src/evaluation/generate_lugha.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split val \
  --output-file submissions/val_predictions.csv

python src/evaluation/evaluate_lugha.py \
  --config "$CONFIG" \
  --predictions submissions/val_predictions.csv \
  --output-json reports/val_metrics.json \
  --output-scored-csv reports/val_scored_predictions.csv
