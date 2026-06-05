#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/config_lugha.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

python src/data/clean_lugha_data.py --config "$CONFIG"
