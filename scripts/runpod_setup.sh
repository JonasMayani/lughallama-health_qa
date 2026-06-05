#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

mkdir -p data/raw data/cleaned data/augmented models/checkpoints submissions reports logs

echo "Setup complete. Activate with: source .venv/bin/activate"
