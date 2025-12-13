#!/usr/bin/env bash
set -euo pipefail

# Run all experiments (WRN+baselines) for early and middle layer datasets.
# Logs and checkpoints go under relation-analysis/scripts/runs/.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p runs

# Early layers (default script path)
echo "Launching early-layer sweep..."
nohup python run_classifier_experiments.py \
  > runs/experiments_all_early.log 2>&1 &
EARLY_PID=$!

# Middle layers: temporary copy pointing to middle_layers
TMP_SCRIPT="run_classifier_experiments_middle.py"
cp run_classifier_experiments.py "$TMP_SCRIPT"
python - <<'PY'
from pathlib import Path
p = Path('run_classifier_experiments_middle.py')
text = p.read_text()
text = text.replace("saliency_datasets' / 'early_layers'", "saliency_datasets' / 'middle_layers'")
p.write_text(text)
PY

echo "Launching middle-layer sweep..."
nohup python "$TMP_SCRIPT" \
  > runs/experiments_all_middle.log 2>&1 &
MIDDLE_PID=$!

echo "Started early PID=$EARLY_PID, middle PID=$MIDDLE_PID"
echo "Tail logs with: tail -f runs/experiments_all_early.log runs/experiments_all_middle.log"
