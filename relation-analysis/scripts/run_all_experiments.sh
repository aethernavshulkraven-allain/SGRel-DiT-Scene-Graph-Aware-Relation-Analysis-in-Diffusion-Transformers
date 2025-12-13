#!/usr/bin/env bash
set -euo pipefail

# Run missing experiments (WRN+tiny_cnn+mlp) for early and middle layer datasets.
# Early: only tiny_cnn/mlp missing modes (cross/triple/diff) are run; WRN + other tiny/mlp already exist.
# Middle: full suite (wrn, tiny_cnn, mlp across modes) is run.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p runs

# Early layers: run only missing tiny_cnn/mlp modes (cross, triple, diff)
echo "Launching early-layer missing tiny_cnn/mlp..."
nohup python - <<'PY' \
  > runs/missing_early.log 2>&1 &
import json
import run_classifier_experiments as exp

missing_configs = [
    {'name':'cross_tinycnn','mode':'cross','arch':'tiny_cnn'},
    {'name':'triple_tinycnn','mode':'triple','arch':'tiny_cnn'},
    {'name':'diff_tinycnn','mode':'diff','arch':'tiny_cnn'},
    {'name':'cross_mlp','mode':'cross','arch':'mlp'},
    {'name':'triple_mlp','mode':'triple','arch':'mlp'},
    {'name':'diff_mlp','mode':'diff','arch':'mlp'},
]

results = []
for cfg in missing_configs:
    print("Running", cfg)
    res = exp.run_experiment({**cfg,
                              'epochs':100,
                              'batch_size':128,
                              'widen_factor':8,
                              'dropout':0.3,
                              'norm':'none',
                              'smooth':0,
                              'balanced':False,
                              'hflip':False,
                              'zero_subject':False,
                              'zero_predicate':False,
                              'zero_object':False,
                              'shuffle':False,
                              'save_ckpt':True})
    results.append(res)
json.dump(results, open('runs/experiment_results_missing_early.json','w'), indent=2)
PY
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

echo "Launching middle-layer full sweep..."
nohup python - <<'PY' \
  > runs/middle_full.log 2>&1 &
import json
import run_classifier_experiments_middle as exp

base_modes = ['saliency','cross','concat','triple','diff']
configs = (
    [{'name':m,'mode':m,'arch':'wrn'} for m in base_modes] +
    [{'name':'late','mode':'late','arch':'wrn'}] +
    [{'name':f"{m}_tinycnn",'mode':m,'arch':'tiny_cnn'} for m in base_modes] +
    [{'name':f"{m}_mlp",'mode':m,'arch':'mlp'} for m in base_modes]
)

results = []
for cfg in configs:
    print("Running", cfg)
    res = exp.run_experiment({**cfg,
                              'epochs':100,
                              'batch_size':128,
                              'widen_factor':8,
                              'dropout':0.3,
                              'norm':'none',
                              'smooth':0,
                              'balanced':False,
                              'hflip':False,
                              'zero_subject':False,
                              'zero_predicate':False,
                              'zero_object':False,
                              'shuffle':False,
                              'save_ckpt':True})
    results.append(res)
json.dump(results, open('runs/experiment_results_middle_full.json','w'), indent=2)
PY
MIDDLE_PID=$!

echo "Started early PID=$EARLY_PID, middle PID=$MIDDLE_PID"
echo "Tail logs with: tail -f runs/missing_early.log runs/middle_full.log"
