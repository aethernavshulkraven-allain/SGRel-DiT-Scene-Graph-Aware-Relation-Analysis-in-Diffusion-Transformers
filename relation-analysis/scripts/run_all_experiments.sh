#!/usr/bin/env bash
set -euo pipefail

# Run missing/complete experiments (WRN+tiny_cnn+mlp) for early, middle, and late layer datasets.
# Early: only tiny_cnn/mlp missing modes (cross/triple/diff) are run; WRN + other tiny/mlp already exist.
# Middle: full suite (wrn, tiny_cnn, mlp across modes) is run.
# Late: full suite (wrn, tiny_cnn, mlp across modes) is run.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
mkdir -p runs

PARALLEL="${PARALLEL:-0}"  # set to 1 to run jobs concurrently

# Early layers: run only missing tiny_cnn/mlp modes (cross, triple, diff)
echo "Starting early-layer missing tiny_cnn/mlp (PARALLEL=$PARALLEL)..."
if [[ "$PARALLEL" == "1" ]]; then
nohup python -u - <<'PY' \
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
else
python -u - <<'PY' \
  > runs/missing_early.log 2>&1
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
EARLY_PID=""
fi

echo "Starting middle-layer full sweep (PARALLEL=$PARALLEL)..."
if [[ "$PARALLEL" == "1" ]]; then
nohup python -u - <<'PY' \
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
else
python -u - <<'PY' \
  > runs/middle_full.log 2>&1
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
MIDDLE_PID=""
fi

echo "Starting late-layer full sweep (PARALLEL=$PARALLEL)..."
if [[ "$PARALLEL" == "1" ]]; then
nohup python -u - <<'PY' \
  > runs/late_full.log 2>&1 &
import json
import run_classifier_experiments_late as exp

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
json.dump(results, open('runs/experiment_results_late_full.json','w'), indent=2)
PY
LATE_PID=$!
else
python -u - <<'PY' \
  > runs/late_full.log 2>&1
import json
import run_classifier_experiments_late as exp

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
json.dump(results, open('runs/experiment_results_late_full.json','w'), indent=2)
PY
LATE_PID=""
fi

if [[ "$PARALLEL" == "1" ]]; then
  echo "Started early PID=$EARLY_PID, middle PID=$MIDDLE_PID, late PID=$LATE_PID"
else
  echo "Completed early/middle/late sweeps."
fi
echo "Tail logs with: tail -f runs/missing_early.log runs/middle_full.log runs/late_full.log"
