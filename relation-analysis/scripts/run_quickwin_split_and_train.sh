#!/usr/bin/env bash

# One-command "quick win" flow:
# 1) Make a tiny balanced VG split (train/test) from SGDiff train.h5
# 2) Train token+temb sequentially on the tiny train split
# 3) (Optional) Run full-sampling eval on the fixed test split (separate script)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs splits

TAG="${TAG:-vg_quickwin}"
SEED="${SEED:-42}"
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-50}"   # 50*16=800
TEST_PER_CLASS="${TEST_PER_CLASS:-10}"     # 10*16=160

echo "Making split: tag=${TAG} seed=${SEED} train_per_class=${TRAIN_PER_CLASS} test_per_class=${TEST_PER_CLASS}"
python -u make_vg_quickwin_split.py \
  --out-dir "${SCRIPT_DIR}/splits" \
  --tag "${TAG}" \
  --seed "${SEED}" \
  --train-per-class "${TRAIN_PER_CLASS}" \
  --test-per-class "${TEST_PER_CLASS}"

export TRAIN_EXAMPLES_JSONL="${SCRIPT_DIR}/splits/${TAG}_train.jsonl"
export VAL_EXAMPLES_JSONL="${SCRIPT_DIR}/splits/${TAG}_test.jsonl"

echo "Train examples: ${TRAIN_EXAMPLES_JSONL}"
echo "Test examples:  ${VAL_EXAMPLES_JSONL}"

./run_all_graph_lora.sh

