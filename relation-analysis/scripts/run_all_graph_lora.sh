#!/usr/bin/env bash

# Train graph-conditioned LoRA using teacher-forced (single-timestep) flow-matching updates.
# - Training is cheap: NO full sampling.
# - Evaluation can be done separately with full sampling via `run_graph_lora_eval_full_sampling.sh`.
#
# Logs are written to `runs/graph_lora_token.log` and `runs/graph_lora_temb.log`.
#
# Note: SGDiff's VG vocab only supports a 16/24 subset of the canonical predicates; training automatically
# filters to that subset (see plan_lora.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BLOCK_START="${BLOCK_START:-7}"
BLOCK_END="${BLOCK_END:-13}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-200}"
SAVE_EVERY="${SAVE_EVERY:-500}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-100}"
T_MIN="${T_MIN:-0.0}"
T_MAX="${T_MAX:-1.0}"
LATENT_CACHE_DIR="${LATENT_CACHE_DIR:-}"

USE_NEGATIVE_GRAPH="${USE_NEGATIVE_GRAPH:-1}"
LAMBDA_REL_RANK="${LAMBDA_REL_RANK:-0.3}"
REL_RANK_MARGIN="${REL_RANK_MARGIN:-1.0}"
ALPHA_GEN_REL="${ALPHA_GEN_REL:-0.0}"
HARD_NEGATIVE_TOPK="${HARD_NEGATIVE_TOPK:-2}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
CLASSIFIER_IN_SIZE="${CLASSIFIER_IN_SIZE:-32}"

EPOCHS_TOKEN="${EPOCHS_TOKEN:-1}"
EPOCHS_TEMB="${EPOCHS_TEMB:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${PROJECT_ROOT}/outputs/graph_flux_lora_diffusion"
OUT_DIR="${OUT_BASE}/run_${RUN_ID}"
mkdir -p "${OUT_DIR}"
ln -sfn "${OUT_DIR}" "${OUT_BASE}/latest"

COMMON_ARGS=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --num-workers "${NUM_WORKERS}"
  --block-start "${BLOCK_START}"
  --block-end "${BLOCK_END}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --output-dir "${OUT_DIR}"
  --log-every "${LOG_EVERY}"
  --val-every "${VAL_EVERY}"
  --save-every "${SAVE_EVERY}"
  --max-train-samples "${MAX_TRAIN_SAMPLES}"
  --max-val-samples "${MAX_VAL_SAMPLES}"
  --t-min "${T_MIN}"
  --t-max "${T_MAX}"
)

if [[ "${CPU_OFFLOAD}" -eq 1 ]]; then
  COMMON_ARGS+=(--cpu-offload)
else
  COMMON_ARGS+=(--no-cpu-offload)
fi

if [[ "${USE_NEGATIVE_GRAPH}" -eq 1 ]]; then
  COMMON_ARGS+=(--use-negative-graph)
else
  COMMON_ARGS+=(--no-use-negative-graph)
fi

COMMON_ARGS+=(--lambda-rel-rank "${LAMBDA_REL_RANK}" --rel-rank-margin "${REL_RANK_MARGIN}" --alpha-gen-rel "${ALPHA_GEN_REL}" --classifier-in-size "${CLASSIFIER_IN_SIZE}")
COMMON_ARGS+=(--hard-negative-topk "${HARD_NEGATIVE_TOPK}")
if [[ -n "${LATENT_CACHE_DIR}" ]]; then
  COMMON_ARGS+=(--latent-cache-dir "${LATENT_CACHE_DIR}")
fi
if [[ -n "${CLASSIFIER_CKPT}" ]]; then
  COMMON_ARGS+=(--classifier-ckpt "${CLASSIFIER_CKPT}")
fi

echo "Output dir: ${OUT_DIR}"
echo "Launching teacher-forced graph-conditioned LoRA (token concat) sequentially..."
nohup python train_flux_graph_lora_diffusion.py \
  --graph-mode token \
  --epochs "${EPOCHS_TOKEN}" \
  "${COMMON_ARGS[@]}" \
  > runs/graph_lora_token.log 2>&1 &
PID_TOKEN=$!
echo "Started token PID=${PID_TOKEN}"
wait ${PID_TOKEN}
echo "Token run finished. Starting temb run..."

nohup python train_flux_graph_lora_diffusion.py \
  --graph-mode temb \
  --epochs "${EPOCHS_TEMB}" \
  "${COMMON_ARGS[@]}" \
  > runs/graph_lora_temb.log 2>&1 &
PID_TEMB=$!

echo "Started temb PID=${PID_TEMB}"
echo "Tail logs with: tail -f runs/graph_lora_token.log runs/graph_lora_temb.log"
echo "Latest checkpoints symlink: ${OUT_BASE}/latest"
echo "For full-sampling Stage B eval: ${SCRIPT_DIR}/run_graph_lora_eval_full_sampling.sh ${OUT_BASE}/latest"
