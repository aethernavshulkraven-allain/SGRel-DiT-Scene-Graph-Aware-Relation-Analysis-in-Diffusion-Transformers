#!/usr/bin/env bash

# Full-sampling evaluation for graph-conditioned LoRA checkpoints (plan_lora.md §7).
# This keeps full sampling ONLY for evaluation; training is teacher-forced.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p runs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CKPT_DIR="${1:-${PROJECT_ROOT}/outputs/graph_flux_lora_diffusion/latest}"
if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "Checkpoint dir not found: ${CKPT_DIR}" >&2
  exit 1
fi

INPUT="${INPUT:-${PROJECT_ROOT}/outputs/stage_a/vg_stage_a.jsonl}"
MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.1-schnell}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
MAX_EXAMPLES="${MAX_EXAMPLES:-8}"
STEPS="${STEPS:-4}"
GUIDANCE="${GUIDANCE:-0.0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEEDS="${SEEDS:-0,1,2}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
SAVE_IMAGES="${SAVE_IMAGES:-1}"
MAKE_PANELS="${MAKE_PANELS:-1}"
HARD_NEGATIVE_TOPK="${HARD_NEGATIVE_TOPK:-2}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-}"
CLASSIFIER_IN_SIZE="${CLASSIFIER_IN_SIZE:-32}"

BLOCK_START="${BLOCK_START:-7}"
BLOCK_END="${BLOCK_END:-13}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="${PROJECT_ROOT}/outputs/graph_lora_eval_full_sampling_${RUN_ID}"
mkdir -p "${OUT_BASE}"

OFFLOAD_FLAG=()
if [[ "${CPU_OFFLOAD}" -eq 1 ]]; then
  OFFLOAD_FLAG+=(--cpu-offload)
fi

TOKEN_CKPT="${CKPT_DIR}/best_graph_lora_token.pt"
TEMB_CKPT="${CKPT_DIR}/best_graph_lora_temb.pt"
if [[ ! -f "${TOKEN_CKPT}" ]]; then
  TOKEN_CKPT="${CKPT_DIR}/final_graph_lora_token.pt"
fi
if [[ ! -f "${TEMB_CKPT}" ]]; then
  TEMB_CKPT="${CKPT_DIR}/final_graph_lora_temb.pt"
fi

echo "Stage A JSONL: ${INPUT}"
echo "Checkpoint dir: ${CKPT_DIR}"
echo "Eval output dir: ${OUT_BASE}"

if [[ -f "${TOKEN_CKPT}" ]]; then
  echo "Launching full-sampling eval (token) using checkpoint: ${TOKEN_CKPT}"
  CMD=(python eval_graph_lora_full_sampling.py
    --input "${INPUT}"
    --output-dir "${OUT_BASE}/token"
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-examples "${MAX_EXAMPLES}" \
    --steps "${STEPS}" \
    --guidance "${GUIDANCE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --seeds "${SEEDS}" \
    --lora-checkpoint "${TOKEN_CKPT}"
    --graph-mode token \
    --block-start "${BLOCK_START}" \
    --block-end "${BLOCK_END}" \
    --hard-negative-topk "${HARD_NEGATIVE_TOPK}" \
    --classifier-in-size "${CLASSIFIER_IN_SIZE}"
  )
  if [[ "${SAVE_IMAGES}" -eq 1 ]]; then
    CMD+=(--save-images)
  else
    CMD+=(--no-save-images)
  fi
  if [[ "${MAKE_PANELS}" -eq 1 ]]; then
    CMD+=(--make-panels)
  else
    CMD+=(--no-make-panels)
  fi
  if [[ -n "${CLASSIFIER_CKPT}" ]]; then
    CMD+=(--classifier-ckpt "${CLASSIFIER_CKPT}")
  fi
  CMD+=("${OFFLOAD_FLAG[@]}")
  nohup "${CMD[@]}" > "runs/graph_lora_eval_token_${RUN_ID}.log" 2>&1 &
  PID_TOKEN=$!
  echo "Started token eval PID=${PID_TOKEN}"
  wait "${PID_TOKEN}"
else
  echo "Token checkpoint not found under: ${CKPT_DIR}" >&2
fi

if [[ -f "${TEMB_CKPT}" ]]; then
  echo "Launching full-sampling eval (temb) using checkpoint: ${TEMB_CKPT}"
  CMD=(python eval_graph_lora_full_sampling.py
    --input "${INPUT}"
    --output-dir "${OUT_BASE}/temb"
    --model-id "${MODEL_ID}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --max-examples "${MAX_EXAMPLES}" \
    --steps "${STEPS}" \
    --guidance "${GUIDANCE}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --seeds "${SEEDS}" \
    --lora-checkpoint "${TEMB_CKPT}"
    --graph-mode temb \
    --block-start "${BLOCK_START}" \
    --block-end "${BLOCK_END}" \
    --hard-negative-topk "${HARD_NEGATIVE_TOPK}" \
    --classifier-in-size "${CLASSIFIER_IN_SIZE}"
  )
  if [[ "${SAVE_IMAGES}" -eq 1 ]]; then
    CMD+=(--save-images)
  else
    CMD+=(--no-save-images)
  fi
  if [[ "${MAKE_PANELS}" -eq 1 ]]; then
    CMD+=(--make-panels)
  else
    CMD+=(--no-make-panels)
  fi
  if [[ -n "${CLASSIFIER_CKPT}" ]]; then
    CMD+=(--classifier-ckpt "${CLASSIFIER_CKPT}")
  fi
  CMD+=("${OFFLOAD_FLAG[@]}")
  nohup "${CMD[@]}" > "runs/graph_lora_eval_temb_${RUN_ID}.log" 2>&1 &
  PID_TEMB=$!
  echo "Started temb eval PID=${PID_TEMB}"
  wait "${PID_TEMB}"
else
  echo "Temb checkpoint not found under: ${CKPT_DIR}" >&2
fi

echo "Done. Logs:"
echo "  runs/graph_lora_eval_token_${RUN_ID}.log"
echo "  runs/graph_lora_eval_temb_${RUN_ID}.log"
