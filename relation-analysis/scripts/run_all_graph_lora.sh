#!/usr/bin/env bash

# Launch both graph-conditioning modes (token concat and temb add) with LoRA on middle blocks.
# Logs are written to runs/graph_lora_token.log and runs/graph_lora_temb.log.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p runs

# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Launching graph-conditioned Flux + LoRA (token concat) sequentially..."
nohup python train_flux_graph_lora.py \
  --graph-mode token \
  --epochs 8 \
  --batch-size 64 \
  --height 256 \
  --width 256 \
  --device cuda:3 \
  --no-cpu-offload \
  > runs/graph_lora_token.log 2>&1 &
PID_TOKEN=$!
echo "Started token PID=${PID_TOKEN}"
wait ${PID_TOKEN}
echo "Token run finished. Starting temb run..."

nohup python train_flux_graph_lora.py \
  --graph-mode temb \
  --epochs 1 \
  --batch-size 1 \
  --height 256 \
  --width 256 \
  --device cuda:3 \
  > runs/graph_lora_temb.log 2>&1 &
PID_TEMB=$!

echo "Started temb PID=${PID_TEMB}"
echo "Tail logs with: tail -f runs/graph_lora_token.log runs/graph_lora_temb.log"
