#!/usr/bin/env bash
# Monitor training and evaluation simultaneously

echo "=== TRAINING STATUS (GPU 3) ==="
tail -5 ../outputs/graph_flux_lora_diffusion/run_20epochs/training.log 2>/dev/null | grep -E "Epoch" | tail -1

echo ""
echo "=== EVALUATION STATUS (GPU 2) ==="
tail -10 eval_test.log 2>/dev/null | tail -3

echo ""
echo "=== GPU USAGE ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "GPU %s: %s%% util, %s/%s MB\n", $1, $2, $3, $4}'

echo ""
echo "=== PROCESSES ==="
ps aux | grep -E "python.*(train_flux|evaluate_graph)" | grep -v grep | wc -l | xargs echo "Active processes:"
