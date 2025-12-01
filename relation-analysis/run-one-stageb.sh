python scripts/run_stage_b.py \
  --input outputs/stage_a/sample_10.jsonl \
  --output-dir outputs/stage_b/runs/sample10 \
  --max-examples 1 \
  --steps 4 \
  --device cuda \
  --dtype bfloat16 \
  --height 512 --width 512 \
  --cpu-offload