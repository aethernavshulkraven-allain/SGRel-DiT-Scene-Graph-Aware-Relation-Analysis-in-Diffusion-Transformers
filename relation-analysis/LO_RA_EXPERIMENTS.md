# Graph-Conditioned FLUX LoRA Experiments (Teacher-Forced Train, Full-Sampling Eval)

This repo implements **graph-conditioned LoRA adaptation** for `black-forest-labs/FLUX.1-schnell` following `plan_lora.md`:

- **Training** uses **teacher-forced single-timestep** rectified-flow updates (no full sampling).
- **Evaluation** uses **full sampling** and reports graph-sensitivity metrics.

Two conditioning modes are supported:
- `token`: graph information is injected as extra conditioning tokens (active only in blocks 7–12).
- `temb`: graph information is injected into timestep embedding (`temb`) (active only in blocks 7–12).

> Note: the legacy SGDiff VG graph encoder vocab only supports **16/24** canonical predicates; scripts automatically filter to the supported subset (see `plan_lora.md`).

---

## Scripts Involved

Training (teacher-forced):
- `relation-analysis/scripts/train_flux_graph_lora_diffusion.py`  
  Main trainer. LoRA on FLUX double blocks 7–12 attention projections + trainable graph projections; frozen base weights + frozen WRN classifier.

Launchers:
- `relation-analysis/scripts/run_all_graph_lora.sh`  
  Runs **token then temb** training sequentially via `nohup`, logs to `relation-analysis/scripts/runs/`.
- `relation-analysis/scripts/run_quickwin_split_and_train.sh`  
  Creates a tiny VG subset split and then calls `run_all_graph_lora.sh` using that split.

Quick-win split generator:
- `relation-analysis/scripts/make_vg_quickwin_split.py`  
  Builds balanced train/test JSONL subsets from SGDiff VG (`train.h5` + images).

Evaluation (full sampling):
- `relation-analysis/scripts/eval_graph_lora_full_sampling.py`  
  Full sampling evaluation + metrics + optional image panels.
- `relation-analysis/scripts/run_graph_lora_eval_full_sampling.sh`  
  Wrapper that runs `eval_graph_lora_full_sampling.py` for token and temb checkpoints via `nohup`.

---

## Environment Setup

From the workspace root:

```bash
conda activate fluxb
cd "SGRel-DiT: Scene Graph–Aware Relation Analysis in Diffusion Transformers/relation-analysis/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Optional (recommended for nohup logging):
```bash
export PYTHONUNBUFFERED=1
```

---

## Quick-Win Run (Tiny VG Subset)

This is the recommended path for fast iteration and “overnight-feasible” experiments.

### 1) Launch training (token → temb) on a small fixed subset

```bash
export DEVICE=cuda DTYPE=bfloat16 CPU_OFFLOAD=0
export HEIGHT=256 WIDTH=256 BATCH_SIZE=1
export T_MIN=0.5 T_MAX=1.0
export LORA_RANK=16 LORA_ALPHA=16
export LAMBDA_REL_RANK=0.3 REL_RANK_MARGIN=1.0 ALPHA_GEN_REL=0.0
export HARD_NEGATIVE_TOPK=2

# Big speedups:
export LATENT_CACHE_DIR="../outputs/latent_cache_256"
export MAX_SEQUENCE_LENGTH=128

# 16 supported predicates → totals: train=16*TRAIN_PER_CLASS, test=16*TEST_PER_CLASS
export TRAIN_PER_CLASS=50   # 800 train examples
export TEST_PER_CLASS=10    # 160 test examples (fixed reporting set)

./run_quickwin_split_and_train.sh
```

Artifacts:
- Splits are written to `relation-analysis/scripts/splits/`:
  - `vg_quickwin_train.jsonl` (gradients)
  - `vg_quickwin_test.jsonl` (fixed test/reporting; never used for gradients)
  - `vg_quickwin_summary.json`
- Logs:
  - `relation-analysis/scripts/runs/graph_lora_token.log`
  - `relation-analysis/scripts/runs/graph_lora_temb.log`
- Checkpoints:
  - `relation-analysis/outputs/graph_flux_lora_diffusion/latest/`

### 2) Monitor

```bash
tail -f runs/graph_lora_token.log
# later
tail -f runs/graph_lora_temb.log
```

---

## Full-Sampling Evaluation (Fixed Test Set)

Evaluate checkpoints using full sampling and report:
- accuracy on `g+`
- mean margin `E[logit(g+)_y - logit(g-)_y]`
- win rate `P(logit(g+)_y > logit(g-)_y)`

### Launch eval (token and temb)

```bash
export DEVICE=cuda DTYPE=bfloat16 CPU_OFFLOAD=0
export HEIGHT=512 WIDTH=512
export STEPS=4 GUIDANCE=0.0
export SEEDS=0,1,2
export MAX_EXAMPLES=160

# Use the quick-win fixed test split as eval input:
export VAL_EXAMPLES_JSONL="relation-analysis/scripts/splits/vg_quickwin_test.jsonl"

./run_graph_lora_eval_full_sampling.sh "relation-analysis/outputs/graph_flux_lora_diffusion/latest"
```

Outputs:
- `relation-analysis/outputs/graph_lora_eval_full_sampling_<timestamp>/token/summary.json`
- `relation-analysis/outputs/graph_lora_eval_full_sampling_<timestamp>/temb/summary.json`
- Optional images/panels under the same folders.

---

## Full Dataset Run (Not Recommended for Iteration)

If you really want to train on the full SGDiff VG set, run:

```bash
unset TRAIN_EXAMPLES_JSONL
unset VAL_EXAMPLES_JSONL
export MAX_TRAIN_SAMPLES=-1
export MAX_VAL_SAMPLES=100
./run_all_graph_lora.sh
```

This is significantly slower (and previously caused stalls if CPU prompt encoding dominates).

---

## Stop / Cleanup

Terminate training/eval:
```bash
pkill -f "train_flux_graph_lora_diffusion.py" || true
pkill -f "eval_graph_lora_full_sampling.py" || true
pkill -f "run_all_graph_lora.sh" || true
pkill -f "run_quickwin_split_and_train.sh" || true
```

---

## Key Knobs (Environment Variables)

Training launcher knobs (`run_all_graph_lora.sh`):
- `DEVICE` (`cuda`), `DTYPE` (`bfloat16|float16|float32`)
- `HEIGHT`, `WIDTH`, `BATCH_SIZE`, `LR`, `NUM_WORKERS`
- `BLOCK_START` (default 7), `BLOCK_END` (default 13)
- `LORA_RANK`, `LORA_ALPHA`
- `T_MIN`, `T_MAX`
- `USE_NEGATIVE_GRAPH` (`1|0`), `HARD_NEGATIVE_TOPK`
- `LAMBDA_REL_RANK`, `REL_RANK_MARGIN`, `ALPHA_GEN_REL`
- `LATENT_CACHE_DIR` (recommended)
- `MAX_SEQUENCE_LENGTH` (recommended: 128 for speed)
- `TRAIN_EXAMPLES_JSONL`, `VAL_EXAMPLES_JSONL` (fixed subset training/eval)

