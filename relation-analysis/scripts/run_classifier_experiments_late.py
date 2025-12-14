#!/usr/bin/env python
"""
Late-layer classifier sweep.

This intentionally reuses the exact training/eval code from `run_classifier_experiments.py`
and only swaps the dataset root + output JSON path.
"""

import json
from pathlib import Path

import run_classifier_experiments as base

ROOT = Path('..').resolve()
DATA_ROOT = ROOT.parent / 'saliency_datasets' / 'late_layers'
assert DATA_ROOT.exists(), DATA_ROOT


def run_experiment(cfg):
    old_root = base.DATA_ROOT
    base.DATA_ROOT = DATA_ROOT
    try:
        return base.run_experiment(cfg)
    finally:
        base.DATA_ROOT = old_root


def main():
    base_modes = [
        {'name': 'saliency', 'mode': 'saliency'},
        {'name': 'cross', 'mode': 'cross'},
        {'name': 'concat', 'mode': 'concat'},
        {'name': 'triple', 'mode': 'triple'},
        {'name': 'diff', 'mode': 'diff'},
    ]
    configs = [
        *[{**m, 'arch': 'wrn'} for m in base_modes],
        {'name': 'late', 'mode': 'late', 'arch': 'wrn'},
        *[{**m, 'name': f"{m['name']}_tinycnn", 'arch': 'tiny_cnn'} for m in base_modes],
        *[{**m, 'name': f"{m['name']}_mlp", 'arch': 'mlp'} for m in base_modes],
    ]

    results = []
    for cfg in configs:
        print(f"\n=== Running {cfg['name']} (late) ===")
        res = run_experiment({**cfg,
                              'epochs': 100,
                              'batch_size': 128,
                              'widen_factor': 8,
                              'dropout': 0.3,
                              'norm': 'none',
                              'smooth': 0,
                              'balanced': False,
                              'hflip': False,
                              'zero_subject': False,
                              'zero_predicate': False,
                              'zero_object': False,
                              'shuffle': False,
                              'save_ckpt': True})
        results.append(res)

    out_path = Path('runs') / 'experiment_results_late_full.json'
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print('Saved results to', out_path.resolve())


if __name__ == '__main__':
    main()

