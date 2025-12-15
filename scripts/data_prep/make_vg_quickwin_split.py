#!/usr/bin/env python3
"""Create a tiny, high-signal Visual Genome subset for quick-win LoRA runs.

Outputs two JSONL files:
- train: small balanced subset for gradients
- test: fixed 100–300 examples for reporting (never used for gradients)

Each line matches the Stage-A item schema used by eval scripts, but includes image path
metadata needed for teacher-forced training:
{
  "prompt": str,
  "concepts": [subj, pred_phrase, obj],
  "triple": {"subject": str, "predicate": <canonical>, "object": str},
  "meta": {
    "image_rel_path": str,
    "predicate_raw": str,
    "class_id": int,
    "h5_split": "train",
    "img_idx": int,
    "rel_idx": int
  }
}
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relation_analysis.data.relations import _DEFAULT_PREDICATES, default_predicate_map
from relation_analysis.prompt_builder import predicate_to_phrase


CLASS_NAMES: List[str] = sorted([spec.name for spec in _DEFAULT_PREDICATES])
CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _read_vocab(vocab_path: Path) -> dict:
    return json.loads(vocab_path.read_text())


def _iter_candidate_examples(
    *,
    vocab: dict,
    h5_path: Path,
    images_dir: Path,
    prompt_template: str,
) -> List[dict]:
    predicate_map = default_predicate_map()

    pred_id_to_canonical: Dict[int, str] = {}
    pred_id_to_class: Dict[int, int] = {}
    for pred_id, pred_name in enumerate(vocab.get("pred_idx_to_name", [])):
        spec = predicate_map.canonicalize(pred_name)
        if spec is None:
            continue
        class_id = CLASS_NAME_TO_ID.get(spec.name)
        if class_id is None:
            continue
        pred_id_to_canonical[int(pred_id)] = spec.name
        pred_id_to_class[int(pred_id)] = int(class_id)

    candidates: List[dict] = []
    with h5py.File(str(h5_path), "r") as h5:
        rels_per_image = h5["relationships_per_image"][()]
        rel_subj = h5["relationship_subjects"][()]
        rel_obj = h5["relationship_objects"][()]
        rel_pred = h5["relationship_predicates"][()]
        obj_names = h5["object_names"][()]
        raw_paths = h5["image_paths"][()]

    image_paths = [p.decode("utf-8") if isinstance(p, (bytes, bytearray)) else str(p) for p in raw_paths]
    object_names = vocab["object_idx_to_name"]
    pred_names = vocab["pred_idx_to_name"]

    for img_idx in range(len(image_paths)):
        count = int(rels_per_image[img_idx])
        if count <= 0:
            continue
        image_rel_path = image_paths[img_idx]
        img_path = images_dir / image_rel_path
        if not img_path.exists():
            continue

        for rel_idx in range(count):
            p_id = int(rel_pred[img_idx, rel_idx])
            canonical = pred_id_to_canonical.get(p_id)
            class_id = pred_id_to_class.get(p_id)
            if canonical is None or class_id is None:
                continue

            s_idx = int(rel_subj[img_idx, rel_idx])
            o_idx = int(rel_obj[img_idx, rel_idx])
            subj_obj_idx = int(obj_names[img_idx, s_idx])
            obj_obj_idx = int(obj_names[img_idx, o_idx])
            subject = object_names[subj_obj_idx]
            obj = object_names[obj_obj_idx]
            predicate_raw = pred_names[p_id]

            pred_phrase = predicate_to_phrase(canonical)
            prompt = prompt_template.format(subject=subject, predicate=pred_phrase, object=obj)

            candidates.append(
                {
                    "prompt": prompt,
                    "concepts": [subject, pred_phrase, obj],
                    "triple": {"subject": subject, "predicate": canonical, "object": obj},
                    "meta": {
                        "image_rel_path": image_rel_path,
                        "predicate_raw": predicate_raw,
                        "class_id": int(class_id),
                        "h5_split": "train",
                        "img_idx": int(img_idx),
                        "rel_idx": int(rel_idx),
                    },
                }
            )

    return candidates


def _write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    p = argparse.ArgumentParser(description="Make a small VG quick-win split (train/test JSONL).")
    p.add_argument(
        "--vocab-path",
        default=str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "vocab.json"),
        help="Path to SGDiff VG vocab.json",
    )
    p.add_argument(
        "--train-h5",
        default=str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "train.h5"),
        help="Path to SGDiff VG train.h5",
    )
    p.add_argument(
        "--images-dir",
        default=str(REPO_ROOT / "SGDiff" / "datasets" / "vg" / "images"),
        help="Path to VG images dir (SGDiff layout)",
    )
    p.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "scripts" / "splits"),
        help="Output directory for JSONL split files",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--train-per-class",
        type=int,
        default=50,
        help="Examples per supported canonical class for training (balanced).",
    )
    p.add_argument(
        "--test-per-class",
        type=int,
        default=10,
        help="Examples per supported canonical class for the fixed test set (balanced).",
    )
    p.add_argument(
        "--prompt-template",
        default="a photo of {subject} {predicate} {object}",
        help="Prompt template (must contain {subject},{predicate},{object}).",
    )
    p.add_argument(
        "--tag",
        default="vg_quickwin",
        help="Prefix name for split files (train/test).",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    vocab_path = Path(args.vocab_path)
    h5_path = Path(args.train_h5)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    vocab = _read_vocab(vocab_path)
    candidates = _iter_candidate_examples(
        vocab=vocab,
        h5_path=h5_path,
        images_dir=images_dir,
        prompt_template=args.prompt_template,
    )

    # Group by canonical predicate for balanced sampling across supported predicates.
    by_pred: Dict[str, List[dict]] = defaultdict(list)
    for ex in candidates:
        by_pred[ex["triple"]["predicate"]].append(ex)

    predicate_map = default_predicate_map()
    canon_to_raw_preds: Dict[str, List[str]] = {}
    for p_name in vocab.get("pred_idx_to_name", []):
        if not isinstance(p_name, str) or p_name.startswith("__"):
            continue
        spec = predicate_map.canonicalize(p_name)
        if spec is None:
            continue
        canon_to_raw_preds.setdefault(spec.name, []).append(p_name)
    supported_canonical = sorted([c for c in canon_to_raw_preds.keys() if c in CLASS_NAME_TO_ID])

    train_rows: List[dict] = []
    test_rows: List[dict] = []
    used_keys: set = set()

    def _key(ex: dict) -> Tuple[int, int]:
        meta = ex.get("meta", {})
        return (int(meta.get("img_idx", -1)), int(meta.get("rel_idx", -1)))

    for canon in supported_canonical:
        pool = list(by_pred.get(canon, []))
        rng.shuffle(pool)
        if not pool:
            continue

        # test first (fixed), then train from remaining
        take_test = min(int(args.test_per_class), len(pool))
        picked_test = []
        for ex in pool:
            k = _key(ex)
            if k in used_keys:
                continue
            picked_test.append(ex)
            used_keys.add(k)
            if len(picked_test) >= take_test:
                break

        take_train = int(args.train_per_class)
        picked_train = []
        for ex in pool:
            k = _key(ex)
            if k in used_keys:
                continue
            picked_train.append(ex)
            used_keys.add(k)
            if len(picked_train) >= take_train:
                break

        test_rows.extend(picked_test)
        train_rows.extend(picked_train)

    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    train_path = out_dir / f"{args.tag}_train.jsonl"
    test_path = out_dir / f"{args.tag}_test.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(test_path, test_rows)

    counts_train = Counter([r["triple"]["predicate"] for r in train_rows])
    counts_test = Counter([r["triple"]["predicate"] for r in test_rows])
    summary = {
        "seed": args.seed,
        "train_h5": str(h5_path),
        "images_dir": str(images_dir),
        "supported_canonical_predicates": supported_canonical,
        "train_total": len(train_rows),
        "test_total": len(test_rows),
        "train_counts_by_predicate": dict(counts_train),
        "test_counts_by_predicate": dict(counts_test),
    }
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote train split: {train_path} ({len(train_rows)} examples)")
    print(f"Wrote test split:  {test_path} ({len(test_rows)} examples)")
    print(f"Summary:           {out_dir / f'{args.tag}_summary.json'}")


if __name__ == "__main__":
    main()

