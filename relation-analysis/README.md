# Stage A – Relation Analysis Dataset

Infra for Stage A of `plan_crit.md`: take Visual Genome triples, map predicates into a 24-class (geo/semantic) set, emit prompts + concept lists for Flux/SD3 ConceptAttention runs.

## Relation classes (24 total)

- Geometric (9): `left of`, `right of`, `above`, `below`, `in front of`, `behind`, `in`, `on`, `around/near`.
- Semantic (15): `holding`, `carrying`, `wearing`, `riding`, `looking at`, `touching`, `using`, `pushing`, `pulling`, `sitting on`, `standing on`, `hanging from`, `eating`, `drinking`, `playing with`.

These are VRD-style, readable by VG. Overrideable later if we want a different set.

## Layout

- `relation_analysis/` python pkg
  - `data/relations.py`: canonical predicate map (24 classes) + helpers to classify geometric vs semantic.
  - `schema.py`: dataclasses for triples/examples.
  - `vg_loader.py`: normalizes Visual Genome `relationships.json` into our triples.
  - `prompt_builder.py`: templated prompts + concept token lists.
- `scripts/build_stage_a.py`: CLI to ingest VG, filter by relation type, and write JSONL.
- `outputs/stage_a/`: default drop location for generated JSONL and stats.

## Quick start (demo without VG)

```bash
cd relation-analysis
python scripts/build_stage_a.py --demo --output outputs/stage_a/demo.jsonl
cat outputs/stage_a/demo.jsonl
```

## With Visual Genome (relationships.json)

```bash
cd relation-analysis
python scripts/build_stage_a.py \
  --relationships /path/to/relationships.json \
  --output outputs/stage_a/vg_stage_a.jsonl \
  --relation-type all \
  --max-samples 5000
```

Flags:
- `--relation-type {all,geometric,semantic}` filters classes.
- `--max-samples` caps processed examples for quick runs.
- `--object-whitelist path.txt` keeps only subject/object names in that list (optional).

Output JSONL schema (one per example):

```jsonc
{
  "triple": {
    "subject": "dog",
    "predicate": "on",
    "object": "bucket",
    "relation_type": "geometric",
    "source_image_id": 12345,
    "source_relationship_id": 67890
  },
  "prompt": "a photo of a dog on a bucket on a plain background",
  "concepts": ["dog", "on", "bucket"],
  "template_id": "simple_v1"
}
```

## Notes

- Designed to stay lightweight (stdlib only) so it runs near the smallest Flux/SD3 stacks.
- Predicate map lives in code for now; we can add YAML override later if the class set changes.
