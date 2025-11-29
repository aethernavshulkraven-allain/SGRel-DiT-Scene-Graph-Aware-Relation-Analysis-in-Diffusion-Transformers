#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from relation_analysis.data.relations import default_predicate_map, normalize
from relation_analysis.prompt_builder import build_prompt, concepts_for
from relation_analysis.schema import RelationTriple, StageAExample
from relation_analysis.vg_loader import load_vg_relationships


def parse_args():
    parser = argparse.ArgumentParser(description="Build Stage A relation-analysis JSONL from Visual Genome.")
    parser.add_argument("--relationships", type=str, help="Path to Visual Genome relationships.json")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path")
    parser.add_argument("--stats", type=str, help="Optional stats JSON path (default: alongside output)")
    parser.add_argument("--relation-type", choices=["all", "geometric", "semantic"], default="all")
    parser.add_argument("--max-samples", type=int, help="Cap on number of examples to emit")
    parser.add_argument("--object-whitelist", type=str, help="Optional txt file: one object label per line")
    parser.add_argument("--template-id", type=str, default="simple_v1", help="Prompt template id to use")
    parser.add_argument("--demo", action="store_true", help="Use small built-in sample instead of VG")
    return parser.parse_args()


def load_whitelist(path: Optional[str]) -> Optional[Set[str]]:
    if not path:
        return None
    with open(path, "r") as f:
        return {normalize(line) for line in f if line.strip()}


def build_examples(triples: List[RelationTriple], template_id: str) -> List[StageAExample]:
    examples: List[StageAExample] = []
    for triple in triples:
        prompt, template_id_used = build_prompt(triple, template_id=template_id)
        examples.append(StageAExample(triple=triple, prompt=prompt, concepts=concepts_for(triple), template_id=template_id_used))
    return examples


def write_jsonl(examples: List[StageAExample], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")


def write_stats(examples: List[StageAExample], stats_path: Path):
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    by_kind = Counter(ex.triple.relation_type for ex in examples)
    by_pred = Counter(ex.triple.predicate for ex in examples)
    payload = {
        "num_examples": len(examples),
        "by_kind": dict(by_kind),
        "by_predicate": dict(by_pred),
    }
    with stats_path.open("w") as f:
        json.dump(payload, f, indent=2)


def demo_triples() -> List[RelationTriple]:
    return [
        RelationTriple("dog", "on", "bucket", relation_type="geometric", source_image_id=-1, source_relationship_id=-1),
        RelationTriple("person", "holding", "umbrella", relation_type="semantic", source_image_id=-2, source_relationship_id=-2),
        RelationTriple("cat", "around/near", "sofa", relation_type="geometric", source_image_id=-3, source_relationship_id=-3),
    ]


def main():
    args = parse_args()
    predicate_map = default_predicate_map()
    whitelist = load_whitelist(args.object_whitelist)

    if args.demo:
        triples = demo_triples()
    else:
        if not args.relationships:
            raise SystemExit("Need --relationships path or use --demo")
        triples = load_vg_relationships(
            path=args.relationships,
            predicate_map=predicate_map,
            relation_type=args.relation_type,
            object_whitelist=whitelist,
            max_samples=args.max_samples,
        )

    examples = build_examples(triples, template_id=args.template_id)
    output_path = Path(args.output)
    write_jsonl(examples, output_path)

    stats_path = Path(args.stats) if args.stats else output_path.with_suffix(".stats.json")
    write_stats(examples, stats_path)
    print(f"Wrote {len(examples)} examples to {output_path}")
    print(f"Stats → {stats_path}")


if __name__ == "__main__":
    main()
