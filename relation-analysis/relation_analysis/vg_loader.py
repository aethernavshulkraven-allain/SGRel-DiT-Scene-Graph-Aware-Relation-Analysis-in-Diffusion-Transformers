import json
from typing import Iterable, List, Optional, Set

from .data.relations import PredicateMap, normalize
from .schema import RelationTriple


def load_vg_relationships(
    path: str,
    predicate_map: PredicateMap,
    relation_type: str = "all",
    object_whitelist: Optional[Set[str]] = None,
    max_samples: Optional[int] = None,
) -> List[RelationTriple]:
    """Load relationships.json → RelationTriple list."""
    with open(path, "r") as f:
        data = json.load(f)

    triples: List[RelationTriple] = []
    for entry in data:
        image_id = entry.get("image_id") or entry.get("id")
        for rel in entry.get("relationships", []):
            spec = predicate_map.canonicalize(rel.get("predicate", ""))
            if not spec:
                continue
            if relation_type != "all" and spec.kind != relation_type:
                continue

            subj_name = _extract_name(rel.get("subject", {}))
            obj_name = _extract_name(rel.get("object", {}))
            if not subj_name or not obj_name:
                continue
            if object_whitelist:
                if normalize(subj_name) not in object_whitelist or normalize(obj_name) not in object_whitelist:
                    continue

            triples.append(
                RelationTriple(
                    subject=subj_name,
                    predicate=spec.name,
                    object=obj_name,
                    relation_type=spec.kind,
                    source_image_id=image_id,
                    source_relationship_id=rel.get("relationship_id") or rel.get("id"),
                )
            )
            if max_samples and len(triples) >= max_samples:
                return triples
    return triples


def _extract_name(node: dict) -> Optional[str]:
    names = node.get("names") or []
    if isinstance(names, str):
        names = [names]
    if names:
        return normalize(names[0])
    name = node.get("name")
    if name:
        return normalize(name)
    return None
