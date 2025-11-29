"""Canonical predicate map for Stage A (24 classes)."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PredicateSpec:
    name: str  # canonical predicate label
    kind: str  # "geometric" or "semantic"
    synonyms: List[str]


class PredicateMap:
    def __init__(self, specs: Iterable[PredicateSpec]):
        self.by_name: Dict[str, PredicateSpec] = {}
        self.synonym_to_name: Dict[str, str] = {}
        for spec in specs:
            key = normalize(spec.name)
            self.by_name[key] = spec
            for syn in spec.synonyms:
                self.synonym_to_name[normalize(syn)] = spec.name

    def canonicalize(self, raw: str) -> Optional[PredicateSpec]:
        """Return canonical predicate spec if we know it."""
        key = normalize(raw)
        if key in self.by_name:
            return self.by_name[key]
        name = self.synonym_to_name.get(key)
        if name:
            return self.by_name.get(normalize(name))
        return None

    def kind_for(self, raw: str) -> Optional[str]:
        spec = self.canonicalize(raw)
        return spec.kind if spec else None


def normalize(text: str) -> str:
    return text.strip().lower()


# 24-class set (9 geometric, 15 semantic)
_DEFAULT_PREDICATES: List[PredicateSpec] = [
    # Geometric
    PredicateSpec("left of", "geometric", ["to the left of", "left"]),
    PredicateSpec("right of", "geometric", ["to the right of", "right"]),
    PredicateSpec("above", "geometric", ["over", "on top of", "higher than"]),
    PredicateSpec("below", "geometric", ["under", "beneath", "lower than"]),
    PredicateSpec("in front of", "geometric", ["before", "ahead of"]),
    PredicateSpec("behind", "geometric", ["in back of", "at the back of"]),
    PredicateSpec("in", "geometric", ["inside", "within"]),
    PredicateSpec("on", "geometric", ["upon"]),
    PredicateSpec("around/near", "geometric", ["around", "near", "beside", "next to", "close to"]),
    # Semantic / HOI-style
    PredicateSpec("holding", "semantic", ["hold", "grasping"]),
    PredicateSpec("carrying", "semantic", ["carry"]),
    PredicateSpec("wearing", "semantic", ["wears", "has on"]),
    PredicateSpec("riding", "semantic", ["ride"]),
    PredicateSpec("looking at", "semantic", ["look at", "watching", "watch"]),
    PredicateSpec("touching", "semantic", ["touch"]),
    PredicateSpec("using", "semantic", ["use"]),
    PredicateSpec("pushing", "semantic", ["push"]),
    PredicateSpec("pulling", "semantic", ["pull"]),
    PredicateSpec("sitting on", "semantic", ["sit on", "sat on"]),
    PredicateSpec("standing on", "semantic", ["stand on"]),
    PredicateSpec("hanging from", "semantic", ["hang from", "attached to"]),
    PredicateSpec("eating", "semantic", ["eat"]),
    PredicateSpec("drinking", "semantic", ["drink"]),
    PredicateSpec("playing with", "semantic", ["play with"]),
]


def default_predicate_map() -> PredicateMap:
    return PredicateMap(_DEFAULT_PREDICATES)


DEFAULT_GEOMETRIC = {spec.name for spec in _DEFAULT_PREDICATES if spec.kind == "geometric"}
DEFAULT_SEMANTIC = {spec.name for spec in _DEFAULT_PREDICATES if spec.kind == "semantic"}
