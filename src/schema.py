from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RelationTriple:
    subject: str
    predicate: str  # canonical predicate label
    object: str
    relation_type: str  # "geometric" or "semantic"
    source_image_id: Optional[int] = None
    source_relationship_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StageAExample:
    triple: RelationTriple
    prompt: str
    concepts: List[str]
    template_id: str

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["triple"] = self.triple.to_dict()
        return data
