from typing import Dict, Tuple

from .schema import RelationTriple

# Simple templates; extend if we want more stylistic diversity.
TEMPLATES: Dict[str, str] = {
    "simple_v1": "a photo of {subj} {pred_phrase} {obj} on a plain background",
    "instructional_v1": "an image depicting {subj} {pred_phrase} {obj}",
}


def predicate_to_phrase(predicate: str) -> str:
    """Turn canonical label into a readable phrase."""
    replacements = {
        "around/near": "near",
    }
    return replacements.get(predicate, predicate)


def build_prompt(triple: RelationTriple, template_id: str = "simple_v1") -> Tuple[str, str]:
    if template_id not in TEMPLATES:
        raise ValueError(f"Unknown template_id {template_id}")
    phrase = predicate_to_phrase(triple.predicate)
    prompt = TEMPLATES[template_id].format(subj=triple.subject, pred_phrase=phrase, obj=triple.object)
    return prompt, template_id


def concepts_for(triple: RelationTriple):
    return [triple.subject, predicate_to_phrase(triple.predicate), triple.object]
