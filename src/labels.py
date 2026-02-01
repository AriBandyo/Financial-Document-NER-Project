from __future__ import annotations

from typing import Dict, List, Set


def expand_bio_labels(tags: List[str]) -> List[str]:
    """
    Ensure that for every B-XXX we also include I-XXX in the label set,
    because tokenization can create extra wordpieces that need I-XXX.
    """
    labels: Set[str] = {"O"}
    for t in tags:
        labels.add(t)
        if t.startswith("B-"):
            labels.add("I-" + t[2:])
    return sorted(labels)


def build_label_map(all_tags: List[str]) -> Dict[str, int]:
    labels = expand_bio_labels(all_tags)
    return {lab: i for i, lab in enumerate(labels)}
