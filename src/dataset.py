from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

from src.data_loader import Sentence, load_conll
from src.labels import build_label_map


@dataclass
class NerBatch:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]


def bio_to_i(tag: str) -> str:
    if tag.startswith("B-"):
        return "I-" + tag[2:]
    return tag


def tokenize_and_align(sentence: Sentence, tokenizer, label2id: Dict[str, int]) -> NerBatch:
    enc = tokenizer(
        sentence.tokens,
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = enc.word_ids()
    labels: List[int] = []

    prev_word_id = None
    for wi in word_ids:
        if wi is None:
            labels.append(-100)
        else:
            orig_tag = sentence.tags[wi]
            tag = orig_tag if wi != prev_word_id else bio_to_i(orig_tag)
            labels.append(label2id.get(tag, label2id["O"]))
        prev_word_id = wi

    return NerBatch(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        labels=labels,
    )


def load_dataset(data_path: Path, model_name: str) -> Tuple[List[Dict], Dict[str, int], Dict[int, str]]:
    sents = load_conll(data_path)

    # Build label map from ALL tags in dataset (not per sentence)
    all_tags: List[str] = []
    for s in sents:
        all_tags.extend(s.tags)
    label2id = build_label_map(all_tags)
    id2label = {v: k for k, v in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows: List[Dict] = []
    for s in sents:
        batch = tokenize_and_align(s, tokenizer, label2id)
        rows.append(
            {
                "input_ids": batch.input_ids,
                "attention_mask": batch.attention_mask,
                "labels": batch.labels,
            }
        )

    return rows, label2id, id2label
