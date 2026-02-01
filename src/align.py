from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from transformers import AutoTokenizer

from src.data_loader import load_conll
from src.labels import build_label_map


def bio_to_i(tag: str) -> str:
    # B-XXX -> I-XXX, otherwise unchanged
    if tag.startswith("B-"):
        return "I-" + tag[2:]
    return tag


def align_labels_with_tokenizer(
    tokens: List[str],
    tags: List[str],
    tokenizer,
) -> Tuple[List[str], List[str], List[int]]:
    """
    Align BIO tags to BERT wordpieces.

    Returns:
      - wp_tokens: wordpiece tokens including special tokens ([CLS], [SEP])
      - aligned_tags: string tags aligned to each wordpiece (IGN for specials)
      - label_ids: numeric IDs aligned to each wordpiece (-100 for specials)
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
    )

    word_ids = encoding.word_ids()  # maps each wordpiece to original token index (or None)
    wp_tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    #  IMPORTANT: build a label map that includes I-XXX for every B-XXX
    label2id = build_label_map(tags)

    aligned_tags: List[str] = []
    label_ids: List[int] = []

    prev_word_id = None
    for wi in word_ids:
        if wi is None:
            aligned_tags.append("IGN")
            label_ids.append(-100)
        else:
            original_tag = tags[wi]

            # first wordpiece -> use original tag
            if wi != prev_word_id:
                tag = original_tag
            else:
                # subsequent wordpieces -> convert B-XXX to I-XXX
                tag = bio_to_i(original_tag)

            aligned_tags.append(tag)
            label_ids.append(label2id.get(tag, label2id["O"]))

        prev_word_id = wi

    return wp_tokens, aligned_tags, label_ids


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "sample.conll"

    sents = load_conll(data_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    for i, s in enumerate(sents, start=1):
        wp_tokens, aligned_tags, label_ids = align_labels_with_tokenizer(
            s.tokens, s.tags, tokenizer
        )

        print(f"\n=== Sentence {i} ===")
        for t, lab, lid in zip(wp_tokens, aligned_tags, label_ids):
            print(f"{t:15} {lab:12} {lid}")


if __name__ == "__main__":
    main()
