from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Sentence:
    tokens: List[str]
    tags: List[str]


def load_conll(path: Path) -> List[Sentence]:
    sentences: List[Sentence] = []
    tokens: List[str] = []
    tags: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # blank line = sentence boundary
            if not line:
                if tokens:
                    sentences.append(Sentence(tokens=tokens, tags=tags))
                    tokens, tags = [], []
                continue

            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Bad line (expected 2 columns TOKEN TAG): {line}")

            tok, tag = parts
            tokens.append(tok)
            tags.append(tag)

    if tokens:
        sentences.append(Sentence(tokens=tokens, tags=tags))

    return sentences


def main() -> None:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    path = PROJECT_ROOT / "data" / "sample.conll"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")

    sents = load_conll(path)
    print(f"Loaded {len(sents)} sentences")

    for i, s in enumerate(sents, start=1):
        print(f"\nSentence {i}")
        print("TOKENS:", s.tokens)
        print("TAGS:  ", s.tags)



if __name__ == "__main__":
    main()
