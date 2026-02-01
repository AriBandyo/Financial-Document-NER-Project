import argparse
from pathlib import Path

from transformers import pipeline


def run_inference(text: str, model_path: str) -> None:
    ner = pipeline(
        task="token-classification",
        model=model_path,
        aggregation_strategy="simple",
    )

    results = ner(text)

    print("\nINPUT:")
    print(text)

    print("\nPREDICTIONS:")
    if not results:
        print("(no entities found)")
        return

    for r in results:
        word = r.get("word", "")
        entity = r.get("entity_group", r.get("entity", ""))
        score = r.get("score", 0.0)
        start = r.get("start", None)
        end = r.get("end", None)
        print(f"- {entity:12} | {word:20} | score={score:.3f} | span=({start},{end})")


def main():
    parser = argparse.ArgumentParser(description="Financial Doc NER - Inference")
    parser.add_argument("--text", type=str, required=True, help="Input text to run NER on")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path("artifacts") / "fin_ner_model"),
        help="Model name or local path. Default uses your fine-tuned model in artifacts/fin_ner_model",
    )
    args = parser.parse_args()
    run_inference(args.text, args.model)


if __name__ == "__main__":
    main()
