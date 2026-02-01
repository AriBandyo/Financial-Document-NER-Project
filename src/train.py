from __future__ import annotations

from pathlib import Path

import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src.dataset import load_dataset


def compute_metrics_fn(id2label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=2)

        true_preds = []
        true_labels = []

        for pred_row, label_row in zip(preds, labels):
            curr_preds = []
            curr_labels = []
            for p_i, l_i in zip(pred_row, label_row):
                if l_i == -100:
                    continue
                curr_preds.append(id2label[int(p_i)])
                curr_labels.append(id2label[int(l_i)])
            true_preds.append(curr_preds)
            true_labels.append(curr_labels)

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return compute_metrics


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / "data" / "train.conll"
    valid_path = project_root / "data" / "valid.conll"

    model_name = "bert-base-cased"

    train_rows, label2id, id2label = load_dataset(train_path, model_name)
    valid_rows, _, _ = load_dataset(valid_path, model_name)

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=str(project_root / "artifacts" / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=1,
        evaluation_strategy="epoch",
        save_strategy="no",
        report_to="none",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn(id2label),
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("\n✅ Final eval:", metrics)

    out_dir = project_root / "artifacts" / "fin_ner_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n✅ Saved fine-tuned model to: {out_dir}")


if __name__ == "__main__":
    main()
