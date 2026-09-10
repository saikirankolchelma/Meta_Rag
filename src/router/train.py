"""Fine-tunes DeBERTa-v3-base as a regression head to predict query complexity_score.

Usage:
    python -m src.router.train [--epochs 5] [--batch-size 32] [--lr 2e-5]
"""

import argparse
import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.utils.config import DATA_PROCESSED_DIR, ROUTER_MODEL_DIR

BASE_MODEL = "microsoft/deberta-v3-base"
DATA_PATH = DATA_PROCESSED_DIR / "synthetic_queries.jsonl"


def load_dataset():
    rows = [json.loads(line) for line in DATA_PATH.open(encoding="utf-8")]
    queries = [r["query"] for r in rows]
    scores = [float(r["complexity_score"]) for r in rows]
    train_q, val_q, train_s, val_s = train_test_split(queries, scores, test_size=0.1, random_state=42)
    train_ds = Dataset.from_dict({"text": train_q, "label": train_s})
    val_ds = Dataset.from_dict({"text": val_q, "label": val_s})
    return train_ds, val_ds


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.squeeze(-1)
    mse = mean_squared_error(labels, predictions)
    return {"mse": mse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found. Run generate_synthetic.py first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_ds, val_ds = load_dataset()
    print(f"Train examples: {len(train_ds)}, Val examples: {len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=64)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=1, problem_type="regression", torch_dtype=torch.float32
    ).to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(ROUTER_MODEL_DIR / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="mse",
        greater_is_better=False,
        # Mixed precision disabled: this laptop GPU hit a CUBLAS_STATUS_EXECUTION_FAILED
        # mid-run on bf16 GEMM. Dataset is tiny, so plain fp32 is fast enough and reliable.
        logging_steps=10,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"Final validation MSE: {metrics['eval_mse']:.4f}")

    ROUTER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ROUTER_MODEL_DIR))
    tokenizer.save_pretrained(str(ROUTER_MODEL_DIR))
    print(f"Saved model to {ROUTER_MODEL_DIR}")


if __name__ == "__main__":
    main()
