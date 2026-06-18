#!/usr/bin/env python3
"""Train DistilRoBERTa from explicit train/val/test CSV files."""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    hamming_loss,
    jaccard_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
]


def parse_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            labels = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            labels = []
        if isinstance(labels, str):
            labels = [labels]
        return [str(label).strip() for label in labels if str(label).strip() in LABELS]
    return [part.strip() for part in text.split(",") if part.strip() in LABELS]


def load_split(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    df["labels"] = df["sentiment"].apply(parse_labels)
    df = df[df["labels"].map(len) > 0].reset_index(drop=True)
    mlb = MultiLabelBinarizer(classes=LABELS)
    mlb.fit([LABELS])
    return df["text"].astype(str).tolist(), mlb.transform(df["labels"])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_overall_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, thresholds: np.ndarray | None = None) -> dict[str, float]:
    if thresholds is None:
        thresholds = np.full(y_prob.shape[1], threshold)
    y_pred = (y_prob >= thresholds).astype(int)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "macro_auc": macro_auc(y_true, y_prob),
        "macro_label_accuracy": float((y_true == y_pred).mean(axis=0).mean()),
    }


def compute_per_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    y_pred = (y_prob >= thresholds).astype(int)
    rows = []
    for i, label in enumerate(LABELS):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        auc = np.nan if len(np.unique(y_true[:, i])) < 2 else roc_auc_score(y_true[:, i], y_prob[:, i])
        rows.append(
            {
                "label": label,
                "accuracy": accuracy_score(y_true[:, i], y_pred[:, i]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "support": int(y_true[:, i].sum()),
                "threshold": float(thresholds[i]),
            }
        )
    return pd.DataFrame(rows)


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray, step: float) -> np.ndarray:
    grid = np.arange(step, 1.0, step)
    best = np.zeros(y_prob.shape[1], dtype=np.float32)
    for i in range(y_prob.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in grid:
            y_pred = (y_prob[:, i] >= t).astype(int)
            f1 = precision_recall_fscore_support(
                y_true[:, i], y_pred, average="binary", zero_division=0
            )[2]
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best[i] = best_t
    return best


def make_training_args(**kwargs) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def maybe_subsample(texts: list[str], labels: np.ndarray, size: int, seed: int) -> tuple[list[str], np.ndarray]:
    if not size or size >= len(texts):
        return texts, labels
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(texts), size=size, replace=False)
    return [texts[i] for i in idx], labels[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--val_path", required=True)
    parser.add_argument("--test_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="distilroberta-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--optimize_thresholds", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split(Path(args.train_path))
    x_val, y_val = load_split(Path(args.val_path))
    x_test, y_test = load_split(Path(args.test_path))
    x_train, y_train = maybe_subsample(x_train, y_train, args.max_train_samples, args.seed)
    x_val, y_val = maybe_subsample(x_val, y_val, args.max_eval_samples, args.seed)
    x_test, y_test = maybe_subsample(x_test, y_test, args.max_eval_samples, args.seed)

    train_ds = Dataset.from_dict({"text": x_train, "labels": y_train.astype(np.float32)})
    val_ds = Dataset.from_dict({"text": x_val, "labels": y_val.astype(np.float32)})
    test_ds = Dataset.from_dict({"text": x_test, "labels": y_test.astype(np.float32)})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=["text"])
    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        label2id=label2id,
        id2label=id2label,
    )

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        return compute_overall_metrics(eval_pred.label_ids, sigmoid(logits), threshold=0.5)

    training_args = make_training_args(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    val_pred = trainer.predict(val_tok)
    val_logits = val_pred.predictions[0] if isinstance(val_pred.predictions, tuple) else val_pred.predictions
    val_probs = sigmoid(val_logits)
    val_labels = val_pred.label_ids

    test_pred = trainer.predict(test_tok)
    test_logits = test_pred.predictions[0] if isinstance(test_pred.predictions, tuple) else test_pred.predictions
    test_probs = sigmoid(test_logits)
    test_labels = test_pred.label_ids

    thresholds_05 = np.full(len(LABELS), 0.5)
    metrics_05 = compute_overall_metrics(test_labels, test_probs, threshold=0.5)
    compute_per_label_metrics(test_labels, test_probs, thresholds_05).to_csv(
        output_dir / "per_label_metrics_thr0.5.csv", index=False
    )

    summary = {
        "protocol": "explicit train/val/test splits",
        "model_name": args.model_name,
        "train_path": args.train_path,
        "val_path": args.val_path,
        "test_path": args.test_path,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "test_size": len(x_test),
        "label_names": LABELS,
        "metrics_thr0.5": metrics_05,
    }

    if args.optimize_thresholds:
        thresholds_opt = optimize_thresholds(val_labels, val_probs, step=args.threshold_step)
        metrics_opt = compute_overall_metrics(test_labels, test_probs, thresholds=thresholds_opt)
        compute_per_label_metrics(test_labels, test_probs, thresholds_opt).to_csv(
            output_dir / "per_label_metrics_thr_opt.csv", index=False
        )
        pd.DataFrame({"label": LABELS, "threshold": thresholds_opt}).to_csv(
            output_dir / "thresholds_opt.csv", index=False
        )
        summary["metrics_thr_opt"] = metrics_opt
        summary["thresholds_opt"] = thresholds_opt.tolist()

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Done. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
