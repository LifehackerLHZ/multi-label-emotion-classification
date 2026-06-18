#!/usr/bin/env python3
import argparse
import ast
import inspect
import json
import os
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
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


def parse_labels(value):
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            labels = ast.literal_eval(text)
            return [str(v).strip() for v in labels if str(v).strip()]
        except Exception:
            pass
    return [v.strip() for v in text.split(",") if v.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def macro_auc(y_true, y_prob):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_overall_metrics(y_true, y_prob, threshold=0.5, thresholds=None):
    if thresholds is None:
        thresholds = np.full(y_prob.shape[1], threshold)
    y_pred = (y_prob >= thresholds).astype(int)

    subset_acc = accuracy_score(y_true, y_pred)
    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    hamming = hamming_loss(y_true, y_pred)

    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    macro_auc_val = macro_auc(y_true, y_prob)
    macro_label_acc = float((y_true == y_pred).mean(axis=0).mean())

    return {
        "subset_accuracy": float(subset_acc),
        "jaccard": float(jaccard),
        "hamming_loss": float(hamming),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "macro_auc": float(macro_auc_val),
        "macro_label_accuracy": float(macro_label_acc),
    }


def compute_per_label_metrics(y_true, y_prob, thresholds):
    y_pred = (y_prob >= thresholds).astype(int)
    rows = []
    for i, label in enumerate(LABELS):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        y_pb = y_prob[:, i]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="binary", zero_division=0
        )
        if len(np.unique(y_t)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_t, y_pb)

        rows.append(
            {
                "label": label,
                "accuracy": accuracy_score(y_t, y_p),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "auc": auc,
                "support": int(y_t.sum()),
                "threshold": float(thresholds[i]),
            }
        )
    return pd.DataFrame(rows)


def optimize_thresholds(y_true, y_prob, step=0.05):
    thresholds = np.arange(step, 1.0, step)
    best = np.zeros(y_prob.shape[1], dtype=np.float32)

    for i in range(y_prob.shape[1]):
        best_f1 = -1.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_prob[:, i] >= t).astype(int)
            f1 = precision_recall_fscore_support(
                y_true[:, i], y_pred, average="binary", zero_division=0
            )[2]
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best[i] = best_t

    return best


def make_training_args(**kwargs):
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="distilroberta-base")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--threshold_step", type=float, default=0.05)
    parser.add_argument("--optimize_thresholds", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path, encoding="utf-8", engine="python")
    df["labels"] = df["sentiment"].apply(parse_labels)
    df["labels"] = df["labels"].apply(lambda labs: [l for l in labs if l in LABELS])
    df = df[df["labels"].map(len) > 0].reset_index(drop=True)

    mlb = MultiLabelBinarizer(classes=LABELS)
    mlb.fit([LABELS])
    y = mlb.transform(df["labels"])

    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_val_idx, test_idx = next(msss.split(df["text"], y))
    X_train_val = df["text"].iloc[train_val_idx].tolist()
    y_train_val = y[train_val_idx]

    val_frac = args.val_size / (1.0 - args.test_size)
    msss_val = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=val_frac, random_state=args.seed
    )
    train_idx, val_idx = next(msss_val.split(X_train_val, y_train_val))
    X_train = [X_train_val[i] for i in train_idx]
    y_train = y_train_val[train_idx]
    X_val = [X_train_val[i] for i in val_idx]
    y_val = y_train_val[val_idx]
    X_test = df["text"].iloc[test_idx].tolist()
    y_test = y[test_idx]

    if args.max_train_samples and args.max_train_samples < len(X_train):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(X_train), size=args.max_train_samples, replace=False)
        X_train = [X_train[i] for i in idx]
        y_train = y_train[idx]

    if args.max_eval_samples and args.max_eval_samples < len(X_val):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(X_val), size=args.max_eval_samples, replace=False)
        X_val = [X_val[i] for i in idx]
        y_val = y_val[idx]

    if args.max_eval_samples and args.max_eval_samples < len(X_test):
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(len(X_test), size=args.max_eval_samples, replace=False)
        X_test = [X_test[i] for i in idx]
        y_test = y_test[idx]

    train_ds = Dataset.from_dict({"text": X_train, "labels": y_train.astype(np.float32)})
    val_ds = Dataset.from_dict({"text": X_val, "labels": y_val.astype(np.float32)})
    test_ds = Dataset.from_dict({"text": X_test, "labels": y_test.astype(np.float32)})

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=["text"])
    test_tok = test_ds.map(tokenize, batched=True, remove_columns=["text"])

    train_tok.set_format("torch")
    val_tok.set_format("torch")
    test_tok.set_format("torch")

    data_collator = DataCollatorWithPadding(tokenizer)

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
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = sigmoid(logits)
        labels = eval_pred.label_ids
        return compute_overall_metrics(labels, probs, threshold=0.5)

    training_args = make_training_args(
        output_dir=args.output_dir,
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
        data_collator=data_collator,
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
    per_label_05 = compute_per_label_metrics(test_labels, test_probs, thresholds_05)
    per_label_05.to_csv(os.path.join(args.output_dir, "per_label_metrics_thr0.5.csv"), index=False)

    summary = {
        "model_name": args.model_name,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "label_names": LABELS,
        "metrics_thr0.5": metrics_05,
    }

    if args.optimize_thresholds:
        thresholds_opt = optimize_thresholds(val_labels, val_probs, step=args.threshold_step)
        metrics_opt = compute_overall_metrics(test_labels, test_probs, thresholds=thresholds_opt)
        per_label_opt = compute_per_label_metrics(test_labels, test_probs, thresholds_opt)
        per_label_opt.to_csv(os.path.join(args.output_dir, "per_label_metrics_thr_opt.csv"), index=False)

        thresholds_df = pd.DataFrame({"label": LABELS, "threshold": thresholds_opt})
        thresholds_df.to_csv(os.path.join(args.output_dir, "thresholds_opt.csv"), index=False)

        summary["metrics_thr_opt"] = metrics_opt
        summary["thresholds_opt"] = thresholds_opt.tolist()

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
