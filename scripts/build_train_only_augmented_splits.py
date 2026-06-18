#!/usr/bin/env python3
"""Build duplicate-aware train-only augmentation splits for Reviewer 2 reruns."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer


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
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = []
        if isinstance(parsed, str):
            return [parsed.strip()] if parsed.strip() else []
        return [str(label).strip() for label in parsed if str(label).strip() in LABELS]
    return [part.strip() for part in text.split(",") if part.strip() in LABELS]


def labels_to_string(labels: list[str]) -> str:
    return repr([label for label in LABELS if label in set(labels)])


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def aggregate_by_text(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = df.groupby("norm_text", sort=False)
    for norm_text, group in grouped:
        labels: set[str] = set()
        sources: set[str] = set()
        for value in group["labels"]:
            labels.update(value)
        for value in group["source"]:
            sources.update(str(value).split("+"))
        labels = {label for label in labels if label in LABELS}
        if not labels:
            continue
        rows.append(
            {
                "text": group["text"].iloc[0],
                "sentiment": labels_to_string(sorted(labels)),
                "labels": sorted(labels),
                "norm_text": norm_text,
                "source": "+".join(sorted(sources)),
                "row_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def load_goemotions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "example_very_unclear" in df.columns:
        df = df[df["example_very_unclear"] == False].copy()  # noqa: E712
    rows = []
    for _, row in df.iterrows():
        labels = [label for label in LABELS if label in row.index and int(row[label]) == 1]
        if not labels and "neutral" in row.index:
            labels = ["neutral"]
        rows.append(
            {
                "text": row["text"],
                "labels": labels,
                "norm_text": normalize_text(row["text"]),
                "source": "goemotions",
            }
        )
    return aggregate_by_text(pd.DataFrame(rows))


def load_sentiment140(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    rows = []
    for _, row in df.iterrows():
        labels = parse_labels(row["model_labels"])
        if labels:
            rows.append(
                {
                    "text": row["text"],
                    "labels": labels,
                    "norm_text": normalize_text(row["text"]),
                    "source": "sentiment140",
                }
            )
    return aggregate_by_text(pd.DataFrame(rows))


def load_gpt_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", engine="python")
    rows = []
    for _, row in df.iterrows():
        labels = parse_labels(row["majority_label"])
        if labels:
            rows.append(
                {
                    "text": row["text"],
                    "labels": labels,
                    "norm_text": normalize_text(row["text"]),
                    "source": "gpt4mini",
                }
            )
    return aggregate_by_text(pd.DataFrame(rows))


def make_y(df: pd.DataFrame) -> np.ndarray:
    mlb = MultiLabelBinarizer(classes=LABELS)
    mlb.fit([LABELS])
    return mlb.transform(df["labels"])


def multilabel_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = make_y(df)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(msss.split(df["text"], y))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    y_train_val = y[train_val_idx]
    val_frac = val_size / (1.0 - test_size)
    msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(msss_val.split(train_val["text"], y_train_val))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return train, val, test


def label_counts(df: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for labels in df["labels"]:
        counter.update(labels)
    return counter


def counts_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    counts = label_counts(df)
    return pd.DataFrame({"split": name, "label": LABELS, "count": [int(counts.get(label, 0)) for label in LABELS]})


def greedy_balance(df: pd.DataFrame, seed: int, max_samples: int, neutral_cap: int, annoyance_cap: int) -> pd.DataFrame:
    caps = {label: max_samples for label in LABELS}
    caps["neutral"] = neutral_cap
    caps["annoyance"] = annoyance_cap

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    counts = {label: 0 for label in LABELS}
    selected = []
    for _, row in shuffled.iterrows():
        labels = row["labels"]
        if not labels:
            continue
        if any(counts[label] < caps[label] for label in labels):
            selected.append(row)
            for label in labels:
                counts[label] += 1
    return pd.DataFrame(selected).reset_index(drop=True)


def strip_internal_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["text", "sentiment", "source"]].copy()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goemotions_path", required=True)
    parser.add_argument("--sentiment140_path", required=True)
    parser.add_argument("--gpt_annotations_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=9000)
    parser.add_argument("--neutral_cap", type=int, default=20000)
    parser.add_argument("--annoyance_cap", type=int, default=15000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    go = load_goemotions(Path(args.goemotions_path))
    go_train, go_val, go_test = multilabel_split(go, args.seed, args.test_size, args.val_size)

    val_test_norm = set(go_val["norm_text"]) | set(go_test["norm_text"])
    s140 = load_sentiment140(Path(args.sentiment140_path))
    gpt = load_gpt_annotations(Path(args.gpt_annotations_path))
    added = pd.concat([s140, gpt], ignore_index=True)
    added_before_filter = len(added)
    added = added[~added["norm_text"].isin(val_test_norm)].reset_index(drop=True)

    train_candidates = pd.concat([go_train, added], ignore_index=True)
    train_candidates = aggregate_by_text(train_candidates[["text", "labels", "norm_text", "source"]])
    train_balanced = greedy_balance(
        train_candidates,
        seed=args.seed,
        max_samples=args.max_samples,
        neutral_cap=args.neutral_cap,
        annoyance_cap=args.annoyance_cap,
    )

    train_norm = set(train_balanced["norm_text"])
    overlap_train_val = len(train_norm & set(go_val["norm_text"]))
    overlap_train_test = len(train_norm & set(go_test["norm_text"]))

    strip_internal_columns(train_balanced).to_csv(output_dir / "train.csv", index=False)
    strip_internal_columns(go_val).to_csv(output_dir / "val.csv", index=False)
    strip_internal_columns(go_test).to_csv(output_dir / "test.csv", index=False)

    counts = pd.concat(
        [
            counts_frame("train", train_balanced),
            counts_frame("val", go_val),
            counts_frame("test", go_test),
        ],
        ignore_index=True,
    )
    counts.to_csv(output_dir / "label_counts_by_split.csv", index=False)

    summary = {
        "protocol": "duplicate-aware train-only augmentation",
        "seed": args.seed,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "goemotions_unique_rows": int(len(go)),
        "goemotions_train_rows": int(len(go_train)),
        "goemotions_val_rows": int(len(go_val)),
        "goemotions_test_rows": int(len(go_test)),
        "sentiment140_unique_rows": int(len(s140)),
        "gpt4mini_unique_rows": int(len(gpt)),
        "added_rows_before_val_test_filter": int(added_before_filter),
        "added_rows_after_val_test_filter": int(len(added)),
        "train_candidate_rows": int(len(train_candidates)),
        "balanced_train_rows": int(len(train_balanced)),
        "balanced_train_unique_texts": int(train_balanced["norm_text"].nunique()),
        "val_unique_texts": int(go_val["norm_text"].nunique()),
        "test_unique_texts": int(go_test["norm_text"].nunique()),
        "overlap_train_val_normalized_texts": int(overlap_train_val),
        "overlap_train_test_normalized_texts": int(overlap_train_test),
        "caps": {
            "default": args.max_samples,
            "neutral": args.neutral_cap,
            "annoyance": args.annoyance_cap,
        },
    }
    (output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
