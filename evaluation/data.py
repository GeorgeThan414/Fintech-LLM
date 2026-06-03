"""Load the Financial PhraseBank (FPB) sentiment benchmark.

Source: ``ChanceFocus/flare-fpb`` on the HuggingFace Hub — a Parquet mirror of the
Financial PhraseBank (Malo et al., 2014) reformatted for the FLARE financial-LLM
benchmark. This is the same FPB split FinGPT and other financial LLMs report
against, so our numbers line up with published leaderboards.

Columns of interest: ``text`` (the sentence) and ``answer`` (positive/neutral/negative).
We use the ``test`` split (970 sentences) as the evaluation set.

The eval set is materialized once to ``results/eval_set.csv`` so that *every* engine
scores the exact same examples (a fair head-to-head). Change the size with --limit
+ --rebuild in run_sentiment.py.
"""

import csv
import os
import random
from typing import List, Optional, Tuple

LABELS = ["positive", "negative", "neutral"]
DATASET = "ChanceFocus/flare-fpb"


def _load_raw(split: str) -> List[Tuple[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=split)
    rows: List[Tuple[str, str]] = []
    for ex in ds:
        text = (ex.get("text") or "").strip()
        label = (ex.get("answer") or "").strip().lower()
        if text and label in LABELS:
            rows.append((text, label))
    return rows


def _stratified_sample(rows: List[Tuple[str, str]], limit: int, seed: int) -> List[Tuple[str, str]]:
    """Subsample to `limit` examples while preserving the class balance."""
    by_label = {}
    for r in rows:
        by_label.setdefault(r[1], []).append(r)

    rng = random.Random(seed)
    total = len(rows)
    out: List[Tuple[str, str]] = []
    for label, items in by_label.items():
        k = max(1, round(limit * len(items) / total))
        rng.shuffle(items)
        out.extend(items[:k])
    rng.shuffle(out)
    return out[:limit]


def build_eval_set(
    path: str,
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 42,
) -> str:
    """Materialize the shared eval set to `path` as CSV (id, text, label)."""
    rows = _load_raw(split)
    if limit and limit < len(rows):
        rows = _stratified_sample(rows, limit, seed)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label"])
        for i, (text, label) in enumerate(rows):
            w.writerow([f"fpb-{i}", text, label])
    return path


def load_eval_set(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
