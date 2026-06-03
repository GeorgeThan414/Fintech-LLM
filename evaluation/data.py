"""Load labeled financial-sentiment benchmarks (FLARE format).

Two datasets, same 3-class scheme (positive / negative / neutral):

  fpb   ChanceFocus/flare-fpb     Financial PhraseBank (Malo et al. 2014).
        IN-DOMAIN for FinBERT/FinGPT (they trained on it). test split = 970.
  fiqa  ChanceFocus/flare-fiqasa  FiQA-2018 SA (tweets/headlines). OUT-OF-DOMAIN
        for FinBERT (it never saw FiQA) — the fair cross-check. test split = 235.

Both expose `text` (the sentence) and `answer` (the gold label). The eval set is
materialized once per dataset to results/<dataset>/eval_set.csv so every engine
scores identical examples.
"""

import csv
import os
import random
from typing import List, Optional, Tuple

LABELS = ["positive", "negative", "neutral"]

# dataset key -> (HF repo, default split)
DATASETS = {
    "fpb": ("ChanceFocus/flare-fpb", "test"),
    "fiqa": ("ChanceFocus/flare-fiqasa", "test"),
}


def _load_raw(dataset: str, split: str) -> List[Tuple[str, str]]:
    from datasets import load_dataset

    repo, default_split = DATASETS[dataset]
    ds = load_dataset(repo, split=split or default_split)
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
    dataset: str = "fpb",
    split: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 42,
) -> str:
    """Materialize the shared eval set to `path` as CSV (id, text, label)."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}' (choices: {list(DATASETS)})")
    rows = _load_raw(dataset, split)
    if limit and limit < len(rows):
        rows = _stratified_sample(rows, limit, seed)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "label"])
        for i, (text, label) in enumerate(rows):
            w.writerow([f"{dataset}-{i}", text, label])
    return path


def load_eval_set(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
