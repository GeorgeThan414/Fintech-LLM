"""Compute sentiment-benchmark metrics from the per-engine prediction files.

Reads results/eval_set.csv + every results/preds_<engine>.csv and writes:
  results/sentiment_metrics.csv   summary (accuracy, macro-F1, weighted-F1, unknown%)
  results/sentiment_metrics.tex   booktabs table ready for the paper
  results/perclass_<engine>.csv   per-class precision / recall / F1
  results/confusion_<engine>.csv  confusion matrix counts (rows=true, cols=pred)

Run after one or more engines have produced predictions:
  python -m evaluation.metrics
"""

import argparse
import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.data import DATASETS, LABELS, load_eval_set  # noqa: E402

RESULTS = os.path.join(ROOT, "results")

DATASET_TITLE = {
    "fpb": "Financial PhraseBank (FLARE-FPB) test split",
    "fiqa": "FiQA-2018 SA (FLARE-FiQA) test split",
}

# Display names for the paper (engine key -> pretty label)
DISPLAY = {
    "finbert": "FinBERT (ProsusAI)",
    "groq": "Groq (Llama-3.3-70B)",
    "fingpt": "FinGPT (Llama-2-7B base)",
}


def _load_preds(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["pred"] for r in csv.DictReader(f)}


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _latex_table(summary, dataset):
    """Render the summary list as a booktabs tabular."""
    title = DATASET_TITLE.get(dataset, dataset)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Sentiment classification on the %s (%d sentences). "
        r"Best per column in \textbf{bold}.}" % (title, summary[0]["n"]),
        r"  \label{tab:sentiment-%s}" % dataset,
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Engine & Accuracy & Macro-F1 & Weighted-F1 & Unknown (\%) \\",
        r"    \midrule",
    ]
    best_acc = max(s["accuracy"] for s in summary)
    best_mf1 = max(s["macro_f1"] for s in summary)
    best_wf1 = max(s["weighted_f1"] for s in summary)

    def fmt(val, is_best):
        s = f"{val:.3f}"
        return rf"\textbf{{{s}}}" if is_best else s

    for s in summary:
        lines.append(
            "    %s & %s & %s & %s & %.1f \\\\"
            % (
                s["display"],
                fmt(s["accuracy"], s["accuracy"] == best_acc),
                fmt(s["macro_f1"], s["macro_f1"] == best_mf1),
                fmt(s["weighted_f1"], s["weighted_f1"] == best_wf1),
                100.0 * s["n_unknown"] / s["n"],
            )
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="fpb", choices=list(DATASETS))
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS, args.dataset)
    eval_set = os.path.join(out_dir, "eval_set.csv")
    if not os.path.exists(eval_set):
        sys.exit(f"eval set not found: {eval_set} (run evaluation.run_sentiment --dataset {args.dataset} first)")

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    truth = {r["id"]: r["label"] for r in load_eval_set(eval_set)}
    pred_files = sorted(glob.glob(os.path.join(out_dir, "preds_*.csv")))
    if not pred_files:
        sys.exit(f"no {out_dir}/preds_*.csv found — run evaluation.run_sentiment first")

    summary = []
    for pf in pred_files:
        engine = os.path.basename(pf)[len("preds_"):-len(".csv")]
        preds = _load_preds(pf)
        ids = [i for i in truth if i in preds]
        y_true = [truth[i] for i in ids]
        y_pred = [preds[i] for i in ids]

        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
        n_unknown = sum(1 for p in y_pred if p not in LABELS)

        summary.append({
            "engine": engine,
            "display": DISPLAY.get(engine, engine),
            "n": len(ids),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "n_unknown": n_unknown,
        })

        # Per-class precision / recall / F1
        p, r, fsc, sup = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        _write_csv(
            os.path.join(out_dir, f"perclass_{engine}.csv"),
            ["label", "precision", "recall", "f1", "support"],
            [[LABELS[i], f"{p[i]:.4f}", f"{r[i]:.4f}", f"{fsc[i]:.4f}", int(sup[i])]
             for i in range(len(LABELS))],
        )

        # Confusion matrix (include 'unknown' as a predicted column if it occurs)
        cm_labels = LABELS + (["unknown"] if n_unknown else [])
        cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
        _write_csv(
            os.path.join(out_dir, f"confusion_{engine}.csv"),
            ["true\\pred"] + cm_labels,
            [[cm_labels[i]] + list(map(int, cm[i])) for i in range(len(LABELS))],
        )

        print(f"[{engine}] n={len(ids)}  acc={acc:.3f}  macroF1={macro_f1:.3f}  "
              f"weightedF1={weighted_f1:.3f}  unknown={n_unknown}")

    # Summary CSV + LaTeX
    summary.sort(key=lambda s: s["macro_f1"], reverse=True)
    _write_csv(
        os.path.join(out_dir, "sentiment_metrics.csv"),
        ["engine", "n", "accuracy", "macro_f1", "weighted_f1", "n_unknown"],
        [[s["engine"], s["n"], f"{s['accuracy']:.4f}", f"{s['macro_f1']:.4f}",
          f"{s['weighted_f1']:.4f}", s["n_unknown"]] for s in summary],
    )
    tex_path = os.path.join(out_dir, "sentiment_metrics.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_latex_table(summary, args.dataset))

    print(f"\n[metrics] {args.dataset}: wrote sentiment_metrics.csv + .tex "
          f"({len(summary)} engine(s))")


if __name__ == "__main__":
    main()
