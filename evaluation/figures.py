"""Generate publication figures (PDF, vector) from the metrics CSVs.

Outputs to results/<dataset>/figures/:
  accuracy_macrof1.pdf    grouped bar chart, accuracy + macro-F1 per engine
  confusion_<engine>.pdf  confusion-matrix heatmap per engine
  forecast_accuracy.pdf   directional + 3-class accuracy per engine (--dataset forecast)

Run after evaluation.metrics:
  python -m evaluation.figures                     # fpb (default)
  python -m evaluation.figures --dataset fiqa
  python -m evaluation.figures --dataset forecast
"""

import argparse
import csv
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DISPLAY = {
    "finbert": "FinBERT",
    "groq": "Groq\n(Llama-3.3-70B)",
    "fingpt": "FinGPT\n(Llama-2-7B)",
    "momentum": "Momentum",
    "naive_rise": "Naive\n(always Rise)",
    "random": "Random",
}

# Order for the forecast chart: models first, then baselines.
FORECAST_ORDER = ["fingpt", "groq", "momentum", "naive_rise", "random"]


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def accuracy_bar(out_dir, figdir):
    path = os.path.join(out_dir, "sentiment_metrics.csv")
    if not os.path.exists(path):
        print("skip accuracy_bar: sentiment_metrics.csv missing")
        return
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    engines = [r["engine"] for r in rows]
    labels = [DISPLAY.get(e, e) for e in engines]
    acc = [float(r["accuracy"]) for r in rows]
    mf1 = [float(r["macro_f1"]) for r in rows]

    x = np.arange(len(engines))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color="#2563eb")
    b2 = ax.bar(x + w / 2, mf1, w, label="Macro-F1", color="#16a34a")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Financial PhraseBank (test) — sentiment performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = os.path.join(figdir, "accuracy_macrof1.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def confusion_heatmaps(out_dir, figdir):
    for path in sorted(glob.glob(os.path.join(out_dir, "confusion_*.csv"))):
        engine = os.path.basename(path)[len("confusion_"):-len(".csv")]
        grid = _read_csv(path)
        col_labels = grid[0][1:]
        row_labels = [row[0] for row in grid[1:]]
        mat = np.array([[int(v) for v in row[1:]] for row in grid[1:]])

        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(col_labels)), labels=col_labels)
        ax.set_yticks(range(len(row_labels)), labels=row_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion — {DISPLAY.get(engine, engine).replace(chr(10), ' ')}")
        thresh = mat.max() / 2 if mat.max() else 0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, mat[i, j], ha="center", va="center",
                        color="white" if mat[i, j] > thresh else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = os.path.join(figdir, f"confusion_{engine}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def forecast_accuracy_bar(out_dir, figdir):
    path = os.path.join(out_dir, "forecast_metrics.csv")
    if not os.path.exists(path):
        print("skip forecast_accuracy_bar: forecast_metrics.csv missing")
        return
    rows = {r["engine"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}
    engines = [e for e in FORECAST_ORDER if e in rows] + \
        [e for e in rows if e not in FORECAST_ORDER]
    labels = [DISPLAY.get(e, e) for e in engines]
    direc = [float(rows[e]["directional_accuracy"]) for e in engines]
    three = [float(rows[e]["three_class_accuracy"]) for e in engines]

    x = np.arange(len(engines))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w / 2, direc, w, label="Directional accuracy", color="#2563eb")
    b2 = ax.bar(x + w / 2, three, w, label="3-class accuracy", color="#2dd4bf")
    ax.axhline(0.5, ls="--", color="#9ca3af", lw=1)
    ax.text(len(engines) - 0.5, 0.5, "chance (0.5)", ha="right", va="bottom",
            fontsize=8, color="#9ca3af")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = os.path.join(figdir, "forecast_accuracy.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="fpb", choices=["fpb", "fiqa", "forecast"])
    args = ap.parse_args()

    out_dir = os.path.join(RESULTS, args.dataset)
    figdir = os.path.join(out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    if args.dataset == "forecast":
        forecast_accuracy_bar(out_dir, figdir)
    else:
        accuracy_bar(out_dir, figdir)
        confusion_heatmaps(out_dir, figdir)


if __name__ == "__main__":
    main()
