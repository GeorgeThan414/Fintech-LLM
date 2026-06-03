"""Generate publication figures (PDF, vector) from the metrics CSVs.

Outputs to results/figures/:
  accuracy_macrof1.pdf    grouped bar chart, accuracy + macro-F1 per engine
  confusion_<engine>.pdf  confusion-matrix heatmap per engine

Run after evaluation.metrics:
  python -m evaluation.figures
"""

import csv
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DISPLAY = {
    "finbert": "FinBERT",
    "groq": "Groq\n(Llama-3.3-70B)",
    "fingpt": "FinGPT\n(Llama-2-7B)",
}


def _read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.reader(f))


def accuracy_bar():
    path = os.path.join(RESULTS, "sentiment_metrics.csv")
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
    out = os.path.join(FIGDIR, "accuracy_macrof1.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def confusion_heatmaps():
    for path in sorted(glob.glob(os.path.join(RESULTS, "confusion_*.csv"))):
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
        out = os.path.join(FIGDIR, f"confusion_{engine}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"wrote {out}")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    accuracy_bar()
    confusion_heatmaps()


if __name__ == "__main__":
    main()
