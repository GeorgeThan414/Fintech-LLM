"""Score forecasting engines + baselines against realized next-week moves.

Reads results/forecast/eval_points.json (ground truth) and every
results/forecast/preds_<engine>.csv, then writes:
  results/forecast/forecast_metrics.csv   summary per model + baselines
  results/forecast/forecast_metrics.tex   booktabs table for the paper
  results/forecast/forecast_per_point.csv  per-point detail (each engine's call vs actual)

Primary metric = binary directional accuracy (up vs down). A model "remain" /
"unknown" is a non-committal call and counts as incorrect (so "decisiveness" — the
share of up/down calls — is reported alongside). 3-class accuracy (rise/fall/remain,
±1% band) is reported as a secondary, stricter metric.

Baselines:
  naive_rise  always "up"      (markets rise most weeks)
  momentum    last week's direction
  random      seeded coin flip

  python -m evaluation.forecast_metrics
"""

import csv
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "forecast")
POINTS = os.path.join(RESULTS, "eval_points.json")

DISPLAY = {
    "groq": "Groq (Llama-3.3-70B)",
    "fingpt": "FinGPT (Llama-2-7B)",
    "naive_rise": "Baseline: always Rise",
    "momentum": "Baseline: momentum",
    "random": "Baseline: random",
}

# model 3-class label -> binary direction
TO_BINARY = {"rise": "up", "fall": "down"}


def _binary_acc(rows, pred_key):
    """rows: list of dicts with actual_direction + pred_key (a binary 'up'/'down'/None)."""
    n = len(rows)
    correct = sum(1 for r in rows if r[pred_key] is not None and r[pred_key] == r["actual_direction"])
    committed = sum(1 for r in rows if r[pred_key] is not None)
    return correct / n, committed / n


def main():
    if not os.path.exists(POINTS):
        sys.exit(f"{POINTS} not found — run evaluation.forecast_data first")
    points = json.load(open(POINTS))
    by_id = {p["id"]: p for p in points}
    ids = [p["id"] for p in points]

    # Gather model predictions
    model_preds = {}  # engine -> {id: 3class label}
    for pf in sorted(glob.glob(os.path.join(RESULTS, "preds_*.csv"))):
        engine = os.path.basename(pf)[len("preds_"):-len(".csv")]
        with open(pf, encoding="utf-8") as f:
            model_preds[engine] = {r["id"]: r["prediction"] for r in csv.DictReader(f)}

    # Build per-engine binary calls
    rng = random.Random(42)
    random_call = {i: rng.choice(["up", "down"]) for i in ids}

    engines = []  # (key, {id: binary_or_None}, {id: 3class_or_None})
    for engine, preds in model_preds.items():
        binmap = {i: TO_BINARY.get(preds.get(i, "unknown")) for i in ids}
        threemap = {i: preds.get(i, "unknown") for i in ids}
        engines.append((engine, binmap, threemap))

    # Baselines
    engines.append(("naive_rise",
                    {i: "up" for i in ids},
                    {i: "rise" for i in ids}))
    engines.append(("momentum",
                    {i: by_id[i]["momentum_prev"] for i in ids},
                    {i: ("rise" if by_id[i]["momentum_prev"] == "up" else "fall") for i in ids}))
    engines.append(("random",
                    {i: random_call[i] for i in ids},
                    {i: ("rise" if random_call[i] == "up" else "fall") for i in ids}))

    # Score
    summary = []
    for key, binmap, threemap in engines:
        rows = [{"actual_direction": by_id[i]["actual_direction"], "b": binmap[i]} for i in ids]
        bin_acc, decisiveness = _binary_acc(rows, "b")
        three_correct = sum(1 for i in ids if threemap[i] == by_id[i]["actual_3class"])
        three_acc = three_correct / len(ids)
        summary.append({
            "engine": key,
            "display": DISPLAY.get(key, key),
            "n": len(ids),
            "dir_acc": bin_acc,
            "three_acc": three_acc,
            "decisiveness": decisiveness,
        })

    # Order: models first (by dir_acc), then baselines
    is_baseline = lambda s: s["engine"] in ("naive_rise", "momentum", "random")
    summary.sort(key=lambda s: (is_baseline(s), -s["dir_acc"]))

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "forecast_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["engine", "n", "directional_accuracy", "three_class_accuracy", "decisiveness"])
        for s in summary:
            w.writerow([s["engine"], s["n"], f"{s['dir_acc']:.4f}",
                        f"{s['three_acc']:.4f}", f"{s['decisiveness']:.4f}"])

    # Per-point detail
    with open(os.path.join(RESULTS, "forecast_per_point.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = ["id", "ticker", "as_of", "actual_return", "actual_direction", "actual_3class"]
        model_keys = list(model_preds.keys())
        w.writerow(cols + [f"{k}_pred" for k in model_keys])
        for i in ids:
            p = by_id[i]
            w.writerow([p["id"], p["ticker"], p["as_of"], p["actual_return"],
                        p["actual_direction"], p["actual_3class"]]
                       + [model_preds[k].get(i, "") for k in model_keys])

    _write_latex(summary)

    print(f"[forecast-metrics] {len(ids)} points")
    for s in summary:
        print(f"  {s['display']:28s} dir_acc={s['dir_acc']:.3f}  "
              f"3class={s['three_acc']:.3f}  decisive={s['decisiveness']:.3f}")


def _write_latex(summary):
    best_dir = max(s["dir_acc"] for s in summary)

    def fmt(v, best):
        s = f"{v:.3f}"
        return rf"\textbf{{{s}}}" if abs(v - best) < 1e-9 else s

    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Next-week directional forecasting on %d (ticker, date) points "
        r"(Dow30, 2024), scored against realized price moves. Directional accuracy is "
        r"up-vs-down; a non-committal ``Remain'' counts as incorrect (see decisiveness). "
        r"Best model directional accuracy in \textbf{bold}.}" % summary[0]["n"],
        r"  \label{tab:forecast}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Engine & Dir. Acc. & 3-class Acc. & Decisiveness \\",
        r"    \midrule",
    ]
    for idx, s in enumerate(summary):
        if s["engine"] in ("naive_rise", "momentum", "random") and idx > 0 and \
                summary[idx - 1]["engine"] not in ("naive_rise", "momentum", "random"):
            lines.append(r"    \midrule")
        lines.append("    %s & %s & %.3f & %.2f \\\\" % (
            s["display"], fmt(s["dir_acc"], best_dir), s["three_acc"], s["decisiveness"]))
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    with open(os.path.join(RESULTS, "forecast_metrics.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
