"""Run ONE forecasting engine over the eval points -> results/forecast/preds_<engine>.csv

  groq    Llama-3.3-70B via Groq API  (cloud, needs GROQ_API_KEY)
  fingpt  FinGPT forecaster LoRA      (GPU/cluster — slow on CPU)

Baselines (naive / momentum / random) are computed analytically in
evaluation.forecast_metrics, so they need no run here.

  python -m evaluation.run_forecast --engine groq
  python -m evaluation.run_forecast --engine fingpt     # on the cluster
  python -m evaluation.forecast_metrics
"""

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.forecast_data import load_points  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "forecast")


def get_forecaster(engine: str):
    if engine == "groq":
        from dotenv import load_dotenv

        from models.groq_forecast import forecast_with_groq, make_groq_client

        load_dotenv(os.path.join(ROOT, ".env"))
        client = make_groq_client(os.environ.get("GROQ_API_KEY", ""))
        if client is None:
            sys.exit("[groq] GROQ_API_KEY not set — add it to .env.")
        return lambda p: forecast_with_groq(client, p["ticker"], p["headlines"], p["prices"])

    if engine == "fingpt":
        from models.fingpt_forecast import load_fingpt_model, run_forecast

        tok, mdl = load_fingpt_model()
        return lambda p: run_forecast(p["ticker"], p["headlines"], p["prices"], tok, mdl)

    sys.exit(f"unknown engine: {engine}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["groq", "fingpt"])
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between calls (default 2.1 for groq)")
    args = ap.parse_args()
    sleep_s = args.sleep if args.sleep is not None else (2.1 if args.engine == "groq" else 0.0)

    points = load_points()
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"preds_{args.engine}.csv")

    # Resume: keep valid prior predictions; only redo unknown/error points (e.g. after
    # a 429). Lets a rate-limited run be finished later without redoing everything.
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["prediction"] not in ("", "unknown") and not str(r["raw"]).lower().startswith("error"):
                    done[r["id"]] = r
        if done:
            print(f"[forecast] resuming: {len(done)} of {len(points)} already valid")

    todo = [p for p in points if p["id"] not in done]
    if not todo:
        print("[forecast] nothing to do — all points already predicted")
        return

    print(f"[forecast] scoring {len(todo)} points with '{args.engine}'")
    forecast = get_forecaster(args.engine)
    t0 = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prediction", "raw"])
        for r in done.values():
            w.writerow([r["id"], r["prediction"], r["raw"]])
        for i, p in enumerate(todo, 1):
            res = forecast(p)
            w.writerow([p["id"], res.get("prediction", "unknown"),
                        str(res.get("raw", "")).replace("\n", " ")[:2000]])
            f.flush()
            print(f"  {i}/{len(todo)} {p['id']}: {res.get('prediction')}")
            if sleep_s:
                time.sleep(sleep_s)

    print(f"[forecast] wrote {out_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
