"""Run ONE sentiment engine over the shared FPB eval set and write predictions.

Engines
  finbert  ProsusAI/finbert classifier            (local, CPU-friendly, ~fast)
  groq     Llama-3.3-70b via Groq API             (cloud, needs GROQ_API_KEY)
  fingpt   Llama-2-7b-chat base (FinGPT app)       (heavy — run on GPU/cluster)

Workflow
  1. Decide the eval-set size ONCE (default: full 970-example FPB test split).
     The first run builds results/eval_set.csv; every engine then scores the
     same rows. To resize, re-run with --limit N --rebuild (and re-run all
     engines so they stay on identical examples).
  2. Run each engine:  python -m evaluation.run_sentiment --engine finbert
  3. Compute metrics:  python -m evaluation.metrics

Output: results/preds_<engine>.csv  (id, pred, raw)
"""

import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.data import build_eval_set, load_eval_set  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
EVAL_SET = os.path.join(RESULTS, "eval_set.csv")


def get_predictor(engine: str):
    """Return a callable text -> result dict with a 'sentiment' key."""
    if engine == "finbert":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from fetchers.sentiment_fetchers import analyze_with_finbert

        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(device).eval()
        print(f"[finbert] loaded on {device}")
        return lambda text: analyze_with_finbert(text, tok, mdl)

    if engine == "groq":
        from dotenv import load_dotenv

        from fetchers.sentiment_fetchers import analyze_with_groq, make_groq_client

        load_dotenv()
        client = make_groq_client(os.environ.get("GROQ_API_KEY", ""))
        if client is None:
            sys.exit("[groq] GROQ_API_KEY not set — add it to .env to run the Groq engine.")
        return lambda text: analyze_with_groq(client, text)

    if engine == "fingpt":
        from fetchers.sentiment_fetchers import analyze_with_fingpt
        from models.fingpt_forecast import load_fingpt_model

        tok, mdl = load_fingpt_model()
        return lambda text: analyze_with_fingpt(text, tok, mdl)

    sys.exit(f"unknown engine: {engine}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["finbert", "groq", "fingpt"])
    ap.add_argument("--split", default="test", help="FPB split (default: test)")
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the shared eval set")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    if args.rebuild or not os.path.exists(EVAL_SET):
        build_eval_set(EVAL_SET, split=args.split, limit=args.limit, seed=args.seed)
        print(f"[eval] built eval set -> {EVAL_SET}")
    rows = load_eval_set(EVAL_SET)
    print(f"[eval] scoring {len(rows)} examples with engine '{args.engine}'")

    predict = get_predictor(args.engine)
    out_path = os.path.join(RESULTS, f"preds_{args.engine}.csv")

    t0 = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred", "raw"])
        for i, row in enumerate(rows, 1):
            res = predict(row["text"])
            w.writerow([row["id"], res.get("sentiment", "unknown"), res.get("raw", "")])
            if i % 25 == 0 or i == len(rows):
                rate = (time.time() - t0) / i
                eta = rate * (len(rows) - i)
                print(f"  {i}/{len(rows)}  {rate:.2f}s/ex  eta {eta:5.0f}s")

    print(f"[eval] wrote {out_path} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
