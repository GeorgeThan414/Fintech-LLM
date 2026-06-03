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
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.data import DATASETS, build_eval_set, load_eval_set  # noqa: E402

RESULTS = os.path.join(ROOT, "results")


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


def _predict_with_retry(predict, text, tries=5):
    """Call predict(); if a generative engine returns an error/rate-limit raw,
    back off and retry so transient 429s don't become permanent 'unknown' rows."""
    delay = 5.0
    for attempt in range(tries):
        res = predict(text)
        raw = str(res.get("raw", ""))
        if not raw.lower().startswith("error:"):
            return res
        if attempt < tries - 1:
            wait = delay
            # honor an explicit retry hint if present in the error text
            m = re.search(r"try again in ([\d.]+)s", raw)
            if m:
                wait = float(m.group(1)) + 0.5
            print(f"    retry {attempt + 1}/{tries - 1} after {wait:.1f}s ({raw[:80]})")
            time.sleep(wait)
            delay *= 1.5
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True, choices=["finbert", "groq", "fingpt"])
    ap.add_argument("--dataset", default="fpb", choices=list(DATASETS),
                    help="benchmark: fpb (in-domain) or fiqa (out-of-domain)")
    ap.add_argument("--split", default=None, help="dataset split (default: test)")
    ap.add_argument("--limit", type=int, default=None, help="stratified subsample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the shared eval set")
    ap.add_argument("--sleep", type=float, default=None,
                    help="seconds between calls (default: 2.1 for groq to respect RPM, else 0)")
    args = ap.parse_args()

    # Groq free tier is ~30 req/min; pace requests unless overridden.
    sleep_s = args.sleep if args.sleep is not None else (2.1 if args.engine == "groq" else 0.0)

    out_dir = os.path.join(RESULTS, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    eval_set = os.path.join(out_dir, "eval_set.csv")
    if args.rebuild or not os.path.exists(eval_set):
        build_eval_set(eval_set, dataset=args.dataset, split=args.split,
                       limit=args.limit, seed=args.seed)
        print(f"[eval] built {args.dataset} eval set -> {eval_set}")
    rows = load_eval_set(eval_set)
    print(f"[eval] {args.dataset}: scoring {len(rows)} examples with engine '{args.engine}'")

    out_path = os.path.join(out_dir, f"preds_{args.engine}.csv")

    # Resume: keep rows already predicted with a valid (non-error/unknown) label so a
    # rate-limited run can be finished later without redoing work.
    done = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["pred"] not in ("", "unknown") and not str(r["raw"]).lower().startswith("error"):
                    done[r["id"]] = r
        if done:
            print(f"[eval] resuming: {len(done)} of {len(rows)} already done")

    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print("[eval] nothing to do — all examples already predicted")
        return

    predict = get_predictor(args.engine)
    t0 = time.time()
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "pred", "raw"])
        for r in done.values():  # carry forward completed rows
            w.writerow([r["id"], r["pred"], r["raw"]])
        for i, row in enumerate(todo, 1):
            res = _predict_with_retry(predict, row["text"])
            w.writerow([row["id"], res.get("sentiment", "unknown"), res.get("raw", "")])
            f.flush()
            if sleep_s:
                time.sleep(sleep_s)
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                eta = rate * (len(todo) - i)
                print(f"  {i}/{len(todo)}  {rate:.2f}s/ex  eta {eta:5.0f}s")

    print(f"[eval] wrote {out_path} ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
