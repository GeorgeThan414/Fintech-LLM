"""Build the forecasting eval set: (ticker, as-of date) points with cached news,
price context, and ground-truth next-week direction.

Ground truth comes from yfinance (objective). Historical news comes from Alpha
Vantage NEWS_SENTIMENT bounded to [as_of - lookback, as_of] so there is NO
look-ahead in the inputs (the realized move is computed but never shown to a model).

Each point is cached to results/forecast/eval_points.json so the (rate-limited)
news fetch happens once; every engine then scores the same inputs.

Alpha Vantage free tier: 25 req/day, 5 req/min — we pace calls 15s apart.

Run:  python -m evaluation.forecast_data            # default tickers/dates
      python -m evaluation.forecast_data --rebuild  # refetch even if cached
"""

import argparse
import json
import os
import sys
import time
from datetime import timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results", "forecast")
POINTS_PATH = os.path.join(RESULTS, "eval_points.json")

# Liquid, well-covered Dow30 names (FinGPT forecaster was trained on Dow30).
TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "WMT"]
# Mondays in 2024 — recent enough for AV news coverage, old enough to have outcomes.
AS_OF_DATES = ["2024-03-04", "2024-04-08", "2024-05-13", "2024-06-10"]

CONTEXT_DAYS = 7       # trading days of price context shown to the model
HORIZON_DAYS = 5       # trading days ahead = "next week"
LOOKBACK_NEWS = 7      # calendar days of news before as_of
REMAIN_BAND = 0.01     # |return| <= 1% -> "remain" (3-class ground truth)
MAX_HEADLINES = 10
AV_SLEEP = 15          # seconds between Alpha Vantage calls (free-tier pacing)


def _av_stamp(d, end=False):
    return d.strftime("%Y%m%dT2359") if end else d.strftime("%Y%m%dT0000")


def build_points(rebuild=False):
    import pandas as pd
    import yfinance as yf
    from dotenv import load_dotenv

    from fetchers.news_fetchers import fetch_alphavantage_news

    load_dotenv(os.path.join(ROOT, ".env"))
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not av_key:
        sys.exit("ALPHAVANTAGE_API_KEY not set in .env")

    os.makedirs(RESULTS, exist_ok=True)
    cached = {}
    if os.path.exists(POINTS_PATH) and not rebuild:
        cached = {p["id"]: p for p in json.load(open(POINTS_PATH))}
        print(f"[forecast-data] {len(cached)} points already cached")

    as_of_ts = [pd.Timestamp(d) for d in AS_OF_DATES]
    start = (min(as_of_ts) - timedelta(days=40)).strftime("%Y-%m-%d")
    end = (max(as_of_ts) + timedelta(days=30)).strftime("%Y-%m-%d")

    points = []
    for ticker in TICKERS:
        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            print(f"[forecast-data] {ticker}: no price data, skipping")
            continue
        hist.index = hist.index.tz_localize(None)
        closes = hist["Close"]

        for as_of in as_of_ts:
            point_id = f"{ticker}_{as_of.strftime('%Y-%m-%d')}"

            past = closes[closes.index <= as_of]
            future = closes[closes.index > as_of]
            if len(past) < CONTEXT_DAYS or len(future) < HORIZON_DAYS:
                print(f"[forecast-data] {point_id}: insufficient price history, skipping")
                continue

            ctx = past.tail(CONTEXT_DAYS)
            as_of_close = float(ctx.iloc[-1])
            future_close = float(future.iloc[HORIZON_DAYS - 1])
            ret = (future_close - as_of_close) / as_of_close

            direction = "up" if ret >= 0 else "down"
            if ret > REMAIN_BAND:
                three = "rise"
            elif ret < -REMAIN_BAND:
                three = "fall"
            else:
                three = "remain"

            # momentum baseline input: direction over the context window
            prev_ret = (as_of_close - float(ctx.iloc[0])) / float(ctx.iloc[0])
            momentum_prev = "up" if prev_ret >= 0 else "down"

            prices = [{"date": d.strftime("%Y-%m-%d"), "close": round(float(c), 2)}
                      for d, c in ctx.items()]

            if point_id in cached and cached[point_id].get("headlines"):
                headlines = cached[point_id]["headlines"]
                print(f"[forecast-data] {point_id}: reuse cached news ({len(headlines)})")
            else:
                tf = _av_stamp(as_of - timedelta(days=LOOKBACK_NEWS))
                tt = _av_stamp(as_of, end=True)
                print(f"[forecast-data] {point_id}: fetching AV news {tf}..{tt}")
                arts = fetch_alphavantage_news(av_key, tickers=ticker, limit=50,
                                               time_from=tf, time_to=tt)
                headlines = [a["title"] for a in arts if a.get("title")][:MAX_HEADLINES]
                time.sleep(AV_SLEEP)

            if len(headlines) < 2:
                print(f"[forecast-data] {point_id}: <2 headlines, skipping")
                continue

            points.append({
                "id": point_id,
                "ticker": ticker,
                "as_of": as_of.strftime("%Y-%m-%d"),
                "prices": prices,
                "headlines": headlines,
                "actual_return": round(ret, 4),
                "actual_direction": direction,
                "actual_3class": three,
                "momentum_prev": momentum_prev,
            })

    with open(POINTS_PATH, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=2)
    print(f"\n[forecast-data] wrote {len(points)} points -> {POINTS_PATH}")
    n_up = sum(1 for p in points if p["actual_direction"] == "up")
    print(f"[forecast-data] ground truth: {n_up} up / {len(points) - n_up} down")
    return points


def load_points():
    if not os.path.exists(POINTS_PATH):
        sys.exit(f"{POINTS_PATH} not found — run evaluation.forecast_data first")
    return json.load(open(POINTS_PATH))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="refetch news even if cached")
    build_points(rebuild=ap.parse_args().rebuild)
