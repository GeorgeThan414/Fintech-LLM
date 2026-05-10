"""
Fetch and cache S&P 500 constituent tickers from Wikipedia.
"""

import io
import os
from pathlib import Path

import pandas as pd
import requests

CACHE_FILE = Path(__file__).parent / "sp500_tickers.csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_tickers(use_cache: bool = True) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: Symbol, Security, Sector, Sub_Industry.
    Uses a local CSV cache; refreshes from Wikipedia if missing or stale (>7 days).
    """
    if use_cache and CACHE_FILE.exists():
        age_days = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(CACHE_FILE), unit="s")).days
        if age_days < 7:
            return pd.read_csv(CACHE_FILE)

    try:
        resp = requests.get(WIKI_URL, headers={"User-Agent": "FintechLLM/1.0"}, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        df = tables[0][["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].copy()
        df.columns = ["Symbol", "Security", "Sector", "Sub_Industry"]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        df.to_csv(CACHE_FILE, index=False)
        return df
    except Exception:
        if CACHE_FILE.exists():
            return pd.read_csv(CACHE_FILE)
        raise


if __name__ == "__main__":
    df = fetch_sp500_tickers(use_cache=False)
    print(f"Fetched {len(df)} S&P 500 tickers")
    print(df.head(10))
