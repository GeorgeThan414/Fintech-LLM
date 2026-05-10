"""
Live stock data via yfinance.
No API key required.
"""

from typing import Optional
import pandas as pd
import yfinance as yf


def get_price_history(symbol: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns OHLCV DataFrame with columns: date, open, high, low, close, volume.
    period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    interval: 1m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=False)

    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    df = df.reset_index()
    date_col = "Date" if "Date" in df.columns else "Datetime"
    df = df.rename(columns={
        date_col: "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    })
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_current_price(symbol: str) -> Optional[float]:
    """Latest live price (or previous close if market closed)."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        price = info.get("last_price") or info.get("previous_close")
        return float(price) if price else None
    except Exception:
        try:
            df = get_price_history(symbol, period="5d")
            return float(df["close"].iloc[-1]) if not df.empty else None
        except Exception:
            return None


def get_company_info(symbol: str) -> dict:
    """Company metadata: name, sector, industry, market cap, etc."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap"),
            "currency": info.get("currency", "USD"),
            "website": info.get("website", ""),
            "summary": info.get("longBusinessSummary", ""),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception:
        return {"name": symbol, "sector": "N/A", "industry": "N/A", "currency": "USD"}


def get_intraday(symbol: str, days: int = 1) -> pd.DataFrame:
    """Intraday minute-level data for last N days (1-7)."""
    period = f"{min(days, 7)}d"
    return get_price_history(symbol, period=period, interval="5m")
