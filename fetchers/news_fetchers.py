"""
News fetchers — multiple sources.

Sources:
- GNews: ticker-specific, free (10/call, 100/day)
- yfinance: ticker-specific, free, unlimited (~10-30 articles per ticker)
- Alpha Vantage NEWS_SENTIMENT: bulk, includes built-in sentiment scores (1000/call, 25/day)
"""

import os
import re
from typing import List, Dict, Optional
from datetime import datetime, timezone

import requests

GNEWS_ENDPOINT = "https://gnews.io/api/v4/search"
ALPHAVANTAGE_ENDPOINT = "https://www.alphavantage.co/query"


def _clean_query(q: str) -> str:
    """GNews chokes on periods, ampersands, and other punctuation. Strip them."""
    cleaned = re.sub(r"[.&'\"]", "", q)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

MACRO_QUERIES = {
    "Federal Reserve": "Federal Reserve OR FOMC OR rate hike",
    "Inflation": "inflation OR CPI OR PPI",
    "Geopolitics": "war OR sanctions OR conflict OR geopolitical",
    "China Markets": "China economy OR Chinese stocks OR PBoC",
    "Oil & Energy": "oil prices OR OPEC OR energy crisis",
    "Tech Sector": "tech stocks OR semiconductor OR AI chips",
    "Crypto": "bitcoin OR ethereum OR crypto regulation",
    "Banking": "bank crisis OR banking sector OR credit",
}


def fetch_news_for_ticker(
    company_name: str,
    api_key: str,
    max_articles: int = 1000,
) -> List[Dict]:
    """
    Fetch latest news for a specific company/ticker.
    Returns list of dicts: {title, description, source, url, published_at}
    """
    if not api_key:
        return []

    params = {
        "q": _clean_query(company_name),
        "lang": "en",
        "max": max_articles,
        "apikey": api_key,
        "sortby": "publishedAt",
    }
    try:
        r = requests.get(GNEWS_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [_normalize(a) for a in articles]
    except Exception as e:
        print(f"GNews error for {company_name}: {e}")
        return []


def fetch_macro_news(
    api_key: str,
    category: str = "Federal Reserve",
    max_articles: int = 1000,
) -> List[Dict]:
    """
    Fetch macro/geopolitics/economy news by category.
    Categories defined in MACRO_QUERIES dict.
    """
    query = MACRO_QUERIES.get(category, category)
    if not api_key:
        return []

    params = {
        "q": _clean_query(query),
        "lang": "en",
        "max": max_articles,
        "apikey": api_key,
        "sortby": "publishedAt",
    }
    try:
        r = requests.get(GNEWS_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        return [_normalize(a) for a in r.json().get("articles", [])]
    except Exception as e:
        print(f"GNews error for {category}: {e}")
        return []


def fetch_news_custom_query(
    query: str,
    api_key: str,
    max_articles: int = 1000,
) -> List[Dict]:
    """Fetch news for an arbitrary query string."""
    if not api_key or not query.strip():
        return []
    params = {
        "q": _clean_query(query),
        "lang": "en",
        "max": max_articles,
        "apikey": api_key,
        "sortby": "publishedAt",
    }
    try:
        r = requests.get(GNEWS_ENDPOINT, params=params, timeout=30)
        r.raise_for_status()
        return [_normalize(a) for a in r.json().get("articles", [])]
    except Exception as e:
        print(f"GNews error: {e}")
        return []


def _normalize(a: Dict) -> Dict:
    return {
        "title": a.get("title", ""),
        "description": a.get("description", ""),
        "content": a.get("content", ""),
        "source": (a.get("source") or {}).get("name", ""),
        "url": a.get("url", ""),
        "published_at": a.get("publishedAt", ""),
        "image": a.get("image", ""),
    }


# ── yfinance news (unlimited, ticker-specific, no sentiment) ──────────────────

def fetch_yfinance_news(symbol: str, max_articles: int = 20) -> List[Dict]:
    """
    Fetch news from yfinance for a specific ticker. Free, unlimited, no API key.
    Returns up to ~20-30 articles per ticker.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception as e:
        print(f"yfinance news error for {symbol}: {e}")
        return []

    out = []
    for item in raw[:max_articles]:
        # yfinance has both old-format (flat) and new-format (nested under 'content')
        content = item.get("content") or item
        title = content.get("title") or item.get("title", "")
        url = (content.get("clickThroughUrl") or {}).get("url") or content.get("canonicalUrl", {}).get("url") or item.get("link", "")
        provider = (content.get("provider") or {}).get("displayName") or item.get("publisher", "")
        pub = content.get("pubDate") or item.get("providerPublishTime", "")

        if isinstance(pub, (int, float)):
            pub = datetime.fromtimestamp(pub, tz=timezone.utc).isoformat()

        out.append({
            "title": title,
            "description": content.get("summary", ""),
            "content": "",
            "source": provider,
            "url": url,
            "published_at": str(pub),
            "image": "",
        })
    return out


# ── Alpha Vantage NEWS_SENTIMENT (bulk + built-in sentiment + topics) ─────────

# Topic codes accepted by Alpha Vantage NEWS_SENTIMENT endpoint
ALPHAVANTAGE_TOPICS = {
    "All": None,
    "Blockchain": "blockchain",
    "Earnings": "earnings",
    "IPO": "ipo",
    "Mergers & Acquisitions": "mergers_and_acquisitions",
    "Financial Markets": "financial_markets",
    "Economy - Fiscal Policy": "economy_fiscal",
    "Economy - Monetary Policy": "economy_monetary",
    "Economy - Macro": "economy_macro",
    "Energy & Transportation": "energy_transportation",
    "Finance": "finance",
    "Life Sciences": "life_sciences",
    "Manufacturing": "manufacturing",
    "Real Estate": "real_estate",
    "Retail & Wholesale": "retail_wholesale",
    "Technology": "technology",
}


def fetch_alphavantage_news(
    api_key: str,
    tickers: Optional[str] = None,
    topic: Optional[str] = None,
    limit: int = 200,
    time_from: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch up to 1000 articles per call from Alpha Vantage NEWS_SENTIMENT.
    Sentiment + topic tags are pre-computed by Alpha Vantage.

    Args:
        api_key: Alpha Vantage API key
        tickers: comma-separated tickers (e.g. "AAPL,MSFT") — optional
        topic: one of ALPHAVANTAGE_TOPICS values — optional
        limit: max articles to return (1-1000)
        time_from: YYYYMMDDTHHMM format — optional

    Returns: list of normalized article dicts with sentiment fields.
    """
    if not api_key:
        return []

    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": api_key,
        "limit": min(limit, 1000),
        "sort": "LATEST",
    }
    if tickers:
        params["tickers"] = tickers
    if topic:
        params["topics"] = topic
    if time_from:
        params["time_from"] = time_from

    try:
        r = requests.get(ALPHAVANTAGE_ENDPOINT, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Alpha Vantage returns errors as string in different keys
        if "Information" in data and "feed" not in data:
            print(f"Alpha Vantage error: {data['Information']}")
            return []
        if "Note" in data:
            print(f"Alpha Vantage rate limit: {data['Note']}")
            return []

        feed = data.get("feed", [])
        return [_normalize_av(a) for a in feed]
    except Exception as e:
        print(f"Alpha Vantage error: {e}")
        return []


def _normalize_av(a: Dict) -> Dict:
    """Normalize Alpha Vantage article. Maps Bearish/Bullish to negative/positive/neutral."""
    score = float(a.get("overall_sentiment_score", 0))
    label_av = a.get("overall_sentiment_label", "Neutral")

    # Map Alpha Vantage 5-class to 3-class
    if "Bearish" in label_av:
        sentiment = "negative"
    elif "Bullish" in label_av:
        sentiment = "positive"
    else:
        sentiment = "neutral"

    # Derive impact from sentiment score magnitude
    abs_score = abs(score)
    if abs_score >= 0.35:
        impact = "high"
    elif abs_score >= 0.15:
        impact = "medium"
    else:
        impact = "low"

    # Top topic by relevance
    topics = a.get("topics", [])
    top_topic = "General"
    if topics:
        top = max(topics, key=lambda t: float(t.get("relevance_score", 0)))
        top_topic = top.get("topic", "General")

    # Per-ticker sentiment (first one if any)
    ticker_sent = a.get("ticker_sentiment", [])
    tickers = [t.get("ticker") for t in ticker_sent[:5]] if ticker_sent else []

    # Format published_at
    pub = a.get("time_published", "")
    if len(pub) == 15:  # YYYYMMDDTHHMMSS
        try:
            pub = datetime.strptime(pub, "%Y%m%dT%H%M%S").isoformat()
        except Exception:
            pass

    return {
        "title": a.get("title", ""),
        "description": a.get("summary", ""),
        "summary": a.get("summary", ""),
        "source": a.get("source", ""),
        "url": a.get("url", ""),
        "published_at": pub,
        "image": a.get("banner_image", ""),
        "sentiment_score": score,
        "sentiment_label_av": label_av,
        "sentiment": sentiment,
        "impact": impact,
        "topic": top_topic,
        "all_topics": [t.get("topic") for t in topics],
        "tickers": tickers,
    }
