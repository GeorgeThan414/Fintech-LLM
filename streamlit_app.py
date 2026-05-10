"""
Fintech-LLM Dashboard — Streamlit
Live stock data + dual-engine sentiment + FinGPT forecasting + Articles Explorer.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from data.sp500_tickers import fetch_sp500_tickers
from fetchers.stock_fetchers import (
    get_price_history,
    get_current_price,
    get_company_info,
)
from fetchers.news_fetchers import (
    fetch_yfinance_news,
    fetch_alphavantage_news,
)
from fetchers.sentiment_fetchers import (
    analyze_with_groq,
    analyze_with_fingpt,
    make_groq_client,
)

load_dotenv()
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fintech-LLM",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light/physical theme
st.markdown("""
<style>
    .stApp {
        background: #f8fafc;
        color: #1e293b;
    }
    .main .block-container {
        padding-top: 2rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 { color: #0f172a !important; }
    p, label, .stMarkdown { color: #334155 !important; }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 500; }
    [data-testid="stMetricValue"] { color: #0f172a !important; font-weight: 700; }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stMarkdown h1 {
        color: #0f172a !important;
        font-size: 1.6rem;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #475569 !important;
    }

    .info-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 0 6px 6px 0;
        padding: 14px 18px;
        margin: 12px 0 20px 0;
        color: #1e40af;
        font-size: 0.92rem;
    }
    .step-box {
        background: #fefce8;
        border-left: 4px solid #ca8a04;
        border-radius: 0 6px 6px 0;
        padding: 12px 18px;
        margin: 10px 0;
        color: #713f12;
        font-size: 0.9rem;
    }
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
        height: 100%;
    }
    .feature-card h3 { margin-top: 0; color: #0f172a !important; }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }
    .news-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }
    .news-card a { color: #1e3a8a; text-decoration: none; font-weight: 600; }
    .news-card a:hover { text-decoration: underline; }
    .news-meta { color: #64748b; font-size: 0.8rem; margin-top: 4px; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-positive { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
    .badge-negative { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-neutral  { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }
    .badge-high     { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-medium   { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .badge-low      { background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; }

    .stButton > button {
        background: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stButton > button:hover { background: #1d4ed8; }

    /* Code blocks / st.text output: ensure visible on light theme */
    [data-testid="stCodeBlock"], pre, code, [data-testid="stMarkdownContainer"] pre {
        background: #f1f5f9 !important;
        color: #1e293b !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 6px !important;
    }
    [data-testid="stCodeBlock"] code, [data-testid="stCodeBlock"] span,
    pre code, pre span {
        color: #1e293b !important;
        background: transparent !important;
    }
    /* Streamlit st.text() / monospace output */
    [data-testid="stText"], .element-container pre {
        color: #1e293b !important;
    }
    /* Expander content text */
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] .stMarkdown,
    [data-testid="stExpander"] code {
        color: #1e293b !important;
    }

    /* ── Blue accent for ALL selection widgets ───────────────────────────── */

    /* Selectbox closed state */
    [data-baseweb="select"] > div {
        background: #eff6ff !important;
        border: 1.5px solid #2563eb !important;
        border-radius: 6px !important;
    }
    [data-baseweb="select"] > div:hover {
        border-color: #1d4ed8 !important;
        background: #dbeafe !important;
    }
    [data-baseweb="select"] svg { color: #2563eb !important; fill: #2563eb !important; }

    /* Selected value text - force black on every possible inner element */
    [data-baseweb="select"] *,
    [data-baseweb="select"] div,
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="select"] [class*="Value"],
    [data-baseweb="select"] [class*="value"],
    [data-baseweb="select"] [class*="SingleValue"],
    [data-baseweb="select"] [class*="singleValue"],
    [data-baseweb="select"] [data-baseweb="select-control"] {
        color: #0f172a !important;
        font-weight: 500 !important;
        -webkit-text-fill-color: #0f172a !important;
        opacity: 1 !important;
    }
    /* Re-style the SVG arrow back to blue (the * rule above hits it too) */
    [data-baseweb="select"] svg,
    [data-baseweb="select"] svg * {
        color: #2563eb !important;
        fill: #2563eb !important;
        -webkit-text-fill-color: #2563eb !important;
    }

    /* Selectbox dropdown menu — force white background */
    [data-baseweb="popover"],
    [data-baseweb="popover"] > div,
    [data-baseweb="popover"] ul,
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="menu"] {
        background: #ffffff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 6px !important;
    }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] li {
        background: #ffffff !important;
        color: #1e293b !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background: #dbeafe !important;
        color: #1e3a8a !important;
    }
    [data-baseweb="popover"] [aria-selected="true"],
    [data-baseweb="menu"] li[aria-selected="true"] {
        background: #2563eb !important;
        color: white !important;
    }

    /* Number input + Text input */
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-baseweb="input"] input {
        background: #eff6ff !important;
        border: 1.5px solid #2563eb !important;
        border-radius: 6px !important;
        color: #0f172a !important;
        font-weight: 500;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus,
    [data-baseweb="input"] input:focus {
        border-color: #1d4ed8 !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
    }
    [data-testid="stNumberInput"] button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
    }
    [data-testid="stNumberInput"] button:hover {
        background: #1d4ed8 !important;
    }

    /* Sidebar radio (navigation) */
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 8px 12px;
        margin: 4px 0;
        transition: all 0.15s;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: #dbeafe;
        border-color: #2563eb;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] *,
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
        color: white !important;
    }

    /* Multiselect tags */
    [data-baseweb="tag"] {
        background: #2563eb !important;
        color: white !important;
        border-radius: 4px !important;
    }
    [data-baseweb="tag"] svg { color: white !important; fill: white !important; }

    /* Slider */
    [data-testid="stSlider"] [role="slider"] {
        background: #2563eb !important;
        border-color: #1d4ed8 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    font=dict(color="#1e3a8a", family="system-ui", size=12),
    xaxis=dict(
        gridcolor="#cbd5e1",
        linecolor="#1e3a8a",
        tickcolor="#1e3a8a",
        tickfont=dict(color="#1e3a8a"),
        title=dict(font=dict(color="#1e3a8a")),
        zerolinecolor="#cbd5e1",
    ),
    yaxis=dict(
        gridcolor="#cbd5e1",
        linecolor="#1e3a8a",
        tickcolor="#1e3a8a",
        tickfont=dict(color="#1e3a8a"),
        title=dict(font=dict(color="#1e3a8a")),
        zerolinecolor="#cbd5e1",
    ),
    legend=dict(font=dict(color="#1e3a8a")),
)

SENTIMENT_BADGE = {
    "positive": "badge-positive",
    "negative": "badge-negative",
    "neutral": "badge-neutral",
    "unknown": "badge-neutral",
}
IMPACT_BADGE = {"high": "badge-high", "medium": "badge-medium", "low": "badge-low"}

SENTIMENT_COLORS = {
    "positive": "#16a34a", "negative": "#dc2626",
    "neutral": "#94a3b8", "unknown": "#cbd5e1",
}
IMPACT_COLORS = {"high": "#dc2626", "medium": "#ca8a04", "low": "#94a3b8"}


def styled(fig, height=400):
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=40, b=0), **PLOTLY_LAYOUT)
    return fig


def info_box(html: str):
    st.markdown(f'<div class="info-box">{html}</div>', unsafe_allow_html=True)


def step_box(html: str):
    st.markdown(f'<div class="step-box">{html}</div>', unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_data(ttl=86400 * 7)
def cached_sp500():
    return fetch_sp500_tickers()


@st.cache_data(ttl=300)
def cached_price_history(symbol: str, period: str):
    return get_price_history(symbol, period=period)


@st.cache_data(ttl=600)
def cached_current_price(symbol: str):
    return get_current_price(symbol)


@st.cache_data(ttl=3600)
def cached_company_info(symbol: str):
    return get_company_info(symbol)


@st.cache_data(ttl=600)
def cached_yfinance_news(symbol: str, max_n: int):
    return fetch_yfinance_news(symbol, max_articles=max_n)


@st.cache_data(ttl=900)
def cached_av_news(tickers: str, topic: str, limit: int):
    return fetch_alphavantage_news(
        ALPHAVANTAGE_API_KEY,
        tickers=tickers if tickers else None,
        topic=topic if topic else None,
        limit=limit,
    )


@st.cache_resource
def cached_groq_client():
    return make_groq_client(GROQ_API_KEY)


@st.cache_resource(show_spinner="Loading FinGPT model (first run downloads ~14GB)...")
def cached_fingpt():
    from models.fingpt_forecast import load_fingpt_model
    return load_fingpt_model()


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Fintech-LLM")
st.sidebar.caption("Live stock intelligence powered by LLMs")

page = st.sidebar.radio(
    "Navigate",
    [
        "Home",
        "Stock Overview",
        "News & Sentiment",
        "FinGPT Forecast",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Data sources**")
st.sidebar.caption("• Yahoo Finance (prices + news)")
st.sidebar.caption("• Alpha Vantage (bulk news + sentiment)")
st.sidebar.caption("• Groq Llama 3.3 70B (cloud LLM)")
st.sidebar.caption("• FinGPT Llama-2-7b (local LLM)")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 0: Home / Welcome
# ══════════════════════════════════════════════════════════════════════════════

if page == "Home":
    st.title("Welcome to Fintech-LLM Stock Market Sentiment and Forecasting Application")
    st.markdown(
        "A live stock intelligence platform combining **real-time market data**, "
        "**LLM-powered news sentiment**, and **FinGPT forecasting** in one dashboard."
    )

    st.subheader("Available Options ")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>Stock Overview</h3>
            <p>Live prices, candlestick chart, and company snapshot for any S&P 500 stock.
            Data refreshes from Yahoo Finance every 5 minutes.</p>
            <p><b>Use it for:</b> quick price checks, technical view, company fundamentals.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>News & Sentiment</h3>
            <p>Latest headlines for the ticker you choose, scored for sentiment
            and impact by <b>Groq</b> (cloud) and/or <b>FinGPT</b> (local model).
            Compare both engines side-by-side.</p>
            <p><b>Use it for:</b> understanding how news is moving a single stock.</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon"></div>
            <h3>FinGPT Forecast</h3>
            <p>Run the FinGPT forecaster (Llama-2 + LoRA fine-tuned on Dow30) on any
            S&P 500 stock. Generates a structured analysis: Positive Developments,
            Concerns, Prediction (Rise / Fall / Remain), Analysis.</p>
            <p><b>Use it for:</b> deeper LLM-driven directional outlook for next week.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("How it works")
    st.markdown("""
    - **Prices:** pulled live from Yahoo Finance via the `yfinance` library
    - **News (per ticker):** pulled live from Yahoo Finance, no rate limits
    - **News (bulk + sentiment):** pulled from Alpha Vantage's NEWS_SENTIMENT endpoint, which returns up to 1000 articles per call with pre-computed sentiment scores and topic tags
    - **Sentiment LLM:** Groq's hosted Llama 3.3 70B (instant) or local Llama-2-7b-chat
    - **Forecasting LLM:** local Llama-2-7b-chat with the FinGPT forecaster LoRA enabled
    """)

    st.caption("⚠️ For educational and analytical purposes only. Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Stock Overview
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Stock Overview":
    st.title(" Stock Overview")
    info_box(
        "Pick any S&P 500 ticker to see live price metrics, an OHLC candlestick chart, "
        "and company information. All data is fetched on demand from Yahoo Finance."
    )
    step_box(
        "<b>How to use:</b> Choose a ticker from the dropdown, pick a time period, "
        "and the chart updates automatically. Open the <i>About</i> expander for company details."
    )

    sp500 = cached_sp500()
    col_pick, col_period = st.columns([3, 1])
    with col_pick:
        ticker_options = sp500.apply(lambda r: f"{r['Symbol']} — {r['Security']}", axis=1).tolist()
        selected = st.selectbox("Select stock", ticker_options, index=0)
        symbol = selected.split(" — ")[0]
    with col_period:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=1)

    info = cached_company_info(symbol)
    df = cached_price_history(symbol, period)
    current = cached_current_price(symbol)

    if df.empty:
        st.error(f"No data for {symbol}")
        st.stop()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    change = float(latest["close"]) - float(prev["close"])
    pct = (change / float(prev["close"])) * 100 if prev["close"] else 0
    currency = info.get("currency", "USD")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"{symbol} (live)", f"{currency} {current:.2f}" if current else "N/A",
              f"{change:+.2f} ({pct:+.2f}%)")
    c2.metric("Open", f"{currency} {float(latest['open']):.2f}")
    c3.metric("High", f"{currency} {float(latest['high']):.2f}")
    c4.metric("Low", f"{currency} {float(latest['low']):.2f}")
    c5.metric("Volume", f"{int(latest['volume']):,}")

    with st.expander(f"About {info.get('name', symbol)}", expanded=False):
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        cc1.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        cc2.markdown(f"**Market Cap:** {info.get('market_cap', 'N/A'):,}" if info.get('market_cap') else "**Market Cap:** N/A")
        cc2.markdown(f"**P/E Ratio:** {info.get('pe_ratio', 'N/A')}")
        cc3.markdown(f"**52W High:** {info.get('52w_high', 'N/A')}")
        cc3.markdown(f"**52W Low:** {info.get('52w_low', 'N/A')}")
        if info.get("summary"):
            st.markdown(info["summary"][:500] + "...")

    st.subheader(f"{symbol} Price Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    )])
    fig.update_layout(xaxis_rangeslider_visible=False, yaxis_title=currency)
    st.plotly_chart(styled(fig, 500), width="stretch")

    st.subheader("Volume")
    fig_vol = px.bar(df, x="date", y="volume", labels={"volume": "Volume", "date": "Date"})
    fig_vol.update_traces(marker_color="#3b82f6")
    st.plotly_chart(styled(fig_vol, 250), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: News & Sentiment (yfinance + dual engine)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "News & Sentiment":
    st.title("News & Sentiment")
    info_box(
        "Pull the latest headlines for any S&P 500 ticker (via Yahoo Finance, unlimited) "
        "and run sentiment analysis with <b>Groq</b> (fast cloud), <b>FinGPT</b> "
        "(local model), or <b>both</b> to compare."
    )
    step_box(
        "<b>How to use:</b> 1) Pick a ticker. 2) Choose how many articles. 3) Pick the engine. "
        "4) Click <b>Fetch & Analyze</b>. The first FinGPT call takes ~30s; subsequent calls are instant."
    )

    sp500 = cached_sp500()
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
    with c1:
        ticker_options = sp500.apply(lambda r: f"{r['Symbol']} — {r['Security']}", axis=1).tolist()
        selected = st.selectbox("Select stock", ticker_options, index=0, key="news_select")
        symbol = selected.split(" — ")[0]
    with c2:
        source = st.selectbox("Source", ["Yahoo Finance", "Alpha Vantage"],
                              help="Yahoo Finance: unlimited daily but ~10-20 articles per ticker. "
                                   "Alpha Vantage: up to 1000 per ticker but 25 calls/day.")
    with c3:
        max_articles = st.number_input("Articles", min_value=3, max_value=1000, value=10)
    with c4:
        engine = st.selectbox("Engine", ["Groq (fast)", "FinGPT (local)", "Both (compare)"])

    fetch_clicked = st.button("Fetch & Analyze", type="primary")

    if fetch_clicked:
        with st.spinner(f"Fetching news for {symbol} from {source}..."):
            if source == "Alpha Vantage":
                if not ALPHAVANTAGE_API_KEY:
                    st.error("ALPHAVANTAGE_API_KEY missing in .env")
                    st.stop()
                articles = cached_av_news(symbol, "", max_articles)
            else:
                articles = cached_yfinance_news(symbol, max_articles)

        if not articles:
            st.warning(f"No articles found for {symbol}.")
            st.stop()

        # Enforce exact count
        articles = articles[:max_articles]

        use_groq = "Groq" in engine or "Both" in engine
        use_fingpt = "FinGPT" in engine or "Both" in engine

        groq_client = cached_groq_client() if use_groq else None
        if use_groq and not groq_client:
            st.error("Groq API key missing or invalid. Set GROQ_API_KEY in .env")
            st.stop()

        fingpt_pair = cached_fingpt() if use_fingpt else None

        st.subheader(f"Analysis ({len(articles)} articles)")
        results = []
        progress = st.progress(0)
        for i, art in enumerate(articles):
            headline = art["title"]
            row = {"title": headline, "source": art["source"], "url": art["url"]}

            if use_groq:
                gr = analyze_with_groq(groq_client, headline)
                row["groq_sentiment"] = gr["sentiment"]
                row["groq_impact"] = gr["impact"]
                row["groq_reason"] = gr.get("reason", "")
                row["groq_raw"] = gr["raw"]

            if use_fingpt:
                tok, mdl = fingpt_pair
                fr = analyze_with_fingpt(headline, tok, mdl)
                row["fingpt_sentiment"] = fr["sentiment"]
                row["fingpt_impact"] = fr["impact"]
                row["fingpt_reason"] = fr.get("reason", "")
                row["fingpt_raw"] = fr["raw"]

            results.append(row)
            progress.progress((i + 1) / len(articles))
        progress.empty()

        # Persist to session state so navigation away & back keeps results
        st.session_state["news_results"] = {
            "results": results,
            "use_groq": use_groq,
            "use_fingpt": use_fingpt,
            "symbol": symbol,
            "source": source,
            "requested": max_articles,
        }

    # ── Render from session_state (so results survive page navigation) ────────
    if "news_results" in st.session_state:
        cache = st.session_state["news_results"]
        results = cache["results"]
        use_groq = cache["use_groq"]
        use_fingpt = cache["use_fingpt"]

        st.caption(f"Showing cached analysis for **{cache['symbol']}** "
                   f"({len(results)} articles, source: {cache['source']}). "
                   f"Click **Fetch & Analyze** above to refresh.")

        if len(results) < cache["requested"]:
            st.info(f"Got {len(results)} articles back (you asked for {cache['requested']}). "
                    f"{'Yahoo Finance has a hard limit of ~10-20 per ticker.' if cache['source'] == 'Yahoo Finance' else 'Alpha Vantage returned all available for this ticker.'}")

        # Total articles always shown
        st.metric("Total Articles Analyzed", len(results))

        # Per-engine breakdown
        if use_groq:
            st.markdown("##### Groq (Llama 3.3 70B)")
            g1, g2, g3, g4 = st.columns(4)
            g1.metric("Positive", sum(1 for r in results if r.get("groq_sentiment") == "positive"))
            g2.metric("Negative", sum(1 for r in results if r.get("groq_sentiment") == "negative"))
            g3.metric("Neutral", sum(1 for r in results if r.get("groq_sentiment") == "neutral"))
            g4.metric("High Impact", sum(1 for r in results if r.get("groq_impact") == "high"))

        if use_fingpt:
            st.markdown("##### FinGPT (Llama-2-7b-chat, LoRA off)")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Positive", sum(1 for r in results if r.get("fingpt_sentiment") == "positive"))
            f2.metric("Negative", sum(1 for r in results if r.get("fingpt_sentiment") == "negative"))
            f3.metric("Neutral", sum(1 for r in results if r.get("fingpt_sentiment") == "neutral"))
            f4.metric("High Impact", sum(1 for r in results if r.get("fingpt_impact") == "high"))

        if use_groq and use_fingpt:
            d1, d2 = st.columns(2)
        else:
            d1 = st.container()
            d2 = None

        if use_groq:
            counts = pd.Series([r["groq_sentiment"] for r in results]).value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title="Groq Sentiment",
                         color=counts.index, color_discrete_map=SENTIMENT_COLORS)
            with d1:
                st.plotly_chart(styled(fig, 280), width="stretch")
        if use_fingpt:
            counts = pd.Series([r["fingpt_sentiment"] for r in results]).value_counts()
            fig = px.pie(values=counts.values, names=counts.index,
                         title="FinGPT Sentiment (LoRA off)",
                         color=counts.index, color_discrete_map=SENTIMENT_COLORS)
            target = d2 if d2 is not None else d1
            with target:
                st.plotly_chart(styled(fig, 280), width="stretch")

        if use_groq and use_fingpt:
            agree = sum(1 for r in results if r["groq_sentiment"] == r["fingpt_sentiment"])
            agree_pct = (agree / len(results)) * 100 if results else 0
            st.metric("Engine Agreement", f"{agree}/{len(results)} ({agree_pct:.0f}%)")

        high_impact = [r for r in results
                       if r.get("groq_impact") == "high" or r.get("fingpt_impact") == "high"]
        if high_impact:
            st.subheader(f"High-Impact Headlines ({len(high_impact)})")
            for r in high_impact:
                st.markdown(f"**[{r['title']}]({r['url']})** — *{r['source']}*")

        st.divider()
        st.subheader("Per-Article Detail")
        for r in results:
            st.markdown(f"**[{r['title']}]({r['url']})** — *{r['source']}*")
            cols = st.columns(2 if (use_groq and use_fingpt) else 1)

            if use_groq and "groq_sentiment" in r:
                with cols[0]:
                    s_class = SENTIMENT_BADGE.get(r["groq_sentiment"], "badge-neutral")
                    i_class = IMPACT_BADGE.get(r["groq_impact"], "badge-medium")
                    st.markdown(f'<span class="badge {s_class}">Groq: {r["groq_sentiment"]}</span> '
                                f'<span class="badge {i_class}">Impact: {r["groq_impact"]}</span>',
                                unsafe_allow_html=True)
                    reason = r.get("groq_reason", "").strip()
                    if reason:
                        st.markdown(f'<div style="color:#475569;font-size:0.88rem;'
                                    f'margin:6px 0 4px 0;font-style:italic;">'
                                    f'{reason}</div>',
                                    unsafe_allow_html=True)

            if use_fingpt and "fingpt_sentiment" in r:
                target_col = cols[1] if (use_groq and use_fingpt) else cols[0]
                with target_col:
                    s_class = SENTIMENT_BADGE.get(r["fingpt_sentiment"], "badge-neutral")
                    i_class = IMPACT_BADGE.get(r["fingpt_impact"], "badge-medium")
                    st.markdown(f'<span class="badge {s_class}">FinGPT: {r["fingpt_sentiment"]}</span> '
                                f'<span class="badge {i_class}">Impact: {r["fingpt_impact"]}</span>',
                                unsafe_allow_html=True)
                    reason = r.get("fingpt_reason", "").strip()
                    if reason:
                        st.markdown(f'<div style="color:#475569;font-size:0.88rem;'
                                    f'margin:6px 0 4px 0;font-style:italic;">'
                                    f'{reason}</div>',
                                    unsafe_allow_html=True)

            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: FinGPT Forecast
# ══════════════════════════════════════════════════════════════════════════════

elif page == "FinGPT Forecast":
    st.title("FinGPT Forecast")
    info_box(
        "Run the <b>FinGPT forecaster</b> (Llama-2-7b-chat + LoRA fine-tuned on Dow30) "
        "on any S&P 500 stock. Combines recent news + 7-day price history to predict "
        "next-week direction (Rise / Fall / Remain) with structured analysis."
    )
    step_box(
        "<b>How to use:</b> " \
        "1) Pick a ticker. " \
        "2) Pick how many news articles to feed in. "
        "3) Click <b>Run FinGPT Forecast</b>. The first run loads the model into VRAM (~30s); "
        "subsequent runs are fast. Output is structured: Positive Developments, Concerns, "
        "Prediction, Analysis."
    )

    sp500 = cached_sp500()
    c1, c2 = st.columns([3, 1])
    with c1:
        ticker_options = sp500.apply(lambda r: f"{r['Symbol']} — {r['Security']}", axis=1).tolist()
        selected = st.selectbox("Select stock", ticker_options, index=0, key="forecast_select")
        symbol = selected.split(" — ")[0]
    with c2:
        n_news = st.number_input("News articles", min_value=3, max_value=20, value=8)

    if st.button("Run FinGPT Forecast", type="primary"):
        with st.spinner("Fetching news + prices..."):
            articles = cached_yfinance_news(symbol, n_news)
            articles = articles[:n_news]  # enforce exact count
            df = cached_price_history(symbol, "1mo")

        if not articles:
            st.error("No news available for this ticker.")
            st.stop()
        if df.empty or len(df) < 5:
            st.error("Not enough price data.")
            st.stop()

        recent = df.tail(7)
        prices = [
            {"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
             "close": round(float(r["close"]), 2)}
            for _, r in recent.iterrows()
        ]
        headlines = [a["title"] for a in articles]

        with st.spinner("Loading FinGPT (cached after first run)..."):
            tokenizer, model = cached_fingpt()

        from models.fingpt_forecast import run_forecast as fingpt_run
        with st.spinner("Running FinGPT inference..."):
            result = fingpt_run(symbol, headlines, prices, tokenizer, model)

        # Persist forecast to session state so it survives navigation
        st.session_state["forecast_result"] = {
            "symbol": symbol,
            "headlines": headlines,
            "prices": prices,
            "result": result,
        }

    # ── Render from session_state ─────────────────────────────────────────────
    if "forecast_result" in st.session_state:
        cache = st.session_state["forecast_result"]
        result = cache["result"]
        prices = cache["prices"]
        headlines = cache["headlines"]
        cached_symbol = cache["symbol"]

        pred = result["prediction"]
        sections = result["sections"]

        st.caption(f"Showing cached forecast for **{cached_symbol}**. "
                   f"Click **Run FinGPT Forecast** above to generate a new one.")

        pred_color = {"rise": "#16a34a", "fall": "#dc2626", "remain": "#ca8a04"}.get(pred, "#94a3b8")
        pred_bg = {"rise": "#dcfce7", "fall": "#fee2e2", "remain": "#fef3c7"}.get(pred, "#f1f5f9")

        st.markdown(f"""
        <div style="background:{pred_bg};
                    border-left:5px solid {pred_color};border-radius:8px;
                    padding:24px;margin:20px 0;">
            <div style="font-size:0.85rem;color:#475569;">Next Week Prediction for {cached_symbol}</div>
            <div style="font-size:2.6rem;font-weight:700;color:{pred_color};text-transform:uppercase;letter-spacing:0.05em;">
                {pred or 'Unknown'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Positive Developments")
            st.markdown(sections["positive"] or "_No positive developments parsed._")
        with c2:
            st.subheader("Potential Concerns")
            st.markdown(sections["concerns"] or "_No concerns parsed._")

        st.subheader("📋 Analysis")
        st.markdown(sections["analysis"] or "_No analysis parsed._")

        with st.expander("Inputs used (for transparency)"):
            st.markdown("**Price history (last 7 days):**")
            st.dataframe(pd.DataFrame(prices), width="stretch")
            st.markdown("**News headlines:**")
            for i, h in enumerate(headlines, 1):
                st.markdown(f"{i}. {h}")

        st.caption("Educational/analytical purposes only. NOT financial advice.")
