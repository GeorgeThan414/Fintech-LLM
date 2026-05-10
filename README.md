# Fintech-LLM

Live stock intelligence platform combining real-time market data, LLM-powered news sentiment, and FinGPT forecasting in one Streamlit dashboard.

## Features

- **Stock Overview** — Live prices, OHLC candlestick charts, and company info for any S&P 500 ticker (Yahoo Finance)
- **News & Sentiment** — Latest headlines per ticker, scored for sentiment + impact by **Groq (Llama 3.3 70B)** and/or **FinGPT (local Llama-2-7b-chat)** side-by-side comparison
- **FinGPT Forecast** — Run the FinGPT forecaster (Llama-2 + LoRA fine-tuned on Dow30) on any S&P 500 stock. Outputs structured: Positive Developments, Concerns, Prediction (Rise/Fall/Remain), Analysis

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with your API keys
cp .env.example .env  # if you have an example, otherwise create manually

# 3. Run locally
python run_streamlit.py

# Or with public URL (for sharing with colleagues)
python run_streamlit.py --public
```

The dashboard opens at `http://localhost:8501`. With `--public`, you also get an `https://xxxx.ngrok.io` URL anyone can access.

## Required environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key_here              # https://console.groq.com (free)
ALPHAVANTAGE_API_KEY=your_alphavantage_key   # https://www.alphavantage.co/support/#api-key (free)
NGROK_AUTHTOKEN=your_ngrok_token             # https://dashboard.ngrok.com (free, optional)
```

## Project structure

```
Fintech-LLM/
├── streamlit_app.py          ← Main Streamlit app (4 pages)
├── run_streamlit.py          ← Launcher (with optional --public ngrok flag)
├── requirements.txt          ← Python dependencies
├── .env                      ← API keys (do NOT commit)
├── data/
│   ├── __init__.py
│   ├── sp500_tickers.py      ← S&P 500 list from Wikipedia (cached 7 days)
│   └── sp500_tickers.csv     ← cached ticker list
├── fetchers/
│   ├── __init__.py
│   ├── stock_fetchers.py     ← yfinance: live prices, history, company info
│   ├── news_fetchers.py      ← yfinance + Alpha Vantage news fetchers
│   └── sentiment_fetchers.py ← Groq + FinGPT (LoRA disabled) sentiment engines
└── models/
    ├── __init__.py
    └── fingpt_forecast.py    ← FinGPT forecaster: load + prompt + parse output
```

## How it works

| Component | Source / Model |
|---|---|
| Stock prices | Yahoo Finance via `yfinance` |
| News (per-ticker) | Yahoo Finance (unlimited, ~10-20 articles/ticker) |
| News (bulk) | Alpha Vantage NEWS_SENTIMENT (up to 1000/call, 25 calls/day) |
| Sentiment (cloud) | Groq Llama 3.3 70B |
| Sentiment (local) | Llama-2-7b-chat (FinGPT base, LoRA adapter disabled) |
| Forecasting | Llama-2-7b-chat + FinGPT/fingpt-forecaster_dow30_llama2-7b_lora |

The local FinGPT model is loaded once per session via `@st.cache_resource`. Sentiment uses the **same model in memory** with the LoRA adapter toggled off (`model.disable_adapter()`); forecasting re-enables it. This avoids loading two separate model copies (~28GB VRAM).

## First-run download

- Llama-2-7b-chat base model: ~14GB (cached at `~/.cache/huggingface/hub/`)
- FinGPT LoRA adapter: ~50MB
- Triggered the first time you click any FinGPT button. ~30 second load into VRAM. After that, every subsequent FinGPT call is instant.


## Disclaimer

For educational and analytical purposes only. **NOT financial advice.**
