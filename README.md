# Fintech-LLM

Live stock intelligence platform combining real-time market data, LLM-powered news sentiment, and FinGPT forecasting in one Streamlit dashboard.

## Features

- **Stock Overview** — Live prices, OHLC candlestick charts, and company info for any S&P 500 ticker (Yahoo Finance)
- **News & Sentiment** — Latest headlines per ticker, scored for sentiment + impact by, and compared across, multiple engines: the dedicated **FinGPT sentiment model (Llama-2-13B + fine-tuned LoRA)**, **Groq (Llama 3.3 70B)**, **FinBERT (ProsusAI local classifier)**, and a plain **Llama-2-7b-chat** base as a generalist baseline
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
├── streamlit_app.py            ← Main Streamlit app (Stock / News & Sentiment / Forecast)
├── run_streamlit.py            ← Launcher (with optional --public ngrok flag)
├── requirements.txt            ← Python dependencies
├── .env                        ← API keys (do NOT commit)
├── data/
│   ├── __init__.py
│   ├── sp500_tickers.py        ← S&P 500 list from Wikipedia (cached 7 days)
│   └── sp500_tickers.csv       ← cached ticker list
├── fetchers/
│   ├── __init__.py
│   ├── stock_fetchers.py       ← yfinance: live prices, history, company info
│   ├── news_fetchers.py        ← yfinance + Alpha Vantage news fetchers
│   └── sentiment_fetchers.py   ← Groq + FinGPT (LoRA disabled) sentiment engines
├── models/
│   ├── __init__.py
│   ├── fingpt_forecast.py      ← FinGPT forecaster: load + prompt + parse output
│   ├── fingpt_sentiment.py     ← FinGPT dedicated sentiment LoRA (Llama-2-13B + adapter)
│   ├── forecast_common.py      ← Shared, framework-free forecast prompt + output parsing
│   └── groq_forecast.py        ← Groq (Llama-3.3-70B) forecaster (same prompt as FinGPT)
├── evaluation/                 ← Model-comparison harness (see evaluation/README.md)
│   ├── data.py · metrics.py · figures.py          ← sentiment: eval set, tables, charts
│   ├── run_sentiment.py                            ← run one sentiment engine
│   ├── forecast_data.py · forecast_metrics.py      ← forecast: eval points, tables
│   ├── run_forecast.py                             ← run one forecast engine
│   └── README.md                                   ← full benchmark methodology
└── results/                    ← Committed benchmark outputs (CSV + LaTeX + PDF figures)
    ├── fpb/        ← sentiment on Financial PhraseBank (in-domain)
    ├── fiqa/       ← sentiment on FiQA-SA (out-of-domain cross-check)
    └── forecast/   ← next-week direction: models vs naive/momentum baselines
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

## Model-comparison harness

The `evaluation/` package benchmarks the models quantitatively (FinBERT vs FinGPT vs Groq
for sentiment; FinGPT vs Groq vs naive/momentum baselines for forecasting) and writes result
tables, LaTeX, and PDF figures under `results/`. See **[`evaluation/README.md`](evaluation/README.md)**
for the datasets, metrics, methodology, and run commands.

```bash
# sentiment (per dataset: fpb / fiqa)
python -m evaluation.run_sentiment --engine finbert --dataset fpb --limit 300
python -m evaluation.metrics --dataset fpb && python -m evaluation.figures --dataset fpb

# forecasting
python -m evaluation.run_forecast --engine groq
python -m evaluation.forecast_metrics
```

## First-run download

- Llama-2-7b-chat base model: ~14GB (cached at `~/.cache/huggingface/hub/`)
- FinGPT LoRA adapter: ~50MB
- Triggered the first time you click any FinGPT button. ~30 second load into VRAM. After that, every subsequent FinGPT call is instant.


## Disclaimer

For educational and analytical purposes only. **NOT financial advice.**
