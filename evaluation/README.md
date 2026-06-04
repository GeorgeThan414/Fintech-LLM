# Model comparison harness (`evaluation/`)

Quantitative comparison of the project's models for the university report. Produces
CSV result tables, ready-to-paste LaTeX (`booktabs`) tables, and vector (PDF) figures
under `results/`.

Two studies:

1. **Sentiment** — FinBERT vs FinGPT vs Groq (Llama-3.3-70B) on two *labeled* benchmarks:
   FPB (in-domain) and FiQA (out-of-domain cross-check).
2. **Forecasting** — FinGPT vs Groq vs naive baselines against *actual* price moves.

---

## 1. Sentiment benchmark

### Dataset
[`ChanceFocus/flare-fpb`](https://huggingface.co/datasets/ChanceFocus/flare-fpb) — a
Parquet mirror of the **Financial PhraseBank** (Malo et al., 2014) used by the FLARE /
FinGPT financial-LLM leaderboards. We evaluate on the **`test` split (970 sentences)**,
3 classes: positive / negative / neutral. Using FLARE-FPB (not raw PhraseBank) means our
numbers are directly comparable to published FinGPT results.

> **Cite:** Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).
> *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST.

### Metrics
Accuracy, macro-F1, weighted-F1, per-class precision/recall/F1, and confusion matrices
(via `scikit-learn`). Macro-F1 is the fair headline metric because the classes are
imbalanced (neutral ≈ 59%, positive ≈ 29%, negative ≈ 12%).

### Datasets (`--dataset`)
| key | repo | split (test) | role |
|---|---|---|---|
| `fpb` | `ChanceFocus/flare-fpb` | 970 (we sample 300) | **in-domain** (FinBERT/FinGPT trained on it) |
| `fiqa` | `ChanceFocus/flare-fiqasa` | 235 (full) | **out-of-domain** cross-check (FinBERT never saw it) |

### How to run
All engines score the **same** materialized eval set (`results/<dataset>/eval_set.csv`) for a
fair head-to-head. Decide the size once; resize with `--limit N --rebuild` (then re-run every
engine). Repeat the block for `--dataset fiqa`.

```bash
# 1. Run each engine (writes results/<dataset>/preds_<engine>.csv)
python -m evaluation.run_sentiment --engine finbert --dataset fpb --limit 300
python -m evaluation.run_sentiment --engine groq    --dataset fpb           # paces 2.1s/call
python -m evaluation.run_sentiment --engine fingpt  --dataset fpb           # GPU/cluster

# 2. Aggregate -> tables   3. Figures
python -m evaluation.metrics  --dataset fpb     # sentiment_metrics.{csv,tex} + per-class + confusion
python -m evaluation.figures  --dataset fpb     # figures/*.pdf
```

`GROQ_API_KEY` is read from `.env`. Where each engine should run:

| Engine | Runs on | Notes |
|---|---|---|
| FinBERT | this laptop (CPU) | ~0.11 s/example, full 970 in ~2 min |
| Groq    | this laptop (cloud API) | rate-limited → use `--limit` (e.g. 300, stratified) |
| FinGPT  | **Aristotle cluster (GPU)** | Llama-2-7B, too slow on CPU; copy `preds_fingpt.csv` back |

### Output → paper mapping (per dataset, under `results/<dataset>/`)

| File | Paper element |
|---|---|
| `sentiment_metrics.tex` | **Table** — main results (accuracy / macro-F1 / weighted-F1) |
| `perclass_<engine>.csv` | per-class precision/recall/F1 (appendix or text) |
| `confusion_<engine>.csv` + `figures/confusion_<engine>.pdf` | **Figure** — confusion matrices |
| `figures/accuracy_macrof1.pdf` | **Figure** — bar chart |
| `preds_<engine>.csv` | error analysis (which sentences each model misses) |

Report **FPB and FiQA side by side**: a model that wins on FPB (in-domain) but drops on
FiQA (out-of-domain) reveals overfitting to the training distribution — a key finding.

---

## Methodological caveats (put these in the paper — they show rigor)

1. **FinBERT and FinGPT have a home-field advantage on FPB.** FinBERT was *trained* on
   Financial PhraseBank, and FinGPT's instruction tuning includes FPB. Groq's Llama-3.3-70B
   is **zero-shot** here. So FPB measures *in-domain* performance for the specialists vs
   *zero-shot generalization* for the 70B generalist — frame the comparison that way rather
   than as "who is best." Consider a second, out-of-domain set (e.g. FiQA `flare-fiqasa`,
   or Twitter financial sentiment) that none of them trained on for a fairer cross-check.

2. **The "FinGPT" sentiment engine is Llama-2-7B-chat base, prompted — not a FinGPT
   sentiment LoRA.** Per `fetchers/sentiment_fetchers.py`, no 7B FinGPT *sentiment* adapter
   exists (only 13B / different base), so the app runs the chat base with adapters disabled.
   Label this engine honestly (e.g. "Llama-2-7B base") in the paper; it is *not* the
   fine-tuned FinGPT sentiment model.

3. **Generative engines can emit unparseable output** → counted as `unknown` (always wrong).
   The `Unknown (%)` column quantifies this; FinBERT is always 0 (fixed classifier).

---

## 2. Forecasting study

`FinGPT` vs `Groq` vs baselines (naive "always Rise", momentum, random) for next-week
direction, scored against the **actual** realized price move.

### Design
- **Eval points** = (ticker, as-of date) pairs over Dow30 names in 2024
  (`evaluation/forecast_data.py`, `TICKERS` × `AS_OF_DATES`).
- **Inputs** (identical for both models): 7-day price context + historical headlines from
  Alpha Vantage NEWS_SENTIMENT bounded to `[as_of − 7d, as_of]` — **no look-ahead** in the
  inputs. Cached to `results/forecast/eval_points.json` (the news fetch is rate-limited).
- **Ground truth**: realized return from `as_of` close to 5 trading days later (yfinance).
- **Both engines use the same prompt** (`models/forecast_common.py`) — FinGPT via the
  Llama-2 `[INST]` wrapper, Groq via the chat system/user split.

### Metrics
- **Directional accuracy (primary)** — up vs down. A model "Remain"/"unknown" is
  non-committal → counts as *incorrect*; the **decisiveness** column reports how often each
  model actually committed to up/down.
- **3-class accuracy (secondary)** — rise/fall/remain with a ±1% "remain" band on ground truth.

### How to run
```bash
python -m evaluation.forecast_data                 # build/cached points (Alpha Vantage)
python -m evaluation.run_forecast --engine groq    # local, cloud API
python -m evaluation.run_forecast --engine fingpt  # on the Aristotle cluster (GPU)
python -m evaluation.forecast_metrics              # forecast_metrics.{csv,tex} + per_point.csv
```

### Output → paper (`results/forecast/`)
| File | Paper element |
|---|---|
| `forecast_metrics.tex` | **Table** — directional/3-class accuracy, models vs baselines |
| `forecast_per_point.csv` | per-prediction detail (ticker, date, each call vs actual) |
| `eval_points.json` | the frozen backtest inputs + ground truth (reproducibility) |

### Caveats (state these explicitly)
- **Small sample.** A handful of (ticker, date) points → wide confidence intervals; treat
  forecasting numbers as illustrative, not definitive. Scale up `TICKERS`/`AS_OF_DATES`
  (more Alpha Vantage quota) for tighter estimates.
- **Look-ahead / memorization.** The LLMs' training cutoffs may post-date the 2024 test
  events, so apparent skill can be recall, not forecasting. The *baselines* (naive/momentum)
  are the honest yardstick: an LLM only "adds value" if it beats them.
- **Alpha Vantage history** starts ~2022 and coverage varies by ticker; thin-news points are
  skipped (`<2` headlines).

---

## Run status (as of this branch)

| Study | Engine | Status |
|---|---|---|
| Sentiment FPB | FinBERT | ✅ 300 — acc 0.893 / macro-F1 0.883 |
| Sentiment FPB | Groq | ✅ 300 — acc 0.633 / macro-F1 0.672 |
| Sentiment FiQA | FinBERT | ✅ 235 — acc 0.557 / macro-F1 0.530 |
| Sentiment FiQA | Groq | ✅ 235 — acc 0.868 / macro-F1 0.692 |
| Forecast | Groq | ✅ 20/20 — dir-acc 0.50 / 3-class 0.50 (decisiveness 0.65) |
| Forecast | baselines | momentum 0.75 / naive 0.65 / random 0.40 dir-acc |
| All | FinGPT | ⏳ pending — run on the Aristotle GPU |

**Headline finding — a domain crossover:** the specialist and the generalist swap places by
domain. FinBERT (trained on FPB) wins in-domain (0.89 vs 0.63 on FPB) but collapses
out-of-domain (0.56 on FiQA), where the zero-shot 70B generalist is far more robust (0.87).
On **forecasting**, Groq (0.50 directional) fails to beat the naive (0.65) or momentum (0.75)
baselines — consistent with LLMs being weak stock forecasters (n=20, illustrative).

**Remaining — FinGPT columns (run on the Aristotle GPU, then copy preds back):**
```bash
python -m evaluation.run_sentiment --engine fingpt --dataset fpb
python -m evaluation.run_sentiment --engine fingpt --dataset fiqa
python -m evaluation.run_forecast  --engine fingpt
python -m evaluation.metrics --dataset fpb  && python -m evaluation.figures --dataset fpb
python -m evaluation.metrics --dataset fiqa && python -m evaluation.figures --dataset fiqa
python -m evaluation.forecast_metrics
```
Note: Groq free tier is **100k tokens/day** — the full Groq sweep (~95k) fits in one day but
leaves little headroom; spread re-runs across days or upgrade at console.groq.com if needed.

## Reproducibility
- Eval set is deterministic (`--seed`, default 42); stratified subsampling preserves class balance.
- Env: the conda `fintech` env (`torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`).
- All result files are written under `results/` and committed so the tables/figures are reproducible from the CSVs.
