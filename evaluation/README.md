# Model comparison harness (`evaluation/`)

Quantitative comparison of the project's models for the university report. Produces
CSV result tables, ready-to-paste LaTeX (`booktabs`) tables, and vector (PDF) figures
under `results/`.

Two studies:

1. **Sentiment** — FinBERT vs FinGPT vs Groq (Llama-3.3-70B) on a *labeled* benchmark.
2. **Forecasting** — FinGPT vs Groq vs naive baselines against *actual* price moves.
   *(planned — see "Forecasting" below; not yet built.)*

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

### How to run
All engines score the **same** materialized eval set (`results/eval_set.csv`) for a fair
head-to-head. Decide the size once; resize with `--limit N --rebuild` (then re-run every
engine).

```bash
# 1. Run each engine (writes results/preds_<engine>.csv)
python -m evaluation.run_sentiment --engine finbert            # local, CPU, ~2 min
python -m evaluation.run_sentiment --engine groq --limit 300   # cloud; needs GROQ_API_KEY
python -m evaluation.run_sentiment --engine fingpt             # GPU/cluster (slow on CPU)

# 2. Aggregate -> tables
python -m evaluation.metrics      # results/sentiment_metrics.{csv,tex} + per-class + confusion

# 3. Figures
python -m evaluation.figures      # results/figures/*.pdf
```

`GROQ_API_KEY` is read from `.env`. Where each engine should run:

| Engine | Runs on | Notes |
|---|---|---|
| FinBERT | this laptop (CPU) | ~0.11 s/example, full 970 in ~2 min |
| Groq    | this laptop (cloud API) | rate-limited → use `--limit` (e.g. 300, stratified) |
| FinGPT  | **Aristotle cluster (GPU)** | Llama-2-7B, too slow on CPU; copy `preds_fingpt.csv` back |

### Output → paper mapping

| File | Paper element |
|---|---|
| `results/sentiment_metrics.tex` | **Table** — main results (accuracy / macro-F1 / weighted-F1) |
| `results/perclass_<engine>.csv` | per-class precision/recall/F1 (appendix or text) |
| `results/confusion_<engine>.csv` + `results/figures/confusion_<engine>.pdf` | **Figure** — confusion matrices |
| `results/figures/accuracy_macrof1.pdf` | **Figure** — bar chart |
| `results/preds_<engine>.csv` | error analysis (which sentences each model misses) |

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

## 2. Forecasting (planned)

`FinGPT` vs `Groq` vs baselines (naive "always Rise", momentum, random) for next-week
direction (Rise/Fall/Remain), scored against the **actual** realized price move.

Key issues to handle, documented when built:
- **Historical news availability** — yfinance only returns recent headlines; need
  AlphaVantage historical news or a dataset to backtest past dates.
- **Look-ahead bias** — the LLMs' training cutoffs may post-date the test events, so
  apparent "forecasting" skill can be memorization. Prefer dates after the model cutoffs,
  and state this limitation explicitly.

---

## Reproducibility
- Eval set is deterministic (`--seed`, default 42); stratified subsampling preserves class balance.
- Env: the conda `fintech` env (`torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`).
- All result files are written under `results/` and committed so the tables/figures are reproducible from the CSVs.
