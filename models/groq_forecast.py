"""Forecasting with Groq (Llama-3.3-70B) — the cloud counterpart to FinGPT.

Uses the SAME prompt + output parsing as the local FinGPT forecaster
(models/forecast_common) so the two engines are directly comparable. The only
difference is the model and that we send a chat system/user message pair instead
of the Llama-2 [INST] wrapper.
"""

from typing import Dict, List, Optional

from groq import Groq

from models.forecast_common import (
    build_forecast_messages,
    extract_prediction,
    parse_sections,
)

GROQ_MODEL = "llama-3.3-70b-versatile"


def forecast_with_groq(
    client: Groq,
    ticker: str,
    headlines: List[str],
    prices: List[Dict],
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> Dict:
    """Return {prediction, sections, raw} — same shape as fingpt_forecast.run_forecast."""
    system, user = build_forecast_messages(ticker, headlines, prices)
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = resp.choices[0].message.content.strip()
        return {
            "prediction": extract_prediction(raw),
            "sections": parse_sections(raw),
            "raw": raw,
        }
    except Exception as e:
        return {"prediction": "unknown", "sections": {}, "raw": f"Error: {e}"}


def make_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None
