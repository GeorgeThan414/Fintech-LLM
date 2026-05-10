"""
Sentiment analysis - dual engine.
- Groq: fast cloud API (Llama 3.3 70B)
- FinGPT: local model (uses the loaded forecaster's chat base + sentiment prompt)
"""

import re
from typing import Dict, Optional

from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"


def _extract_sentiment(text: str) -> str:
    matches = re.findall(r"\b(positive|negative|neutral)\b", text.lower())
    return matches[0] if matches else "unknown"


def _extract_impact(text: str) -> str:
    matches = re.findall(r"\b(high|medium|low)\b", text.lower())
    return matches[0] if matches else "medium"


def _extract_reason(text: str) -> str:
    """Pull the 'Reason: ...' line from the model's structured output."""
    match = re.search(r"Reason:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the raw text minus any structured prefix lines
    cleaned = re.sub(r"^(Sentiment|Impact):\s*\S+\s*\n?", "", text, flags=re.MULTILINE | re.IGNORECASE)
    return cleaned.strip()[:300] or "(no reason provided)"


def analyze_with_groq(client: Groq, headline: str) -> Dict:
    """
    Returns: {sentiment, impact, raw, engine}
    Single call returns sentiment + impact + reason.
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior equity analyst. Given a news headline, classify market "
                        "sentiment and explain the underlying mechanism that drives it. "
                        "Do NOT rephrase or summarize the headline. Instead, in the Reason field, "
                        "explain WHY this matters to investors — what business/financial mechanism "
                        "is at play (e.g. margin pressure, market share shift, regulatory risk, "
                        "competitive moat, demand signal, valuation re-rating). Reference market "
                        "context, comparable events, or downstream effects when relevant.\n\n"
                        "Respond in this exact format:\n"
                        "Sentiment: <positive|negative|neutral>\n"
                        "Impact: <high|medium|low>\n"
                        "Reason: <1-2 sentences of analysis, NOT a paraphrase of the headline>"
                    ),
                },
                {"role": "user", "content": f"Headline: {headline}"},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        return {
            "sentiment": _extract_sentiment(raw),
            "impact": _extract_impact(raw),
            "reason": _extract_reason(raw),
            "raw": raw,
            "engine": "Groq (Llama 3.3 70B)",
        }
    except Exception as e:
        return {
            "sentiment": "unknown",
            "impact": "medium",
            "reason": f"Error: {e}",
            "raw": f"Error: {e}",
            "engine": "Groq (error)",
        }


def analyze_with_fingpt(headline: str, tokenizer, model) -> Dict:
    """
    Use the local Llama-2-7b-chat base model for sentiment with both LoRAs disabled.
    Why not use a FinGPT sentiment LoRA?
      - FinGPT/fingpt-sentiment_llama2-7b_lora doesn't exist (only the 13B variant does)
      - FinGPT/fingpt-mt_llama2-7b_lora is multi-task and persistently confuses sentiment
        with binary headline classification, returning Yes/No instead of pos/neg/neutral.
    The pure chat base reliably produces sentiment + impact + reasoning via prompting.
    """
    import torch

    # Disable BOTH adapters for sentiment so we get the pure chat base.
    # PEFT's disable_adapter() is a context manager that auto-re-enables on exit.
    use_disable_ctx = hasattr(model, "disable_adapter")

    system_prompt = (
        "You are a senior equity analyst. Given a news headline, classify market sentiment "
        "and explain the underlying mechanism. Do NOT rephrase or summarize the headline. "
        "In the Reason field, explain WHY this matters to investors — the business/financial "
        "mechanism at play (margin pressure, market share, regulatory risk, demand signal, "
        "competitive position, valuation impact). Reference market context when relevant.\n\n"
        "Respond in this exact format and nothing else:\n"
        "Sentiment: <positive|negative|neutral>\n"
        "Impact: <high|medium|low>\n"
        "Reason: <1-2 sentences of actual analysis, NOT a paraphrase of the headline>"
    )
    prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nHeadline: {headline} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        if use_disable_ctx:
            with model.disable_adapter():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()

    return {
        "sentiment": _extract_sentiment(generated),
        "impact": _extract_impact(generated),
        "reason": _extract_reason(generated),
        "raw": generated,
        "engine": "Llama-2-7b-chat (FinGPT base, both LoRAs disabled)",
    }


def make_groq_client(api_key: str) -> Optional[Groq]:
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None
