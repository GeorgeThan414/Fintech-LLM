"""
FinGPT - Llama-2-7b-chat-hf with TWO LoRA adapters:
- 'sentiment' = FinGPT/fingpt-sentiment_llama2-7b_lora (financial sentiment classification)
- 'forecast'  = FinGPT/fingpt-forecaster_dow30_llama2-7b_lora (stock direction prediction)

Both LoRAs share the same base model in memory. Switch tasks with:
    model.set_adapter("sentiment")  # before sentiment inference
    model.set_adapter("forecast")   # before forecast inference
"""

from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Prompt building + output parsing live in forecast_common (torch-free) so the Groq
# engine can reuse the exact same prompt/parse logic for a fair comparison.
from models.forecast_common import (  
    SYSTEM_PROMPT,
    build_forecast_prompt,
    extract_prediction,
    parse_sections,
)

BASE_MODEL = "daryl149/llama-2-7b-chat-hf"
# fingpt-mt = multi-task LoRA (sentiment + NER + relation extraction + headline classification)
# Note: there is no standalone fingpt-sentiment for 7B — only mt_llama2-7b or sentiment_llama2-13b.
# We use mt_llama2-7b for sentiment because the base must match our 7B base.
SENTIMENT_ADAPTER = "FinGPT/fingpt-mt_llama2-7b_lora"
FORECAST_ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"


def load_fingpt_model() -> Tuple:
    """
    Load tokenizer + 4-bit base model + BOTH LoRA adapters (sentiment + forecast).
    Returns (tokenizer, model). Switch tasks via model.set_adapter("sentiment" | "forecast").
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit base model (chat variant)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading FinGPT multi-task LoRA adapter (handles sentiment, ~170MB)...")
    model = PeftModel.from_pretrained(base_model, SENTIMENT_ADAPTER, adapter_name="sentiment")

    print("Loading FinGPT forecaster LoRA adapter (~170MB)...")
    model.load_adapter(FORECAST_ADAPTER, adapter_name="forecast")

    # Default to forecast adapter active; sentiment_fetchers will switch when needed
    model.set_adapter("forecast")
    model.eval()
    print("[FinGPT] Both adapters loaded. Default active: 'forecast'.")
    return tokenizer, model


def run_forecast(
    ticker: str,
    headlines: List[str],
    prices: List[Dict],
    tokenizer,
    model,
    max_new_tokens: int = 512,
) -> Dict:
    """Run the full forecast pipeline. Returns dict with prediction + sections + raw."""
    # Activate the forecast adapter (sentiment_fetchers switches to "sentiment" for that task)
    if hasattr(model, "set_adapter"):
        model.set_adapter("forecast")
        print("[FinGPT Forecast] Active adapter: 'forecast'")
    elif hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()

    prompt = build_forecast_prompt(ticker, headlines, prices)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip()
    print(f"[FinGPT Forecast] Generated {len(generated)} chars: {generated[:200]}...")

    sections = parse_sections(generated)
    label = extract_prediction(generated)

    return {
        "prediction": label,
        "sections": sections,
        "raw": generated,
    }
