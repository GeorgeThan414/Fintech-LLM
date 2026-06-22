"""
FinGPT *dedicated* sentiment model — the real fine-tuned sentiment LoRA.

Unlike models/fingpt_forecast.py (which uses the 7B chat base with the multi-task
LoRA DISABLED because that LoRA returns yes/no headline-classification instead of
sentiment), this loads the purpose-built sentiment adapter:

    base : NousResearch/Llama-2-13b-hf   (plain 13B, NOT chat — matches the adapter)
    lora : FinGPT/fingpt-sentiment_llama2-13b_lora

It outputs clean negative / neutral / positive labels via the FinGPT instruction
prompt. Loaded in 4-bit so the 13B base fits in ~9GB VRAM (the model card's 8-bit
recipe needs ~14GB and will OOM on a 12GB card).
"""

from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Confirmed from the adapter's adapter_config.json:
#   base_model_name_or_path = NousResearch/Llama-2-13b-hf  (ungated, MIT)
BASE_MODEL = "NousResearch/Llama-2-13b-hf"
SENTIMENT_ADAPTER = "FinGPT/fingpt-sentiment_llama2-13b_lora"


def load_fingpt_sentiment_model() -> Tuple:
    """Load tokenizer + 4-bit 13B base + the FinGPT sentiment LoRA. Returns (tok, model)."""
    print("Loading tokenizer (Llama-2-13B)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit 13B base model...")
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
    )

    print("Loading FinGPT sentiment LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, SENTIMENT_ADAPTER)
    model.eval()
    print("[FinGPT-Sentiment] 13B base + sentiment LoRA loaded.")
    return tokenizer, model