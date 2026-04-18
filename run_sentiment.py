import csv
import os
import re
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "meta-llama/Llama-2-13b-hf"
ADAPTER_MODEL = "FinGPT/fingpt-sentiment_llama2-13b_lora"

DEFAULT_HEADLINES = [
    "Tesla shares rose after the company reported stronger-than-expected quarterly earnings.",
    "Apple faces weaker iPhone demand in China according to new analyst estimates.",
    "Markets remained largely unchanged as investors waited for the Federal Reserve decision.",
    "Nvidia stock jumped after the company beat revenue expectations.",
    "A major bank warned of rising credit losses in the commercial real estate sector.",
]

INPUT_FILE = "headlines.txt"
OUTPUT_FILE = "sentiment_results.csv"


def load_headlines(path: str) -> List[str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        if lines:
            return lines
    return DEFAULT_HEADLINES


def build_prompt(text: str) -> str:
    return (
        "Instruction: What is the sentiment of this news? "
        "Please choose an answer from {negative/neutral/positive}.\n"
        f"Input: {text}\n"
        "Answer: "
    )


def extract_label(text: str) -> str:
    text_lower = text.lower()
    matches = re.findall(r"\b(negative|neutral|positive)\b", text_lower)
    if matches:
        return matches[-1]
    return "unknown"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading 4-bit base model...")
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

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    model.eval()

    headlines = load_headlines(INPUT_FILE)
    print(f"Loaded {len(headlines)} headlines.")

    results = []

    for idx, headline in enumerate(headlines, start=1):
        prompt = build_prompt(headline)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = extract_label(decoded)

        print(f"[{idx}] {label} | {headline}")
        results.append(
            {
                "headline": headline,
                "prediction": label,
                "raw_output": decoded,
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "prediction", "raw_output"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()