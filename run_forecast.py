import csv
import torch
import os, re
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Must be the chat variant — the LoRA was trained on Llama-2-7b-chat-hf
# daryl149/llama-2-7b-chat-hf is an ungated mirror of meta-llama/Llama-2-7b-chat-hf
BASE_MODEL = "daryl149/llama-2-7b-chat-hf"
ADAPTER_MODEL = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

OUTPUT_FILE = "forecast_results.csv"

TICKETS = [
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "NVDA", "name": "Nvidia"},
]

SYSTEM_PROMPT = (
    "You are a seasoned stock market analyst. Your task is to list the positive developments "
    "and potential concerns for companies based on relevant news and basic financials from the "
    "past weeks, then provide an analysis and prediction for the companies' stock price movement "
    "for the upcoming week. Your answer format should be as follows:\n\n"
    "[Positive Developments]:\n1. ...\n\n"
    "[Potential Concerns]:\n1. ...\n\n"
    "[Prediction & Analysis]\n"
    "Prediction: {Rise/Fall/Remain}\n"
    "Analysis: ..."
)


def fetch_news(company_name, max_articles):
    url = (
        f"https://gnews.io/api/v4/search"
        f"?q={company_name}&lang=en&max={max_articles}&token={GNEWS_API_KEY}"
    )
    response = requests.get(url, timeout=200)
    response.raise_for_status()
    articles = response.json().get("articles", [])
    return [a["title"] for a in articles if a.get("title")]


def fetch_prices(symbol, days):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    response = requests.get(url, timeout=200)
    response.raise_for_status()
    data = response.json().get("Time Series (Daily)", {})
    sorted_dates = sorted(data.keys(), reverse=True)[:days]
    return [{"date": d, "close": data[d]["4. close"]} for d in reversed(sorted_dates)]


def build_prompt(ticker, headlines, prices):
    start = prices[0]
    end = prices[-1]
    direction = "increased" if float(end["close"]) > float(start["close"]) else "decreased"

    news_block = "\n".join(
        f"[Headline]: {h}\n[Summary]: N/A" for h in headlines
    )

    user_message = (
        f"From {start['date']} to {end['date']}, {ticker}'s stock price "
        f"{direction} from ${start['close']} to ${end['close']}. "
        f"Company news during this period are listed below:\n\n"
        f"{news_block}\n\n"
        f"[Basic Financials]:\nNo basic financial reported.\n\n"
        f"Based on all the information before {end['date']}, let's first analyze the positive "
        f"developments and potential concerns for {ticker}. Come up with 2-4 most important "
        f"factors respectively and keep them concise. Then make your prediction of the {ticker} "
        f"stock price movement for next week. Provide a summary analysis to support your prediction."
    )

    return f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_message} [/INST]"


def extract_label(text: str) -> str:
    # Look for the Prediction line in the model's structured response
    match = re.search(r"Prediction:\s*(Rise|Fall|Remain)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    # Fallback: find any of these words
    matches = re.findall(r"\b(rise|fall|remain|up|down|neutral)\b", text.lower())
    return matches[-1] if matches else "unknown"


def load_model():
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

    print("Loading FinGPT forecaster LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL)
    model.eval()
    return tokenizer, model


def main():
    tokenizer, model = load_model()
    results = []

    for stock in TICKETS:
        symbol = stock["symbol"]
        name = stock["name"]
        print(f"\nProcessing {name} ({symbol})...")

        headlines = fetch_news(name, max_articles=10)
        prices = fetch_prices(symbol, days=7)

        if not headlines:
            print(f"  No news found for {name}, skipping.")
            continue
        if not prices:
            print(f"  No price data found for {symbol}, skipping.")
            continue

        prompt = build_prompt(symbol, headlines, prices)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = decoded[len(prompt):].strip()
        label = extract_label(generated) if generated else "unknown"

        print(f"  Raw generated:\n{generated}\n")
        print(f"  Prediction: {label}")
        results.append({
            "ticker": symbol,
            "company": name,
            "prediction": label,
            "headlines": " | ".join(headlines),
            "last_close": prices[-1]["close"],
            "raw_output": generated,
        })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ticker", "company", "prediction", "headlines", "last_close", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved forecast results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
