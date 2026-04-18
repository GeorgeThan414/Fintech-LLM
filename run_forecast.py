import csv
import torch
import os, re
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Import yoru API keys
# Gnews - > News regargind specific stocks. Limited Requests calls 100 per day.
#Fetches real financial news headlines automatically
GNEWS_API_KEY = os.environ.get("GNEWS_API_KEY", "")

#Provides real stock market data (prices, charts, historical data)
ALPHAVANTAGE_API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")

# Load the model from Hugging face
# NousResearch shares the same model as FintechGPT, without the need of the permission
GROQ_API_KEY = os.environ.get("GROQ_API_KEY","")
GROQ_MODEL = "llama-3.3-70b-versatile"

#BASE_MODEL = "NousResearch/Llama-2-13b-hf"
#ADAPTER_MODEL = "FinGPT/fingpt-forecaster_llama2-13b_lora"

#Output temporary storage 
OUTPUT_FILE = "forecast_results.csv"


TICKETS= [
    {"symbol": "TSLA", "name": "Tesla"},
    {"symbol": "AAPL", "name": "Apple"},
    {"symbol": "NVDA", "name": "Nvidia"},
]

def fetch_news(company_name, max_articles):
    url = (
        f"https://gnews.io/api/v4/search"
        f"?q={company_name}&lang=en&max={max_articles}&token={GNEWS_API_KEY}")
    response = requests.get(url, timeout=200)
    response.raise_for_status()
    articles= response.json().get("articles", [])
    return [a['title'] for a in articles if a.get("title")]


def fetch_prices(symbol, days):
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHAVANTAGE_API_KEY}"
    )
    response= requests.get(url,timeout=200)
    response.raise_for_status()
    data= response.json().get("Time Series (Daily)", {})
    sorted_dates= sorted(data.keys(), reverse=True)[:days]
    out= [{ "date": date, "close": data[date]["4. close"]} for date in reversed(sorted_dates)]
    
    return out

def build_prompt(company_name, headlines, prices):
    news_block = "\n".join(f"- {h}" for h in headlines)
    price_block = "\n".join(f"- {p['date']}: ${p['close']}" for p in prices)
    return (
        f"Instruction: Based on the recent news and stock price history below, "
        f"predict the stock price movement for {company_name}. "
        f"Please choose an answer from {{up/down/neutral}}.\n"
        f"News:\n{news_block}\n"
        f"Price History:\n{price_block}\n"
        f"Answer: "
    )

def extract_label(text: str) -> str:
    text_lower = text.lower().strip()
    matches = re.findall(r"\b(up|down|neutral)\b", text_lower)
    if matches:
        return matches[0]
    return "unknown"

def load_model():
   return Groq(api_key=GROQ_API_KEY)


def main():
    client = load_model()
    results = []

    for stock in TICKETS:
        symbol = stock["symbol"]
        name = stock["name"]
        print(f"\nProcessing {name} ({symbol})...")

        headlines = fetch_news(name, max_articles=50)
        prices = fetch_prices(symbol, days=7)

        if not headlines:
            print(f"  No news found for {name}, skipping.")
            continue
        if not prices:
            print(f"  No price data found for {symbol}, skipping.")
            continue

        prompt = build_prompt(name, headlines, prices)

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a financial forecasting expert. Based on the news and price history, predict the stock movement. Answer with only one word: up, down, or neutral."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=40,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        label = extract_label(raw)

        print(f"  Prediction: {label}")
        results.append({
            "ticker": symbol,
            "company": name,
            "prediction": label,
            "headlines": " | ".join(headlines),
            "last_close": prices[-1]["close"],
            "raw_output": raw,
        })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ticker", "company", "prediction", "headlines", "last_close", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved forecast results to: {OUTPUT_FILE}")

"""
def load_model():
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

        headlines = fetch_news(name, max_articles=50)
        prices = fetch_prices(symbol, days=7)

        if not headlines:
            print(f"  No news found for {name}, skipping.")
            continue
        if not prices:
            print(f"  No price data found for {symbol}, skipping.")
            continue

        prompt = build_prompt(name, headlines, prices)
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

        print(f"  Prediction: {label}")
        results.append({
            "ticker": symbol,
            "company": name,
            "prediction": label,
            "headlines": " | ".join(headlines),
            "last_close": prices[-1]["close"],
            "raw_output": decoded,
        })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ticker", "company", "prediction", "headlines", "last_close", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved forecast results to: {OUTPUT_FILE}")


"""



if __name__ == "__main__":
    main()