import csv
import os
import re
from typing import List

from groq import Groq
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY","")
MODEL = "llama-3.3-70b-versatile"

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


def extract_label(text: str) -> str:
    text_lower = text.lower()
    matches = re.findall(r"\b(negative|neutral|positive)\b", text_lower)
    if matches:
        return matches[0]
    return "unknown"


def analyze_sentiment(client: Groq, headline: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment analysis expert. "
                    "When given a news headline, respond with exactly one word: "
                    "positive, negative, or neutral."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"What is the sentiment of this financial news headline?\n"
                    f"{headline}\n"
                    f"Answer with only one word: positive, negative, or neutral."
                ),
            },
        ],
        max_tokens=10,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    label = extract_label(raw)
    return label, raw


def main():
    if GROQ_API_KEY == "your_groq_api_key_here":
        print("ERROR: Set your Groq API key in GROQ_API_KEY environment variable.")
        print("Get a free key at: https://console.groq.com")
        return

    client = Groq(api_key=GROQ_API_KEY)

    headlines = load_headlines(INPUT_FILE)
    print(f"Loaded {len(headlines)} headlines.")
    print(f"Using model: {MODEL}\n")

    results = []

    for idx, headline in enumerate(headlines, start=1):
        label, raw = analyze_sentiment(client, headline)
        print(f"[{idx}] {label} | {headline}")
        results.append(
            {
                "headline": headline,
                "prediction": label,
                "raw_output": raw,
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["headline", "prediction", "raw_output"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
