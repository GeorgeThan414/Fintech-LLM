"""
FinGPT - Llama-2-7b-chat-hf with TWO LoRA adapters:
- 'sentiment' = FinGPT/fingpt-sentiment_llama2-7b_lora (financial sentiment classification)
- 'forecast'  = FinGPT/fingpt-forecaster_dow30_llama2-7b_lora (stock direction prediction)

Both LoRAs share the same base model in memory. Switch tasks with:
    model.set_adapter("sentiment")  # before sentiment inference
    model.set_adapter("forecast")   # before forecast inference
"""

import re
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "daryl149/llama-2-7b-chat-hf"
# fingpt-mt = multi-task LoRA (sentiment + NER + relation extraction + headline classification)
# Note: there is no standalone fingpt-sentiment for 7B — only mt_llama2-7b or sentiment_llama2-13b.
# We use mt_llama2-7b for sentiment because the base must match our 7B base.
SENTIMENT_ADAPTER = "FinGPT/fingpt-mt_llama2-7b_lora"
FORECAST_ADAPTER = "FinGPT/fingpt-forecaster_dow30_llama2-7b_lora"

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


def build_forecast_prompt(ticker: str, headlines: List[str], prices: List[Dict]) -> str:
    """
    prices: list of {"date": "YYYY-MM-DD", "close": float}
    headlines: list of news headline strings
    """
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
        f"stock price movement for next week. Provide a summary analysis to support your prediction.\n\n"
        f"IMPORTANT: End your response with EXACTLY this format on its own line:\n"
        f"Prediction: Rise\n"
        f"OR\n"
        f"Prediction: Fall\n"
        f"OR\n"
        f"Prediction: Remain\n"
        f"Then write 'Analysis:' followed by your reasoning."
    )

    return f"[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_message} [/INST]"


def extract_prediction(text: str) -> str:
    # 1. Look for the structured "Prediction: Rise/Fall/Remain" line
    match = re.search(r"Prediction:\s*(Rise|Fall|Remain)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()

    # 2. Look for directional language anywhere in the text (last match wins — usually the conclusion)
    text_lower = text.lower()

    rise_terms = r"\b(rise|increase|increasing|upward|bullish|growth|growing|gain|gains|outperform|outperforming|positive trend|uptrend|appreciate|appreciation|surge|rally|climb|climbing)\b"
    fall_terms = r"\b(fall|decline|declining|drop|drops|dropping|decrease|decreasing|downward|bearish|loss|losses|underperform|underperforming|negative trend|downtrend|depreciate|depreciation|plunge|crash|tumble)\b"
    remain_terms = r"\b(remain|remains|remaining|stable|flat|sideways|unchanged|neutral|hold steady|consolidate|consolidating|range-bound)\b"

    rise_count = len(re.findall(rise_terms, text_lower))
    fall_count = len(re.findall(fall_terms, text_lower))
    remain_count = len(re.findall(remain_terms, text_lower))

    if rise_count == 0 and fall_count == 0 and remain_count == 0:
        return "unknown"

    if rise_count > fall_count and rise_count > remain_count:
        return "rise"
    if fall_count > rise_count and fall_count > remain_count:
        return "fall"
    if remain_count > rise_count and remain_count > fall_count:
        return "remain"
    # Tie: prefer rise over fall over remain (most common bias in financial analysis)
    if rise_count == fall_count:
        return "remain"
    return "rise" if rise_count > fall_count else "fall"


def parse_sections(text: str) -> Dict[str, str]:
    """Parse the model output into structured sections for nice rendering."""
    sections = {"positive": "", "concerns": "", "prediction": "", "analysis": "", "percent": ""}

    # 1. Try the bracketed sections first (when model follows the format strictly)
    pos_match = re.search(
        r"\[Positive Developments\]:?(.*?)(?=\[Potential Concerns\]|\[Prediction|Prediction:|Analysis:|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    con_match = re.search(
        r"\[Potential Concerns\]:?(.*?)(?=\[Prediction|Prediction:|Analysis:|$)",
        text, re.IGNORECASE | re.DOTALL,
    )
    pred_match = re.search(r"Prediction:\s*([^\n]+)", text, re.IGNORECASE)
    anal_match = re.search(r"Analysis:\s*(.*?)$", text, re.IGNORECASE | re.DOTALL)

    if pos_match:
        sections["positive"] = pos_match.group(1).strip()
    if con_match:
        sections["concerns"] = con_match.group(1).strip()
    if pred_match:
        sections["prediction"] = pred_match.group(1).strip()
    if anal_match:
        sections["analysis"] = anal_match.group(1).strip()

    # 2. Fallback: if Positive/Concerns are empty, extract them from the Analysis paragraphs
    if not sections["positive"] or not sections["concerns"]:
        analysis_text = sections["analysis"] or text
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", analysis_text)

        positive_sentences = []
        concern_sentences = []

        positive_keywords = re.compile(
            r"\b(positive|growth|approval|increase|gain|rally|beat|strong|success|"
            r"launch|partnership|approval|recognition|boost|breakthrough|expansion|"
            r"upward|bullish|outperform|record|leadership)\b", re.IGNORECASE,
        )
        concern_keywords = re.compile(
            r"\b(concern|risk|decline|decrease|drop|loss|weak|miss|lawsuit|recall|"
            r"investigation|downgrade|warning|threat|competition|pressure|delay|"
            r"downward|bearish|underperform|debt|deficit)\b", re.IGNORECASE,
        )

        for s in sentences:
            s = s.strip()
            if not s:
                continue
            has_pos = bool(positive_keywords.search(s))
            has_neg = bool(concern_keywords.search(s))
            if has_pos and not has_neg:
                positive_sentences.append(s)
            elif has_neg and not has_pos:
                concern_sentences.append(s)
            elif has_pos and has_neg:
                # Mixed sentence — try splitting on "however", "but", "although"
                parts = re.split(r"\b(however|but|although|despite|while)\b", s, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if positive_keywords.search(part) and not concern_keywords.search(part):
                        positive_sentences.append(part)
                    elif concern_keywords.search(part) and not positive_keywords.search(part):
                        concern_sentences.append(part)

        if not sections["positive"] and positive_sentences:
            sections["positive"] = "\n".join(f"- {s}" for s in positive_sentences[:4])
        if not sections["concerns"] and concern_sentences:
            sections["concerns"] = "\n".join(f"- {s}" for s in concern_sentences[:4])

    # 3. Extract a percentage estimate if mentioned (e.g. "1-2% increase", "rise by 3%")
    pct_patterns = [
        r"(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?%)\s+(?:increase|rise|gain|growth|jump|surge|rally|appreciation)",
        r"(?:increase|rise|gain|growth|jump|surge|rally|appreciation)\s+(?:of\s+)?(?:approximately\s+|about\s+|around\s+)?(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?%)",
        r"(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?%)\s+(?:decrease|decline|drop|fall|loss|depreciation)",
        r"(?:decrease|decline|drop|fall|loss|depreciation)\s+(?:of\s+)?(?:approximately\s+|about\s+|around\s+)?(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?%)",
        r"(?:up|down|by)\s+(\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?%)",
    ]
    for pattern in pct_patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            sections["percent"] = m.group(1)
            break

    return sections


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
