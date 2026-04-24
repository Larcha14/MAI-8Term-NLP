import json
import os
import re

import httpx
from fastapi import FastAPI, HTTPException
from fastmcp import Client

from app.schemas import AnalyzeRequest

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:9000/mcp")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")

app = FastAPI(title="Credit Approval PoC")


def infer_age_from_text(message: str):
    """
    Infer client age from text.

    Args:
        message: Raw user message.

    Returns:
        Integer age if found, otherwise None.
    """
    text = message.lower()
    match = re.search(r"(\d{1,2})\s*(?:год|года|лет)", text)
    if match:
        return int(match.group(1))
    return None


def infer_income_from_text(message: str):
    """
    Infer client income from text.

    Supports forms like:
    - доход 120000
    - 120к
    - доход 120к

    Args:
        message: Raw user message.

    Returns:
        Float income if found, otherwise None.
    """
    text = message.lower().replace(" ", "")

    match = re.search(r"доход[:=]?(\d+(?:[.,]\d+)?)к\b", text)
    if match:
        return float(match.group(1).replace(",", ".")) * 1000

    match = re.search(r"(\d+(?:[.,]\d+)?)к\b", text)
    if match:
        return float(match.group(1).replace(",", ".")) * 1000

    match = re.search(r"доход[:=]?(\d{4,})", text)
    if match:
        return float(match.group(1))

    match = re.search(r"(\d{4,})", text)
    if match:
        return float(match.group(1))

    return None


def infer_employment_years_from_text(message: str):
    """
    Infer employment duration from text.

    Args:
        message: Raw user message.

    Returns:
        Float number of years if found, otherwise None.
    """
    text = message.lower()

    if "полгода" in text:
        return 0.5

    match = re.search(r"работает\s*(\d+(?:[.,]\d+)?)\s*(?:год|года|лет)", text)
    if match:
        return float(match.group(1).replace(",", "."))

    match = re.search(r"стаж\s*(\d+(?:[.,]\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", "."))

    return None


def infer_married_from_text(message: str):
    """
    Infer marital status from text.

    Args:
        message: Raw user message.

    Returns:
        True, False, or None.
    """
    text = message.lower()

    negative_patterns = [
        "не женат",
        "не замужем",
        "холост",
        "разведен",
        "разведён",
    ]
    positive_patterns = [
        "женат",
        "замужем",
        "состоит в браке",
        "в браке",
    ]

    if any(pattern in text for pattern in negative_patterns):
        return False
    if any(pattern in text for pattern in positive_patterns):
        return True
    return None


def infer_overdues_from_text(message: str):
    """
    Infer overdue history from text.

    Args:
        message: Raw user message.

    Returns:
        True, False, or None.
    """
    text = message.lower()

    negative_patterns = [
        "просрочек не было",
        "случаев просрочки не было",
        "долгов по кредитам не было",
        "просрочки отсутствовали",
        "без просрочек",
    ]
    positive_patterns = [
        "были просрочки",
        "есть просрочки",
        "допускал просрочки",
        "имел просрочки",
        "просрочки были",
    ]

    if any(pattern in text for pattern in negative_patterns):
        return False
    if any(pattern in text for pattern in positive_patterns):
        return True
    return None


def apply_rule_fallbacks(features: dict, message: str) -> dict:
    """
    Apply rule-based extraction and overwrite unstable LLM fields when rules are clear.

    Args:
        features: Parsed features from LLM.
        message: Raw user message.

    Returns:
        Updated features dictionary.
    """
    age_value = infer_age_from_text(message)
    if age_value is not None:
        features["age"] = age_value

    income_value = infer_income_from_text(message)
    if income_value is not None:
        features["income"] = income_value

    employment_value = infer_employment_years_from_text(message)
    if employment_value is not None:
        features["employment_years"] = employment_value

    married_value = infer_married_from_text(message)
    if married_value is not None:
        features["married"] = married_value

    overdues_value = infer_overdues_from_text(message)
    if overdues_value is not None:
        features["has_overdues"] = overdues_value

    return features


async def extract_features(message: str) -> dict:
    """
    Extract structured client features from free-text description using Ollama.

    Args:
        message: Free-text client description.

    Returns:
        Dictionary with normalized client features.

    Raises:
        HTTPException: If required fields are missing or request fails.
    """
    prompt = f"""
Извлеки признаки клиента из текста.

Верни ТОЛЬКО валидный JSON.
Без markdown.
Без пояснений.
Не выдумывай значения.
Если значение не указано явно, поставь null.

Используй ровно такие ключи:
age, income, has_overdues, has_higher_education, married, employment_years

Правила интерпретации:
- "женат", "замужем", "состоит в браке" -> married = true
- "не женат", "не замужем", "холост", "разведен", "разведён" -> married = false
- "высшее образование", "вышка", "имеет высшее образование" -> has_higher_education = true
- "без высшего образования", "среднее образование", "среднее специальное" -> has_higher_education = false
- "просрочек не было", "случаев просрочки не было", "долгов по кредитам не было" -> has_overdues = false
- "были просрочки", "есть просрочки", "просрочки были" -> has_overdues = true
- "120к" означает 120000
- "5 лет" в контексте работы означает employment_years = 5

Текст:
{message}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "Ты извлекаешь признаки клиента в структурированном виде.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "age": {"type": ["integer", "null"]},
                "income": {"type": ["number", "null"]},
                "has_overdues": {"type": ["boolean", "null"]},
                "has_higher_education": {"type": ["boolean", "null"]},
                "married": {"type": ["boolean", "null"]},
                "employment_years": {"type": ["number", "null"]},
            },
            "required": [
                "age",
                "income",
                "has_overdues",
                "has_higher_education",
                "married",
                "employment_years",
            ],
        },
        "options": {
            "temperature": 0
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ollama request failed: {e}")

    content = data["message"]["content"]
    features = json.loads(content)

    features = apply_rule_fallbacks(features, message)

    required_keys = [
        "age",
        "income",
        "has_overdues",
        "has_higher_education",
        "married",
        "employment_years",
    ]

    missing_or_null = [
        key for key in required_keys
        if key not in features or features[key] is None
    ]

    if missing_or_null:
        raise HTTPException(
            status_code=422,
            detail=f"Insufficient client data. Missing or null fields: {missing_or_null}",
        )

    return {
        "age": int(features["age"]),
        "income": float(features["income"]),
        "has_overdues": bool(features["has_overdues"]),
        "has_higher_education": bool(features["has_higher_education"]),
        "married": bool(features["married"]),
        "employment_years": float(features["employment_years"]),
    }


@app.post("/analyze")
async def analyze(request: AnalyzeRequest) -> dict:
    """
    Analyze a client's free-text description.

    Args:
        request: Request body with client message.

    Returns:
        Structured preliminary decision with extracted features, score, and risk.

    Raises:
        HTTPException: If extraction or MCP call fails.
    """
    features = await extract_features(request.message)

    try:
        async with Client(MCP_SERVER_URL) as client:
            score_result = await client.call_tool(
                "calculate_credit_score",
                {
                    "age": features["age"],
                    "income": features["income"],
                    "has_overdues": features["has_overdues"],
                    "has_higher_education": features["has_higher_education"],
                    "married": features["married"],
                    "employment_years": features["employment_years"],
                },
            )

            risk_result = await client.call_tool(
                "assess_credit_risk",
                {
                    "income": features["income"],
                    "has_overdues": features["has_overdues"],
                    "employment_years": features["employment_years"],
                    "married": features["married"],
                },
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MCP call failed: {e}")

    score = score_result.data["score"]
    risk = risk_result.data["risk"]

    if score >= 75 and risk == "low":
        decision = "preliminarily_approved"
    elif risk == "high" or score < 40:
        decision = "preliminarily_rejected"
    else:
        decision = "manual_review"

    return {
        "status": "valid",
        "decision": decision,
        "features": features,
        "score": score,
        "risk": risk,
    }