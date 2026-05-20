#!/usr/bin/env python3
"""Mock FastAPI-style classification service.

Simulates a real-world pattern: a service that classifies customer support
tickets through a pipeline of pre-processing → LLM classification → post-processing.

This is NOT a real FastAPI server (no HTTP dependency needed for the demo).
It exposes a classify() function that the eval wrapper calls directly.
"""

import re


CATEGORY_KEYWORDS = {
    "hardware": [
        "printer", "monitor", "keyboard", "webcam", "hard drive", "pixel",
        "wifi", "cable", "mouse", "screen", "usb", "battery", "charger",
        "overheating", "fan", "dock", "hdmi", "bluetooth",
    ],
    "billing": [
        "refund", "charged", "invoice", "subscription", "upgrade", "cancel",
        "payment", "billing", "plan", "price", "coupon", "discount", "receipt",
        "overcharged", "credit", "renewal", "trial",
    ],
    "account": [
        "password", "email", "export", "unsubscribe", "two-factor", "login",
        "sign", "account", "profile", "authentication", "sso", "locked out",
        "reset", "deactivate", "permissions", "role",
    ],
    "software": [
        "crash", "app", "website", "loading", "error", "search",
        "notification", "bug", "update", "slow", "freeze", "sync",
        "install", "uninstall", "compatibility", "dark mode",
    ],
}

CONFIDENCE_THRESHOLD = 2

PII_PATTERNS = [
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD]"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]"),
]


def preprocess(text: str) -> str:
    """Pre-processing: normalize text and redact PII."""
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    for pattern, replacement in PII_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)

    return cleaned


def classify_core(text: str) -> tuple[str, int]:
    """Core classification: keyword matching (simulates an LLM call)."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text_lower)

    best = max(scores, key=lambda k: scores[k])
    return best, scores[best]


def postprocess(category: str, confidence: int) -> str:
    """Post-processing: apply confidence thresholds and fallback routing."""
    if confidence < CONFIDENCE_THRESHOLD:
        return "general"
    return category


def classify(text: str) -> dict:
    """Full pipeline: preprocess → classify → postprocess."""
    cleaned = preprocess(text)
    category, confidence = classify_core(cleaned)
    final_category = postprocess(category, confidence)
    return {
        "category": final_category,
        "confidence": confidence,
        "preprocessed_text": cleaned,
    }
