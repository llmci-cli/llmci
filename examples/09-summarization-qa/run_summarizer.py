#!/usr/bin/env python3
"""Mock article summarizer for the summarization QA demo.

Simulates a production summarization pipeline with configurable strategies:
- Extractive: picks top-scoring sentences by keyword density
- Length control: enforces max sentence count
- Lead bias: boosts earlier sentences (mimics journalistic "inverted pyramid")

Usage: python3 run_summarizer.py --input {input_file} --output {output_file}
"""

import argparse
import json
import re
from collections import Counter

MAX_SENTENCES = 3
LEAD_BIAS = 0.8
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
    "for", "of", "and", "or", "but", "with", "by", "from", "that", "this",
    "it", "its", "has", "had", "have", "been", "will", "would", "could",
    "should", "can", "may", "not", "no", "do", "did", "does", "be", "as",
}


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences, handling common abbreviations."""
    text = re.sub(r"(Mr|Mrs|Dr|Jr|Sr|vs|etc)\.", r"\1<DOT>", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.replace("<DOT>", ".") for s in sentences if len(s.split()) >= 4]


def score_sentence(sentence: str, word_freq: Counter, position: int, total: int) -> float:
    """Score a sentence by keyword density and position."""
    words = re.findall(r"[a-z0-9]+", sentence.lower())
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]

    if not content_words:
        return 0.0

    keyword_score = sum(word_freq.get(w, 0) for w in content_words) / len(content_words)

    position_score = 1.0 - (position / max(total, 1)) * LEAD_BIAS

    return keyword_score * position_score


def summarize(text: str) -> str:
    """Generate an extractive summary of the input text."""
    sentences = extract_sentences(text)
    if len(sentences) <= MAX_SENTENCES:
        return text.strip()

    all_words = re.findall(r"[a-z0-9]+", text.lower())
    content_words = [w for w in all_words if w not in STOPWORDS and len(w) > 2]
    word_freq = Counter(content_words)

    scored = []
    for i, sent in enumerate(sentences):
        score = score_sentence(sent, word_freq, i, len(sentences))
        scored.append((score, i, sent))

    scored.sort(key=lambda x: -x[0])
    top = scored[:MAX_SENTENCES]
    top.sort(key=lambda x: x[1])

    return " ".join(s for _, _, s in top)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = summarize(data["input"])
    json.dump({"output": result}, open(args.output, "w"))


if __name__ == "__main__":
    main()
