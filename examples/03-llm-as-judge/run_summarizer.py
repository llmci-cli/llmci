#!/usr/bin/env python3
"""Simple extractive summarizer for the LLM-as-judge demo.

Picks the most important sentences based on keyword density.
Usage: python3 run_summarizer.py --input {input_file} --output {output_file}
"""

import json
import re


def summarize(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= max_sentences:
        return text.strip()

    scored = []
    words = text.lower().split()
    word_freq = {}
    for w in words:
        w = re.sub(r"[^a-z0-9]", "", w)
        if len(w) > 3:
            word_freq[w] = word_freq.get(w, 0) + 1

    for sent in sentences:
        score = sum(
            word_freq.get(re.sub(r"[^a-z0-9]", "", w.lower()), 0)
            for w in sent.split()
        )
        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    top = scored[:max_sentences]
    ordered = sorted(top, key=lambda x: sentences.index(x[1]))

    return " ".join(s for _, s in ordered)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = summarize(data["input"])
    json.dump({"output": result}, open(args.output, "w"))


if __name__ == "__main__":
    main()
