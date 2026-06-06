#!/usr/bin/env python3
"""A deterministic mock RAG pipeline for the retrieval example.

Usage: python3 run_rag.py --input {input_file} --output {output_file}

It "retrieves" passages from a tiny fixed corpus by keyword overlap and emits the
structured output the RAG judge expects:

    {"output": "<answer>", "contexts": [...], "retrieved_ids": [...]}

Only the retrieval criteria (recall/precision) are scored here, so the pipeline is fully
deterministic and needs no API key.
"""

import argparse
import json

# id -> (keywords that should retrieve it, passage text)
CORPUS = {
    "doc1": (["france", "capital", "paris"], "Paris is the capital of France."),
    "doc2": (["france", "population"], "France has about 68 million people."),
    "doc3": (["japan", "capital", "tokyo"], "Tokyo is the capital of Japan."),
    "doc4": (["japan", "currency", "yen"], "Japan's currency is the yen."),
    "doc5": (["water", "boil", "celsius"], "Water boils at 100 degrees Celsius."),
}


def retrieve(question: str, k: int = 3) -> tuple[list[str], list[str]]:
    """Return (ids, passages) ranked by keyword overlap with the question."""
    q = question.lower()
    scored = [
        (sum(1 for kw in keywords if kw in q), doc_id, text)
        for doc_id, (keywords, text) in CORPUS.items()
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = [(doc_id, text) for score, doc_id, text in scored if score > 0][:k]
    return [doc_id for doc_id, _ in top], [text for _, text in top]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    ids, passages = retrieve(data["input"])
    answer = passages[0] if passages else "I don't know."

    with open(args.output, "w") as f:
        json.dump({
            "output": answer,
            "contexts": passages,
            "retrieved_ids": ids,
        }, f)


if __name__ == "__main__":
    main()
