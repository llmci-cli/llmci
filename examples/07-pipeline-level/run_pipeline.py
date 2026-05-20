#!/usr/bin/env python3
"""Mock RAG pipeline for the Scaffold demo.

Simulates a retrieval-augmented generation pipeline:
1. Retrieves relevant documents from a mock knowledge base
2. Generates an answer based on retrieved context

This tests the full pipeline — not just the prompt, but also the
retrieval quality and how the LLM uses the retrieved context.

Usage: python3 run_pipeline.py --input {input_file} --output {output_file}
"""

import argparse
import json

KNOWLEDGE_BASE = {
    "python": {
        "doc": "Python is a high-level programming language. "
               "It supports multiple paradigms including OOP and functional. "
               "Python 3.12 added improved error messages and f-string improvements.",
        "keywords": ["python", "programming", "language", "pip", "virtualenv"],
    },
    "docker": {
        "doc": "Docker is a containerization platform. "
               "It uses images and containers to package applications. "
               "Docker Compose orchestrates multi-container applications.",
        "keywords": ["docker", "container", "image", "compose", "dockerfile"],
    },
    "git": {
        "doc": "Git is a distributed version control system. "
               "Key commands: git add, git commit, git push, git pull. "
               "Branching allows parallel development workflows.",
        "keywords": ["git", "version control", "branch", "commit", "merge"],
    },
    "kubernetes": {
        "doc": "Kubernetes (K8s) orchestrates containerized workloads. "
               "Core concepts: pods, services, deployments, namespaces. "
               "kubectl is the CLI for interacting with clusters.",
        "keywords": ["kubernetes", "k8s", "pod", "cluster", "kubectl", "deploy"],
    },
}


def retrieve(query: str, top_k: int = 2) -> list[str]:
    """Mock retrieval: match documents by keyword overlap."""
    query_lower = query.lower()
    scored = []
    for topic, entry in KNOWLEDGE_BASE.items():
        score = sum(1 for kw in entry["keywords"] if kw in query_lower)
        if score > 0:
            scored.append((score, entry["doc"]))

    scored.sort(key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_k]]


def generate(query: str, context: list[str]) -> str:
    """Mock generation: combine query with retrieved context."""
    if not context:
        return "I don't have enough information to answer that question."

    combined_context = " ".join(context)

    query_lower = query.lower()
    if "what is" in query_lower or "explain" in query_lower:
        return f"Based on the documentation: {combined_context}"
    elif "how" in query_lower:
        return f"Here's how: {combined_context}"
    else:
        return f"Answer: {combined_context}"


def run_pipeline(input_data: dict) -> dict:
    """Run the full RAG pipeline."""
    query = input_data["input"]
    docs = retrieve(query)
    answer = generate(query, docs)
    return {"output": answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    result = run_pipeline(data)
    json.dump(result, open(args.output, "w"))


if __name__ == "__main__":
    main()
