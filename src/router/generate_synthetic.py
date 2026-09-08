"""Generates synthetic (query, complexity_score) pairs for training the DeBERTa complexity router.

Usage:
    python -m src.router.generate_synthetic [--samples 2000] [--batch-size 20]
"""

import argparse
import json
import sys
import time

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import DATA_PROCESSED_DIR, GROQ_API_KEY

MODEL = "llama-3.3-70b-versatile"
OUTPUT_PATH = DATA_PROCESSED_DIR / "synthetic_queries.jsonl"

SYSTEM_PROMPT = """You are a dataset generator for training a query-complexity classifier used in a RAG system.

Complexity score definitions (float, 0.0 to 1.0):
- 0.0-0.3 (Easy): Single-fact lookup, direct factual question, answerable from one short passage. E.g. "What is the capital of France?"
- 0.3-0.7 (Medium): Requires synthesizing a few sentences of context, comparing two things, or a "how/why" question with a moderately long answer.
- 0.7-1.0 (Hard): Multi-hop reasoning, requires combining multiple distant pieces of context, ambiguous or open-ended questions, or questions needing significant inference beyond the text.

Generate diverse, realistic user queries covering a wide range of domains (science, history, technology, business, health, everyday life, etc).

Respond ONLY with a JSON array of objects, no markdown fences, no commentary. Each object: {"query": "...", "complexity_score": 0.0}"""

USER_PROMPT_TEMPLATE = """Generate {n} NEW synthetic (query, complexity_score) pairs.
Distribute roughly evenly across easy/medium/hard bands. Do not repeat previous examples.
Vary phrasing, domain, and length."""


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
def generate_batch(client: Groq, n: int) -> list[dict]:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(n=n)},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    data = json.loads(content)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array from the model")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()

    if not GROQ_API_KEY:
        sys.exit("GROQ_API_KEY not set. Copy .env.example to .env and add your key.")

    client = Groq(api_key=GROQ_API_KEY)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    collected = 0
    seen_queries = set()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        while collected < args.samples:
            n = min(args.batch_size, args.samples - collected)
            try:
                batch = generate_batch(client, n)
            except Exception as e:
                print(f"Batch failed after retries, skipping: {e}", file=sys.stderr)
                continue

            for item in batch:
                query = (item.get("query") or "").strip()
                score = item.get("complexity_score")
                if not query or score is None or query in seen_queries:
                    continue
                seen_queries.add(query)
                f.write(json.dumps({"query": query, "complexity_score": float(score)}) + "\n")
                f.flush()
                collected += 1

            print(f"Collected {collected}/{args.samples}")
            time.sleep(2)  # stay comfortably under Groq free-tier rate limit

    print(f"Done. Wrote {collected} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
