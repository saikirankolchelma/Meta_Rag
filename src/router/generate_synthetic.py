"""Generates synthetic (query, complexity_score) pairs for training the DeBERTa complexity router.

Generates in balanced per-band quotas (easy/medium/hard) and rotates through a fixed
domain list, feeding recently generated queries back in as a "do not repeat" hint, to
avoid the label skew and template repetition a single unconstrained prompt produces.

Uses OpenAI `gpt-4o-mini` ($0.15/$0.60 per 1M input/output tokens) — cheap and more
than capable for this structured, low-complexity generation task. Other providers
tried and rejected for this account's free tiers:
- Groq `openai/gpt-oss-120b`: only 200,000 tokens/day free-tier quota, exhausted quickly.
- Groq `qwen/qwen3.8-27b`: ~1000 output-tokens/minute cap, forces multi-minute waits.
- Gemini `gemini-3.6-flash`: free tier caps this project at 20 requests/day for the model.

Usage:
    python -m src.router.generate_synthetic [--samples 2000] [--batch-size 15]
"""

import argparse
import json
import sys
import time
from collections import deque

from openai import OpenAI, RateLimitError

from src.utils.config import DATA_PROCESSED_DIR, OPENAI_API_KEY

MODEL = "gpt-4o-mini"
OUTPUT_PATH = DATA_PROCESSED_DIR / "synthetic_queries.jsonl"
BATCH_SLEEP_SECONDS = 2
MAX_ATTEMPTS_PER_BATCH = 6

DOMAINS = [
    "science and physics", "history", "technology and computing", "business and economics",
    "health and medicine", "everyday life and household", "law and politics", "geography",
    "arts and culture", "sports", "environment and climate", "psychology", "mathematics",
    "food and cooking", "space and astronomy", "biology and nature",
]

BANDS = {
    "easy": (0.05, 0.28),
    "medium": (0.32, 0.68),
    "hard": (0.72, 0.98),
}

BAND_DESCRIPTIONS = {
    "easy": "Single-fact lookup, direct factual question, answerable from one short passage.",
    "medium": "Requires synthesizing a few sentences of context, comparing two things, or a moderately detailed how/why question.",
    "hard": "Multi-hop reasoning, combining multiple distant pieces of context, ambiguous/open-ended, or requiring significant inference beyond the text.",
}

SYSTEM_PROMPT_TEMPLATE = """You are a dataset generator for training a query-complexity classifier used in a RAG system.

You must generate queries ONLY in the "{band}" difficulty band:
{band_description}
All complexity_score values you output MUST fall strictly within [{low}, {high}].

Respond ONLY with a JSON object of the form {{"items": [{{"query": "...", "complexity_score": 0.0}}, ...]}}. No markdown fences, no commentary."""

USER_PROMPT_TEMPLATE = """Generate {n} NEW synthetic queries in the domain of "{domain}", all within the "{band}" difficulty band.
Vary sentence structure, phrasing, and specific subject matter as much as possible.
Do NOT reuse or closely paraphrase any of these already-generated queries:
{avoid_list}"""


def generate_batch(client: OpenAI, band: str, domain: str, avoid: list[str], n: int) -> list[dict]:
    low, high = BANDS[band]
    system = SYSTEM_PROMPT_TEMPLATE.format(band=band, band_description=BAND_DESCRIPTIONS[band], low=low, high=high)
    avoid_list = "\n".join(f"- {q}" for q in avoid) if avoid else "(none yet)"
    user = USER_PROMPT_TEMPLATE.format(n=n, domain=domain, band=band, avoid_list=avoid_list)

    last_err: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS_PER_BATCH + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            items = data["items"] if isinstance(data, dict) and "items" in data else data
            if not isinstance(items, list):
                raise ValueError("Expected a JSON array (optionally under an 'items' key)")
            return items
        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (AttributeError, TypeError, ValueError):
                pass
            wait_s = (retry_after if retry_after is not None else 20) + 2
            print(f"[{band}] Rate limited (attempt {attempt}/{MAX_ATTEMPTS_PER_BATCH}), waiting {wait_s:.0f}s", file=sys.stderr)
            time.sleep(wait_s)
            last_err = e
        except Exception as e:
            time.sleep(min(2 * attempt, 20))
            last_err = e

    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY not set. Copy .env.example to .env and add your key.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    band_names = list(BANDS.keys())
    per_band_quota = args.samples // len(band_names)
    quotas = {b: per_band_quota for b in band_names}
    quotas[band_names[-1]] += args.samples - per_band_quota * len(band_names)

    collected = 0
    seen_queries = set()
    recent_by_band = {b: deque(maxlen=25) for b in band_names}
    consecutive_failures = 0
    max_consecutive_failures = 5

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for band in band_names:
            band_collected = 0
            domain_idx = 0
            low, high = BANDS[band]
            while band_collected < quotas[band]:
                n = min(args.batch_size, quotas[band] - band_collected)
                domain = DOMAINS[domain_idx % len(DOMAINS)]
                domain_idx += 1
                try:
                    batch = generate_batch(client, band, domain, list(recent_by_band[band]), n)
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    print(f"[{band}] Batch failed after retries, skipping: {e}", file=sys.stderr)
                    if consecutive_failures >= max_consecutive_failures:
                        sys.exit(
                            f"Aborting after {max_consecutive_failures} consecutive batch failures. "
                            "Check the MODEL constant is a valid, currently-served OpenAI model id."
                        )
                    continue

                for item in batch:
                    query = (item.get("query") or "").strip()
                    score = item.get("complexity_score")
                    if not query or score is None or query in seen_queries:
                        continue
                    score = min(max(float(score), low), high)  # clamp into this band's range
                    seen_queries.add(query)
                    recent_by_band[band].append(query)
                    f.write(json.dumps({"query": query, "complexity_score": score}) + "\n")
                    f.flush()
                    band_collected += 1
                    collected += 1

                print(f"[{band}] {band_collected}/{quotas[band]} (total {collected}/{args.samples})")
                time.sleep(BATCH_SLEEP_SECONDS)

    print(f"Done. Wrote {collected} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
