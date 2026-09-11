"""Nightly shadow evaluation: for a sample of queries, brute-forces all three
retrieval+generation strategies (ignoring the router), judges every answer,
and writes results for the optimizer to consume.

Real production traffic (request_logs) is used when there's enough of it;
otherwise this falls back to sampling the synthetic router-training queries
so the pipeline is runnable and testable before real traffic accumulates.

Usage:
    python -m src.eval.shadow_harness [--samples 20]
"""

import argparse
import json
import random
import time

import psycopg2

from src.eval.judge import judge_answer
from src.gateway.dispatcher import (
    generate_groq,
    generate_local,
    generate_openai_hard,
    retrieve_hyde,
    retrieve_naive,
    retrieve_parent,
)
from src.router.predict import predict_complexity
from src.utils.config import DATA_PROCESSED_DIR, POSTGRES_URL

OUTPUT_PATH = DATA_PROCESSED_DIR / "shadow_eval_results.jsonl"
SYNTHETIC_PATH = DATA_PROCESSED_DIR / "synthetic_queries.jsonl"

STRATEGIES = {
    "naive": (retrieve_naive, generate_local),
    "parent": (retrieve_parent, generate_groq),
    "hyde": (retrieve_hyde, generate_openai_hard),
}


def sample_from_logs(n: int) -> list[str]:
    try:
        conn = psycopg2.connect(POSTGRES_URL)
        with conn.cursor() as cur:
            cur.execute(
                """SELECT query FROM request_logs
                   GROUP BY query ORDER BY MAX(created_at) DESC LIMIT %s""",
                (n,),
            )
            rows = [r[0] for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"Could not sample from request_logs: {e}")
        return []


def sample_from_synthetic(n: int) -> list[str]:
    if not SYNTHETIC_PATH.exists():
        return []
    rows = [json.loads(line) for line in SYNTHETIC_PATH.open(encoding="utf-8")]
    sample = random.sample(rows, min(n, len(rows)))
    return [r["query"] for r in sample]


def get_sample_queries(n: int) -> list[str]:
    from_logs = sample_from_logs(n)
    if len(from_logs) >= n:
        print(f"Sampled {n} queries from production logs (request_logs).")
        return from_logs

    print(
        f"Only {len(from_logs)} distinct production queries available; "
        f"supplementing with synthetic queries to reach {n} (not enough real traffic yet)."
    )
    needed = n - len(from_logs)
    return from_logs + sample_from_synthetic(needed)


def brute_force_query(query: str) -> dict:
    complexity_score = predict_complexity(query)
    result = {"query": query, "complexity_score": complexity_score, "strategies": {}}

    for name, (retrieve_fn, generate_fn) in STRATEGIES.items():
        contexts = retrieve_fn(query)
        answer, cost = generate_fn(query, contexts)
        if name == "parent":
            time.sleep(1)  # light pacing on the Groq (qwen3.8-27b) parent-tier call
        judged = judge_answer(query, answer, contexts)  # OpenAI — no Groq contention
        result["strategies"][name] = {
            "answer": answer,
            "cost_usd": cost,
            "faithfulness": judged["faithfulness"],
            "relevancy": judged["relevancy"],
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    queries = get_sample_queries(args.samples)
    if not queries:
        raise SystemExit("No queries available to evaluate (no logs, no synthetic dataset).")

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, query in enumerate(queries, start=1):
            record = brute_force_query(query)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"[{i}/{len(queries)}] score={record['complexity_score']:.3f} '{query[:60]}'")
            time.sleep(0.5)

    print(f"Done. Wrote {len(queries)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
