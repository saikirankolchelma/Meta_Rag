"""Logs every gateway request to PostgreSQL for the nightly shadow-eval sampler."""

import psycopg2

from src.utils.config import POSTGRES_URL

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS request_logs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    route TEXT NOT NULL,
    complexity_score DOUBLE PRECISION NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    cost_usd DOUBLE PRECISION NOT NULL,
    cache_hit BOOLEAN NOT NULL
);
"""


def _connect():
    conn = psycopg2.connect(POSTGRES_URL)
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def log_request(query: str, answer: str, route: str, complexity_score: float,
                latency_ms: float, cost_usd: float, cache_hit: bool) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO request_logs
                   (query, answer, route, complexity_score, latency_ms, cost_usd, cache_hit)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (query, answer, route, complexity_score, latency_ms, cost_usd, cache_hit),
            )
        conn.commit()
    finally:
        conn.close()
