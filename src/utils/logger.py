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

# Idempotent, not a migration framework — there isn't one in this repo, and
# ADD COLUMN IF NOT EXISTS is the pragmatic equivalent for a table this small.
_ADD_EV_COLUMNS = """
ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS used_ev BOOLEAN NOT NULL DEFAULT false;
ALTER TABLE request_logs ADD COLUMN IF NOT EXISTS ev_margin DOUBLE PRECISION;
"""


def _connect():
    conn = psycopg2.connect(POSTGRES_URL)
    with conn.cursor() as cur:
        cur.execute(_CREATE_TABLE)
        cur.execute(_ADD_EV_COLUMNS)
    conn.commit()
    return conn


def log_request(query: str, answer: str, route: str, complexity_score: float,
                latency_ms: float, cost_usd: float, cache_hit: bool,
                used_ev: bool = False, ev_margin: float | None = None) -> None:
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO request_logs
                   (query, answer, route, complexity_score, latency_ms, cost_usd, cache_hit,
                    used_ev, ev_margin)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (query, answer, route, complexity_score, latency_ms, cost_usd, cache_hit,
                 used_ev, ev_margin),
            )
        conn.commit()
    finally:
        conn.close()
