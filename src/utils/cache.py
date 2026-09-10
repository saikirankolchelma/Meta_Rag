"""Exact-match response cache backed by Redis.

Note: the architecture calls for a semantic cache (fuzzy-matching similar
queries via embedding distance). This ships the exact-match version first —
simpler and immediately useful — with semantic matching as a follow-up.
"""

import hashlib
import json

import redis

from src.utils.config import REDIS_URL

TTL_SECONDS = 24 * 60 * 60
_client: redis.Redis | None = None


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def _key(query: str) -> str:
    return "chat_cache:" + hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()


def get_cached_response(query: str) -> dict | None:
    raw = get_client().get(_key(query))
    return json.loads(raw) if raw else None


def set_cached_response(query: str, response: dict) -> None:
    get_client().set(_key(query), json.dumps(response), ex=TTL_SECONDS)
