from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    complexity_score: float
    latency_ms: float
    cost_usd: float
    cache_hit: bool
    # Defaults required: cache entries written before this field existed are
    # replayed via ChatResponse(**cached, ...) for up to the 24h cache TTL,
    # and would otherwise fail validation post-deploy.
    used_ev: bool = False
    ev_scores: dict[str, float] | None = None
    ev_margin: float | None = None
