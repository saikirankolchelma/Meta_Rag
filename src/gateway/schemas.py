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
