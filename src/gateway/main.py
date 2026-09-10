"""FastAPI entrypoint. Run with: uvicorn src.gateway.main:app --reload --port 8000"""

import time

from fastapi import FastAPI, Response

from src.gateway.dispatcher import dispatch
from src.gateway.schemas import ChatRequest, ChatResponse
from src.utils.cache import get_cached_response, set_cached_response
from src.utils.logger import log_request

app = FastAPI(title="Meta-RAG Adaptive Gateway")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, response: Response):
    start = time.time()
    cached = get_cached_response(request.query)
    if cached is not None:
        cached = {**cached, "latency_ms": (time.time() - start) * 1000}
        response.headers["X-Route-Chosen"] = cached["route"]
        response.headers["X-Cache-Hit"] = "true"
        return ChatResponse(**cached, cache_hit=True)

    result = dispatch(request.query)
    log_request(
        query=request.query,
        answer=result["answer"],
        route=result["route"],
        complexity_score=result["complexity_score"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
        cache_hit=False,
    )
    set_cached_response(request.query, result)

    response.headers["X-Route-Chosen"] = result["route"]
    response.headers["X-Cache-Hit"] = "false"
    return ChatResponse(**result, cache_hit=False)
