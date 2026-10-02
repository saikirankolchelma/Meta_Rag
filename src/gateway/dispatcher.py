"""Routes a query to one of three retrieval+generation tiers, choosing between
the EV decision model (expected quality minus cost minus latency, estimated
from shadow-eval history) and a static two-threshold fallback when there's
no historical data to trust yet. See decision_model.py for the EV logic.

Tiers:
- naive:  Naive RAG retrieval      -> local Qwen2.5-1.5B (free, on-GPU)
- parent: Parent-doc retrieval     -> Groq qwen/qwen3.8-27b (free tier)
- hyde:   HyDE + cross-encoder rerank -> OpenAI gpt-4o (paid, hard tier)
"""

import json
import time

import torch
from groq import Groq
from openai import OpenAI
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.decision_model import choose_route_by_ev
from src.eval.routing_rules import T1_DEFAULT, T2_DEFAULT
from src.ingestion.embedder import embed_texts
from src.ingestion.vector_loader import COLLECTION_HYDE, COLLECTION_NAIVE, COLLECTION_PARENT, get_client
from src.router.predict import predict_complexity
from src.utils.cache import get_client as get_redis_client
from src.utils.config import GROQ_API_KEY, OPENAI_API_KEY
from src.utils.groq_retry import call_with_retry

REDIS_KEY_T1 = "router:threshold:T1"
REDIS_KEY_T2 = "router:threshold:T2"
REDIS_KEY_DECISION_CONFIG = "router:decision_config"


def get_thresholds() -> tuple[float, float]:
    """Reads the legacy two-key thresholds, falling back to the defaults if
    they're absent — kept standalone (dispatch() uses get_decision_config()
    instead, which reads the newer consolidated key)."""
    try:
        client = get_redis_client()
        t1 = client.get(REDIS_KEY_T1)
        t2 = client.get(REDIS_KEY_T2)
        return (
            float(t1) if t1 is not None else T1_DEFAULT,
            float(t2) if t2 is not None else T2_DEFAULT,
        )
    except Exception:
        return T1_DEFAULT, T2_DEFAULT


def get_decision_config() -> tuple[float, float, dict | None, float, float]:
    """Reads the EV decision config (thresholds, EV weights, tier profiles)
    the nightly optimizer wrote to Redis as one JSON blob, falling back to
    defaults with no profile data if it hasn't run yet or Redis errors —
    choose_route_by_ev degrades gracefully to the threshold rule in that case."""
    try:
        client = get_redis_client()
        raw = client.get(REDIS_KEY_DECISION_CONFIG)
        if raw is None:
            return T1_DEFAULT, T2_DEFAULT, None, 0.0, 0.0
        config = json.loads(raw)
        return (
            float(config["t1"]),
            float(config["t2"]),
            config["profiles"],
            float(config["lambda_cost"]),
            float(config["lambda_latency"]),
        )
    except Exception:
        return T1_DEFAULT, T2_DEFAULT, None, 0.0, 0.0

LOCAL_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
# Not groq/compound-mini: it silently proxies through llama-3.3-70b-versatile,
# which has its own 100k-tokens/day cap shared across every compound-mini call
# from this account (generation AND judging both hit the same hidden budget).
GROQ_MODEL = "qwen/qwen3.8-27b"
OPENAI_HARD_MODEL = "gpt-4o"
OPENAI_HYDE_DRAFT_MODEL = "gpt-4o-mini"  # only drafts the hypothetical doc to embed, not the final answer

# Rough blended per-1M-token pricing for cost estimation, gpt-4o hard tier only
# (local generation and Groq's free tier cost 0 by construction).
OPENAI_HARD_INPUT_PER_1M = 2.50
OPENAI_HARD_OUTPUT_PER_1M = 10.00

_local_model = None
_local_tokenizer = None
_cross_encoder = None
_qdrant = None
_openai_client = None
_groq_client = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        _qdrant = get_client()
    return _qdrant


def get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_local_model():
    global _local_model, _local_tokenizer
    if _local_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
        _local_model = (
            AutoModelForCausalLM.from_pretrained(
                LOCAL_MODEL_NAME, torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            .to(device)
            .eval()
        )
    return _local_model, _local_tokenizer


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _build_prompt(query: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return (
        "Answer the question using ONLY the context below. If the context doesn't contain "
        "the answer, say you don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )


def generate_local(query: str, contexts: list[str]) -> tuple[str, float]:
    model, tokenizer = get_local_model()
    prompt = _build_prompt(query, contexts)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), 0.0


def generate_groq(query: str, contexts: list[str]) -> tuple[str, float]:
    client = get_groq()
    prompt = _build_prompt(query, contexts)
    resp = call_with_retry(
        lambda: client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
        )
    )
    return resp.choices[0].message.content.strip(), 0.0


def generate_openai_hard(query: str, contexts: list[str]) -> tuple[str, float]:
    client = get_openai()
    prompt = _build_prompt(query, contexts)
    resp = client.chat.completions.create(
        model=OPENAI_HARD_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
    )
    usage = resp.usage
    cost = (
        usage.prompt_tokens / 1_000_000 * OPENAI_HARD_INPUT_PER_1M
        + usage.completion_tokens / 1_000_000 * OPENAI_HARD_OUTPUT_PER_1M
    )
    return resp.choices[0].message.content.strip(), cost


def retrieve_naive(query: str, top_k: int = 3) -> list[str]:
    vec = embed_texts([query])[0]
    hits = get_qdrant().query_points(collection_name=COLLECTION_NAIVE, query=vec, limit=top_k).points
    return [h.payload["text"] for h in hits]


def retrieve_parent(query: str, top_k: int = 5) -> list[str]:
    vec = embed_texts([query])[0]
    hits = get_qdrant().query_points(collection_name=COLLECTION_PARENT, query=vec, limit=top_k).points
    seen = set()
    parents = []
    for h in hits:
        parent_text = h.payload["parent_text"]
        if parent_text not in seen:
            seen.add(parent_text)
            parents.append(parent_text)
    return parents


def retrieve_hyde(query: str, top_k: int = 10, final_k: int = 3) -> list[str]:
    client = get_openai()
    hyp_resp = client.chat.completions.create(
        model=OPENAI_HYDE_DRAFT_MODEL,
        messages=[{"role": "user", "content": f"Write a short hypothetical passage that would answer: {query}"}],
        temperature=0.7,
        max_tokens=150,
    )
    hypothetical_doc = hyp_resp.choices[0].message.content.strip()
    vec = embed_texts([hypothetical_doc])[0]
    hits = get_qdrant().query_points(collection_name=COLLECTION_HYDE, query=vec, limit=top_k).points
    candidates = [h.payload["text"] for h in hits]
    if not candidates:
        return []

    reranker = get_cross_encoder()
    scores = reranker.predict([(query, c) for c in candidates])
    ranked = [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ranked[:final_k]


def dispatch(query: str, t1: float | None = None, t2: float | None = None) -> dict:
    start = time.time()
    # Built fresh per call (not a module-level dict) so monkeypatching
    # retrieve_*/generate_* as module attributes in tests actually takes
    # effect: a dict literal built once at import time would instead bind
    # permanently to whatever functions existed at that moment.
    strategies = {
        "naive": (retrieve_naive, generate_local),
        "parent": (retrieve_parent, generate_groq),
        "hyde": (retrieve_hyde, generate_openai_hard),
    }
    profiles, lambda_cost, lambda_latency = {}, 0.0, 0.0

    # Only consult Redis when the caller didn't pin both thresholds explicitly
    # — this keeps dispatch(query, t1=.., t2=..) fully deterministic (no EV,
    # plain threshold routing) regardless of what a local optimizer run has
    # pushed to Redis, which is exactly what the existing unit tests rely on.
    if t1 is None or t2 is None:
        dynamic_t1, dynamic_t2, dynamic_profiles, lambda_cost, lambda_latency = get_decision_config()
        t1 = t1 if t1 is not None else dynamic_t1
        t2 = t2 if t2 is not None else dynamic_t2
        profiles = dynamic_profiles or {}

    complexity_score = predict_complexity(query)
    # With no profile data (explicit-override callers, or a fresh deployment
    # with no shadow-eval history), choose_route_by_ev degrades to the plain
    # score_to_tier(score, t1, t2) threshold rule on its own.
    route, used_ev, ev_scores = choose_route_by_ev(complexity_score, profiles, lambda_cost, lambda_latency, t1, t2)

    retrieve_fn, generate_fn = strategies[route]
    contexts = retrieve_fn(query)
    answer, cost = generate_fn(query, contexts)

    ev_margin = None
    if ev_scores is not None:
        top_two = sorted(ev_scores.values(), reverse=True)
        ev_margin = top_two[0] - top_two[1]

    latency_ms = (time.time() - start) * 1000
    return {
        "answer": answer,
        "route": route,
        "complexity_score": complexity_score,
        "latency_ms": latency_ms,
        "cost_usd": cost,
        "used_ev": used_ev,
        "ev_scores": ev_scores,
        "ev_margin": ev_margin,
    }
