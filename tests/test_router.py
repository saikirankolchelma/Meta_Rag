"""Regression test for the trained DeBERTa complexity router.

Skips if the model hasn't been trained yet (models/router/ is gitignored —
regenerate it with `python -m src.router.train`). Checks qualitative
easy < medium < hard ordering on held-out queries rather than an exact MSE
against a stored validation split, since train.py doesn't persist one.
"""

import pytest

from src.utils.config import ROUTER_MODEL_DIR

pytestmark = pytest.mark.skipif(
    not (ROUTER_MODEL_DIR / "config.json").exists(),
    reason="Router model not trained yet; run `python -m src.router.train` first.",
)

EASY_QUERIES = [
    "What is the capital of France?",
    "What year did World War 2 end?",
]
MEDIUM_QUERIES = [
    "Compare the tradeoffs between REST and GraphQL APIs for a mobile app backend.",
    "How does inflation affect bond yields in an economic downturn?",
]
HARD_QUERIES = [
    "Analyze how quantum decoherence challenges the many-worlds interpretation "
    "and what this implies for the measurement problem in cosmological contexts.",
    "What are the long-term geopolitical implications of AI-driven automation "
    "on global labor migration patterns?",
]


@pytest.fixture(scope="module")
def predict():
    from src.router.predict import predict_complexity

    return predict_complexity


def _avg_score(predict, queries):
    return sum(predict(q) for q in queries) / len(queries)


def test_scores_are_in_valid_range(predict):
    for query in EASY_QUERIES + MEDIUM_QUERIES + HARD_QUERIES:
        score = predict(query)
        assert 0.0 <= score <= 1.0


def test_easy_queries_score_lower_than_hard_queries(predict):
    assert _avg_score(predict, EASY_QUERIES) < _avg_score(predict, HARD_QUERIES)


def test_band_ordering_is_monotonic(predict):
    easy_avg = _avg_score(predict, EASY_QUERIES)
    medium_avg = _avg_score(predict, MEDIUM_QUERIES)
    hard_avg = _avg_score(predict, HARD_QUERIES)
    assert easy_avg < medium_avg < hard_avg
