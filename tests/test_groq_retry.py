from unittest.mock import MagicMock

import pytest
from groq import RateLimitError

from src.utils.groq_retry import call_with_retry


def _rate_limit_error(retry_after: str = "0.01") -> RateLimitError:
    fake_response = MagicMock()
    fake_response.headers = {"retry-after": retry_after}
    return RateLimitError("rate limited", response=fake_response, body=None)


def test_call_with_retry_succeeds_after_a_transient_rate_limit(monkeypatch):
    monkeypatch.setattr("src.utils.groq_retry.time.sleep", lambda s: None)
    calls = {"count": 0}

    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise _rate_limit_error()
        return "ok"

    assert call_with_retry(flaky, max_attempts=3) == "ok"
    assert calls["count"] == 2


def test_call_with_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("src.utils.groq_retry.time.sleep", lambda s: None)
    calls = {"count": 0}

    def always_fails():
        calls["count"] += 1
        raise _rate_limit_error()

    with pytest.raises(RateLimitError):
        call_with_retry(always_fails, max_attempts=3)
    assert calls["count"] == 3


def test_call_with_retry_passes_through_non_rate_limit_errors_immediately(monkeypatch):
    monkeypatch.setattr("src.utils.groq_retry.time.sleep", lambda s: None)
    calls = {"count": 0}

    def raises_value_error():
        calls["count"] += 1
        raise ValueError("not a rate limit")

    with pytest.raises(ValueError):
        call_with_retry(raises_value_error, max_attempts=3)
    assert calls["count"] == 1  # no retry for a non-RateLimitError
