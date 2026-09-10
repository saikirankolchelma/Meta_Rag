"""Shared retry-with-backoff for Groq calls, honoring the server's `retry-after`
header instead of guessing — the same TPM wall hit repeatedly during Week 1
synthetic data generation and the Week 4 shadow harness.
"""

import time

from groq import RateLimitError


def call_with_retry(fn, max_attempts: int = 6):
    """Calls fn() with no args, retrying RateLimitError up to max_attempts times."""
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (AttributeError, TypeError, ValueError):
                pass
            wait_s = (retry_after if retry_after is not None else 10) + 1
            print(f"Rate limited (attempt {attempt}/{max_attempts}), waiting {wait_s:.0f}s")
            time.sleep(wait_s)
            last_err = e
    raise last_err
