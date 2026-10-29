"""Retry with exponential backoff and Retry-After support.

Small, dependency-free version of the pattern used by tenacity and httpx:
retry only what is worth retrying (transient network errors, 408/429/5xx), honor
`Retry-After`, back off exponentially with jitter, and never retry a POST blindly
unless the caller says it is idempotent.
"""

from __future__ import annotations

import random
import time

import requests

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
IDEMPOTENT = frozenset({"GET", "HEAD", "OPTIONS", "PUT", "DELETE"})


def _sleep_for(attempt: int, base: float, cap: float, response: requests.Response | None = None) -> float:
    if response is not None:
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(0.0, min(float(header), cap))
            except ValueError:
                pass
    delay = min(cap, base * (2 ** (attempt - 1)))
    return delay * (0.7 + random.random() * 0.6)  # jitter so parallel callers spread out


def request_with_retry(
    method: str,
    url: str,
    *,
    attempts: int = 3,
    base_delay: float = 0.6,
    max_delay: float = 8.0,
    idempotent: bool | None = None,
    sleep: float | None = None,
    **kwargs,
) -> tuple[requests.Response, int]:
    """Return (response, attempts_used). The last response always comes back."""
    method = method.upper()
    if idempotent is None:
        idempotent = method in IDEMPOTENT
    attempts = max(1, min(int(attempts), 6))
    last_error: Exception | None = None
    used = 0

    for attempt in range(1, attempts + 1):
        used = attempt
        try:
            response = requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            delay = _sleep_for(attempt, base_delay, max_delay)
            if sleep is not None:
                sleep += delay
            time.sleep(delay)
            continue

        retryable = response.status_code in RETRY_STATUS
        if retryable and not idempotent and method == "POST" and response.status_code != 429:
            # A 5xx on a non-idempotent write may already have landed; don't duplicate it.
            return response, used
        if retryable and attempt < attempts:
            delay = _sleep_for(attempt, base_delay, max_delay, response)
            if sleep is not None:
                sleep += delay
            time.sleep(delay)
            continue
        return response, used

    if last_error is not None:  # pragma: no cover - loop always returns or raises
        raise last_error
    raise requests.RequestException("retry loop exited without a response")
