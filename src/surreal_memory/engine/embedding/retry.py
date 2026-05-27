"""Shared retry helper for transient embedding-provider API errors.

Embedding providers (Gemini, OpenAI, OpenRouter) talk to remote APIs that
fail transiently: 429 rate limits (Gemini's free tier has a tight RPM cap and
multiple ``smem-mcp`` processes share one key), 5xx, and timeouts. Without
backoff, a single transient blip becomes a hard failure that the MCP layer
surfaces as an opaque ``-32000``. This helper retries transient errors with
exponential backoff + jitter and re-raises non-transient errors immediately.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_MAX_ATTEMPTS = 4
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 16.0
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_TRANSIENT_MARKERS = (
    "resource_exhausted",
    "unavailable",
    "deadline_exceeded",
    "rate limit",
    "timeout",
    "timed out",
    "connection",
    "temporarily",
)


def is_transient(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient, retryable API error."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _TRANSIENT_STATUS:
        return True
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_MARKERS)


async def call_with_retry(
    factory: Callable[[], Awaitable[_T]],
    *,
    provider: str = "embedding",
) -> _T:
    """Run an async API call, retrying transient failures with backoff + jitter.

    Args:
        factory: A zero-arg callable producing a *fresh* awaitable per call
            (awaitables are single-use, so the call cannot be retried directly).
        provider: Label used in log messages.

    Non-transient errors (e.g. 400/401/403/404) are re-raised immediately.
    Transient errors are retried up to ``_MAX_ATTEMPTS`` times; the last
    exception is re-raised once attempts are exhausted.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return await factory()
        except Exception as exc:
            last_exc = exc
            if attempt >= _MAX_ATTEMPTS or not is_transient(exc):
                raise
            delay = min(_BASE_DELAY_S * (2 ** (attempt - 1)), _MAX_DELAY_S)
            delay += random.uniform(0, delay * 0.25)  # jitter to de-sync clients
            logger.warning(
                "%s transient error (attempt %d/%d): %s — retrying in %.1fs",
                provider,
                attempt,
                _MAX_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc
