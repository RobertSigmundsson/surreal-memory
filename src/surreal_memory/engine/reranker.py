"""Cross-encoder reranker — optional post-SA refinement for recall precision.

Over-fetches candidates from spreading activation, then reranks with a
cross-encoder model that scores (query, candidate) pairs for relevance.
The final score blends reranker confidence with SA activation level.

This module is entirely optional. Core recall works without it.
Install: pip install surreal-memory[reranker]
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from surreal_memory.engine.activation import ActivationResult

logger = logging.getLogger(__name__)

# Sentinel for cross-encoder availability
_CROSS_ENCODER_AVAILABLE: bool | None = None


def _check_cross_encoder() -> bool:
    """Check if sentence-transformers CrossEncoder is available."""
    global _CROSS_ENCODER_AVAILABLE
    if _CROSS_ENCODER_AVAILABLE is None:
        try:
            from sentence_transformers import CrossEncoder  # noqa: F401

            _CROSS_ENCODER_AVAILABLE = True
        except ImportError:
            _CROSS_ENCODER_AVAILABLE = False
    return _CROSS_ENCODER_AVAILABLE


# --- Connection reuse for the HTTP reranker ---------------------------------------------------
# Every rerank used to open a new connection. Through the SSH tunnel to the reranker box that costs
# ~225 ms (measured 2026-09-23: new connection ttfb 300-580 ms, reused within <=3 s 102-303 ms);
# the server (uvicorn) closes an idle connection after 5 s. KEEPALIVE_S (< 5, e.g. 4) bounds reuse; 0 (default) turns
# pooling off and restores the one-connection-per-call path unchanged. Same request, same response:
# ranking is unaffected by construction.
KEEPALIVE_S: float = float(os.environ.get("SURREAL_MEMORY_RERANK_KEEPALIVE_S", "0") or 0.0)
_LOCK_TIMEOUT_S = 1.0

# --- Response cache of a LiteLLM reranker ------------------------------------------------------
# A LiteLLM /rerank endpoint caches responses: an identical body comes back with the same response
# id in ~210 ms instead of ~400 ms (measured 2026-09-23). Timing runs that repeat a query set (A/B of
# two code versions) would then time the cache, not the code. NO_CACHE=1 asks LiteLLM to skip it
# ({"cache": {"no-cache": true}}); default 0 sends today's body unchanged. Scores are unaffected.
NO_CACHE: bool = os.environ.get("SURREAL_MEMORY_RERANK_NO_CACHE", "").strip() in (
    "1",
    "true",
    "yes",
)


class _Pooled:
    __slots__ = ("conn", "used_at")

    def __init__(self, conn: http.client.HTTPConnection, used_at: float) -> None:
        self.conn = conn
        self.used_at = used_at


_POOL: dict[str, _Pooled] = {}
_POOL_LOCKS: dict[str, threading.Lock] = {}
_POOL_GUARD = threading.Lock()
POOL_STATS: dict[str, int] = {
    "reused": 0,
    "opened": 0,
    "reconnect": 0,
    "warm_ok": 0,
    "warm_fail": 0,
    # #255 context-overflow splits, on either transport: a split is extra requests on a
    # recall's critical path, so it is counted rather than left to be inferred.
    "context_split": 0,
}


class RerankHTTPError(RuntimeError):
    """An HTTP error status from ``/rerank`` on the pooled (``http.client``) path.

    The urllib path raises ``urllib.error.HTTPError``; the pooled path used to raise a bare
    ``RuntimeError`` and dropped the body, so #255's context-overflow split could never see it.
    The body is already read (the connection stays reusable); ``str()`` keeps the old text.
    """

    def __init__(self, status: int, body: bytes) -> None:
        super().__init__(f"reranker HTTP {status}")
        self.status = status
        self.body = body


def _is_context_overflow(exc: Exception) -> bool:
    """#255's predicate on either transport: HTTP 400 whose body names the context limit."""
    if isinstance(exc, RerankHTTPError):
        detail = exc.body.lower() if exc.status == 400 else b""
    elif isinstance(exc, urllib.error.HTTPError):
        detail = exc.read().lower() if exc.code == 400 else b""
    else:
        return False
    return b"maximum context length" in detail and b"input_tokens" in detail


def _endpoint_lock(endpoint: str) -> threading.Lock:
    with _POOL_GUARD:
        return _POOL_LOCKS.setdefault(endpoint, threading.Lock())


def _new_conn(endpoint: str, timeout: float) -> http.client.HTTPConnection:
    u = urllib.parse.urlsplit(endpoint)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    POOL_STATS["opened"] += 1
    return cls(u.hostname or "127.0.0.1", u.port, timeout=timeout)


def _base_path(endpoint: str) -> str:
    return urllib.parse.urlsplit(endpoint).path.rstrip("/")


def _pooled_request(
    endpoint: str,
    method: str,
    path: str,
    body: bytes | None,
    headers: dict[str, str],
    timeout: float,
) -> tuple[int, bytes]:
    """One request on the endpoint's pooled connection (caller holds the endpoint lock).

    A reused connection that the server already closed is reopened ONCE (counted as
    `reconnect`, not as a degradation); a failure on a fresh connection propagates as before.
    """
    now = time.monotonic()
    pooled = _POOL.get(endpoint)
    reused = pooled is not None and (now - pooled.used_at) < KEEPALIVE_S
    if pooled is not None and not reused:
        pooled.conn.close()
        _POOL.pop(endpoint, None)
    conn = pooled.conn if (pooled is not None and reused) else _new_conn(endpoint, timeout)
    for attempt in (0, 1):
        try:
            conn.timeout = timeout
            conn.request(method, _base_path(endpoint) + path, body=body, headers=headers)
            resp = conn.getresponse()
            data = resp.read()
            _POOL[endpoint] = _Pooled(conn, time.monotonic())
            if reused and attempt == 0:
                POOL_STATS["reused"] += 1
            return resp.status, data
        except (
            http.client.RemoteDisconnected,
            ConnectionResetError,
            BrokenPipeError,
            http.client.CannotSendRequest,
        ):
            conn.close()
            _POOL.pop(endpoint, None)
            if not (reused and attempt == 0):
                raise
            POOL_STATS["reconnect"] += 1
            conn = _new_conn(endpoint, timeout)
        except Exception:
            conn.close()
            _POOL.pop(endpoint, None)
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def rozgrzej(endpoint: str, *, timeout: float = 5.0) -> bool:
    """Open (or keep) the pooled connection with a cheap GET <endpoint>/health before a recall needs it.

    Never raises; returns whether a response arrived. Skips when pooling is off or a fresh pooled
    connection already exists.
    """
    if (
        KEEPALIVE_S <= 0
        or not isinstance(endpoint, str)
        or not endpoint.startswith(("http://", "https://"))
    ):
        return False
    endpoint = endpoint.rstrip("/")
    lock = _endpoint_lock(endpoint)
    if not lock.acquire(timeout=0.2):
        return False
    try:
        pooled = _POOL.get(endpoint)
        if pooled is not None and (time.monotonic() - pooled.used_at) < KEEPALIVE_S:
            return True
        _pooled_request(endpoint, "GET", "/health", None, {}, timeout)
        POOL_STATS["warm_ok"] += 1
        return True
    except Exception:
        POOL_STATS["warm_fail"] += 1
        logger.debug("Reranker warm-up failed (non-critical)", exc_info=True)
        return False
    finally:
        lock.release()


def _rerank_endpoint() -> str:
    """Base URL of an OpenAI-compatible ``/rerank`` endpoint (e.g. llama.cpp /
    llamastash, ``http://127.0.0.1:11435/v1``). When set, reranking is served over
    HTTP instead of loading an in-process sentence-transformers CrossEncoder — no
    torch dependency and the model runs on the shared inference server (GPU)."""
    return os.environ.get("SURREAL_MEMORY_RERANKER_ENDPOINT", "").strip()


def reranker_available() -> bool:
    """Reranking is available when an HTTP endpoint is configured OR the local
    sentence-transformers CrossEncoder is installed."""
    return bool(_rerank_endpoint()) or _check_cross_encoder()


@dataclass(frozen=True)
class RerankedResult:
    """Result of reranking a single candidate."""

    neuron_id: str
    activation_level: float
    rerank_score: float
    blended_score: float


class CrossEncoderReranker:
    """Optional cross-encoder reranking for recall precision.

    Scores (query, candidate_content) pairs with a cross-encoder model.
    Blends reranker score with spreading activation level.

    The model is loaded lazily on first use (~300MB download for bge-reranker).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        blend_weight: float = 0.7,
        min_score: float = 0.15,
        max_candidates: int = 30,
    ) -> None:
        self._model_name = model_name
        self._blend_weight = min(max(blend_weight, 0.0), 1.0)
        self._min_score = min_score
        self._max_candidates = min(max_candidates, 100)  # hard cap
        self._model: Any = None

    def _ensure_model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, max_length=512)
            logger.info("Loaded cross-encoder model: %s", self._model_name)
            return self._model
        except ImportError:
            raise ImportError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install with: pip install surreal-memory[reranker]"
            ) from None
        except Exception:
            logger.error("Failed to load cross-encoder model: %s", self._model_name)
            raise

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        limit: int,
    ) -> list[RerankedResult]:
        """Rerank candidates by cross-encoder relevance.

        Args:
            query: The original query text.
            candidates: List of (neuron_id, content, activation_level) tuples.
            limit: Maximum results to return.

        Returns:
            Reranked results with blended scores, sorted descending.
        """
        if not candidates:
            return []

        # Cap candidates for performance
        candidates = candidates[: self._max_candidates]

        model = self._ensure_model()

        # Score (query, content) pairs
        pairs = [(query, content) for _, content, _ in candidates]
        raw_scores: list[float] = model.predict(pairs).tolist()

        # Normalize raw scores to [0, 1] range via sigmoid-like mapping
        normalized = _normalize_scores(raw_scores)

        # Blend reranker score with spreading activation level
        results: list[RerankedResult] = []
        sa_weight = 1.0 - self._blend_weight
        for (neuron_id, _, activation), norm_score, raw_score in zip(
            candidates, normalized, raw_scores, strict=True
        ):
            blended = self._blend_weight * norm_score + sa_weight * activation
            results.append(
                RerankedResult(
                    neuron_id=neuron_id,
                    activation_level=activation,
                    rerank_score=float(raw_score),
                    blended_score=blended,
                )
            )

        # Sort by blended score descending
        results.sort(key=lambda r: r.blended_score, reverse=True)

        # Filter by min_score (on normalized reranker score), with fallback
        filtered = [r for r in results if _sigmoid(r.rerank_score) >= self._min_score]
        if not filtered:
            # Fallback: return top 3 even if below threshold
            filtered = results[:3]

        return filtered[:limit]


class HttpReranker:
    """Rerank over an OpenAI-compatible ``/rerank`` endpoint (llama.cpp / llamastash).

    Scores (query, document) pairs via HTTP rather than loading a model in-process.
    llama.cpp returns raw relevance logits (unbounded, commonly negative), so the
    batch is min-max normalised *within the candidate set* — a global sigmoid would
    collapse those logits toward 0 and erase the reranker's discrimination in the
    blend. The blended score keeps the SA activation as a floor.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str,
        blend_weight: float = 0.7,
        max_candidates: int = 30,
        timeout: float = 15.0,
        api_key: str = "",
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model_name
        self._blend_weight = min(max(blend_weight, 0.0), 1.0)
        self._max_candidates = min(max_candidates, 100)
        self._timeout = timeout
        # Bearer auth for endpoints that require it (e.g. the self-hosted BGE-M3
        # /rerank service). llamastash/llama.cpp need none — an empty key sends no header.
        self._api_key = (
            api_key
            or os.environ.get("SURREAL_MEMORY_RERANKER_API_KEY", "")
            or os.environ.get("BGE_M3_API_KEY", "")
        ).strip()

    def _raw_scores(self, query: str, documents: list[str]) -> list[float]:
        body: dict[str, Any] = {"model": self._model, "query": query, "documents": documents}
        if NO_CACHE:
            body["cache"] = {"no-cache": True}
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            data = json.loads(self._post_rerank(payload, headers).decode("utf-8"))
        except (urllib.error.HTTPError, RerankHTTPError) as exc:
            # LiteLLM forwards the rerank model's 4096-token context rejection.
            # A single large stored memory can overflow it even with a small
            # max_candidates setting. Split only this specific error; an unknown
            # model, invalid auth, etc. must still surface as a degradation.
            if not _is_context_overflow(exc):
                raise
            POOL_STATS["context_split"] += 1
            if len(documents) > 1:
                middle = len(documents) // 2
                return self._raw_scores(query, documents[:middle]) + self._raw_scores(
                    query, documents[middle:]
                )
            if documents and len(documents[0]) > 256:
                content = documents[0]
                middle = len(content) // 2
                overlap = min(32, middle // 4)  # do not cut a keyword across chunks
                return [
                    max(
                        self._raw_scores(query, [content[: middle + overlap]])[0],
                        self._raw_scores(query, [content[middle - overlap :]])[0],
                    )
                ]
            raise
        # Accept both the OpenAI-compatible field (`relevance_score`, llamastash) and
        # the BGE-M3 service field (`score`).
        by_index = {
            int(r["index"]): float(r.get("relevance_score", r.get("score", 0.0)))
            for r in data.get("results", [])
        }
        return [by_index.get(i, float("-inf")) for i in range(len(documents))]

    def _post_rerank(self, payload: bytes, headers: dict[str, str]) -> bytes:
        lock = _endpoint_lock(self._endpoint) if KEEPALIVE_S > 0 else None
        if lock is not None and lock.acquire(timeout=_LOCK_TIMEOUT_S):
            try:
                status, body = _pooled_request(
                    self._endpoint, "POST", "/rerank", payload, headers, self._timeout
                )
            finally:
                lock.release()
            if status >= 400:
                raise RerankHTTPError(status, body)
            return body
        req = urllib.request.Request(  # noqa: S310 - fixed local rerank endpoint
            f"{self._endpoint}/rerank",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            return bytes(resp.read())

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str, float]],
        limit: int,
    ) -> list[RerankedResult]:
        if not candidates:
            return []
        candidates = candidates[: self._max_candidates]
        documents = [content for _, content, _ in candidates]
        raw = self._raw_scores(query, documents)
        norm = _minmax(raw)
        sa_weight = 1.0 - self._blend_weight
        results: list[RerankedResult] = []
        for (neuron_id, _, activation), norm_score, raw_score in zip(
            candidates, norm, raw, strict=True
        ):
            blended = self._blend_weight * norm_score + sa_weight * activation
            results.append(
                RerankedResult(
                    neuron_id=neuron_id,
                    activation_level=activation,
                    rerank_score=float(raw_score),
                    blended_score=blended,
                )
            )
        results.sort(key=lambda r: r.blended_score, reverse=True)
        return results[:limit]


def _minmax(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1] within the batch; missing scores map to 0."""
    finite = [s for s in scores if s != float("-inf")]
    if not finite:
        return [0.0 for _ in scores]
    lo, hi = min(finite), max(finite)
    if hi <= lo:
        return [1.0 if s != float("-inf") else 0.0 for s in scores]
    return [((s - lo) / (hi - lo)) if s != float("-inf") else 0.0 for s in scores]


def _sigmoid(x: float) -> float:
    """Sigmoid function mapping any real number to [0, 1]."""
    import math

    return 1.0 / (1.0 + math.exp(-x))


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize scores to [0, 1] using sigmoid."""
    return [_sigmoid(s) for s in scores]


_URL_RE = re.compile(r"https?://\S+")


def _degradation_reason(exc: Exception | None) -> str:
    """Describe a rerank failure without echoing anything sensitive.

    The reason travels all the way into the recall response (MCP
    ``rerank_degraded_reason`` / the CLI warning), so it must stay a diagnostic
    label rather than a raw exception dump: today the endpoint is a local
    llamastash with no credentials, but a future remote endpoint could carry a
    token in its URL and this channel would happily publish it. Strip URLs and
    cap the length.
    """
    if exc is None:
        return "unknown reranker failure"
    detail = _URL_RE.sub("<endpoint>", str(exc)).strip()
    if len(detail) > 160:
        detail = detail[:157] + "..."
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def rerank_activations(
    query: str,
    activations: dict[str, ActivationResult],
    neuron_contents: dict[str, str],
    *,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    blend_weight: float = 0.7,
    min_score: float = 0.15,
    max_candidates: int = 30,
    limit: int = 50,
    endpoint: str | None = None,
    on_degraded: Callable[[str], None] | None = None,
    on_raw_top1: Callable[[float], None] | None = None,
) -> dict[str, ActivationResult]:
    """Convenience function: rerank activations and return updated dict.

    Replaces activation_level with blended_score for reranked neurons.
    Non-reranked neurons (below limit) are dropped.

    ``endpoint`` selects HTTP reranking over an OpenAI-compatible ``/rerank``
    server (e.g. llamastash). When ``None``/empty, falls back to the
    ``SURREAL_MEMORY_RERANKER_ENDPOINT`` env var, then to an in-process
    sentence-transformers CrossEncoder.

    ``on_raw_top1``, when given, is called once with the *unnormalised*
    ``RerankedResult.rerank_score`` of the candidate ranked first by
    ``blended_score`` — the only value in this pipeline that reads the
    (query, content) pair itself rather than the shape of the activation
    landscape (measured AUC 0.9728 on golden-vs-out-of-base, smem-recall-
    brama-odmowy DIAGNOZA.md §2/§7). It is a callback in the style of
    ``on_degraded`` rather than a return-shape change: ``blended_score`` /
    the candidate ordering / the filtering below are untouched bit-for-bit.
    Not called when reranking degrades (``on_degraded`` fires instead) or
    when there are no candidates to score.
    """
    resolved_endpoint = (endpoint or "").strip() or _rerank_endpoint()
    if not resolved_endpoint and not _check_cross_encoder():
        logger.debug("Reranker not available, returning activations unchanged")
        if on_degraded is not None:
            on_degraded("no reranker configured (no endpoint and no local CrossEncoder)")
        return activations

    reranker: Any
    if resolved_endpoint:
        reranker = HttpReranker(
            endpoint=resolved_endpoint,
            model_name=model_name,
            blend_weight=blend_weight,
            max_candidates=max_candidates,
        )
    else:
        reranker = CrossEncoderReranker(
            model_name=model_name,
            blend_weight=blend_weight,
            min_score=min_score,
            max_candidates=max_candidates,
        )

    # Build candidate list from activations
    candidates: list[tuple[str, str, float]] = []
    for nid, result in activations.items():
        content = neuron_contents.get(nid, "")
        if content:
            candidates.append((nid, content, result.activation_level))

    if not candidates:
        return activations

    # Sort by activation level descending (over-fetch from top)
    candidates.sort(key=lambda c: c[2], reverse=True)

    # Reranking must never break recall, but a *silent* fall-back to the raw SA
    # ordering is worse than no reranking: the caller cannot tell the results were
    # never reranked. Retry once (the endpoint is local, so the common failure is a
    # model that has not finished loading yet), then report the degradation through
    # ``on_degraded`` so recall can surface it instead of hiding it in a log line.
    reranked = None
    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            reranked = reranker.rerank(query, candidates, limit)
            break
        except Exception as exc:  # reported via on_degraded, not swallowed
            last_error = exc
            logger.warning(
                "Reranking attempt %d/2 failed: %s", attempt, exc, exc_info=(attempt == 2)
            )

    if reranked is None:
        if on_degraded is not None:
            on_degraded(_degradation_reason(last_error))
        return activations

    if on_raw_top1 is not None and reranked:
        # ``reranked`` is sorted by blended_score descending (both
        # HttpReranker.rerank and CrossEncoderReranker.rerank do this before
        # returning) — index 0 is the query's top-1 candidate, unchanged by
        # this callback.
        on_raw_top1(reranked[0].rerank_score)

    # Build new activations dict with blended scores
    from dataclasses import replace as dc_replace

    new_activations: dict[str, ActivationResult] = {}
    for rr in reranked:
        original = activations[rr.neuron_id]
        new_activations[rr.neuron_id] = dc_replace(original, activation_level=rr.blended_score)

    return new_activations
