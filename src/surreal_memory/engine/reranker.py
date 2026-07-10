"""Cross-encoder reranker — optional post-SA refinement for recall precision.

Over-fetches candidates from spreading activation, then reranks with a
cross-encoder model that scores (query, candidate) pairs for relevance.
The final score blends reranker confidence with SA activation level.

This module is entirely optional. Core recall works without it.
Install: pip install surreal-memory[reranker]
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
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
        payload = json.dumps({"model": self._model, "query": query, "documents": documents}).encode(
            "utf-8"
        )
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = urllib.request.Request(  # noqa: S310 - fixed local rerank endpoint
            f"{self._endpoint}/rerank",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:  # noqa: S310
            data = json.loads(resp.read().decode("utf-8"))
        # Accept both the OpenAI-compatible field (`relevance_score`, llamastash) and
        # the BGE-M3 service field (`score`).
        by_index = {
            int(r["index"]): float(r.get("relevance_score", r.get("score", 0.0)))
            for r in data.get("results", [])
        }
        return [by_index.get(i, float("-inf")) for i in range(len(documents))]

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
) -> dict[str, ActivationResult]:
    """Convenience function: rerank activations and return updated dict.

    Replaces activation_level with blended_score for reranked neurons.
    Non-reranked neurons (below limit) are dropped.

    ``endpoint`` selects HTTP reranking over an OpenAI-compatible ``/rerank``
    server (e.g. llamastash). When ``None``/empty, falls back to the
    ``SURREAL_MEMORY_RERANKER_ENDPOINT`` env var, then to an in-process
    sentence-transformers CrossEncoder.
    """
    resolved_endpoint = (endpoint or "").strip() or _rerank_endpoint()
    if not resolved_endpoint and not _check_cross_encoder():
        logger.debug("Reranker not available, returning activations unchanged")
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

    try:
        reranked = reranker.rerank(query, candidates, limit)
    except Exception:
        # Reranking must never break recall — fall back to the SA ordering.
        logger.warning("Reranking failed; returning activations unchanged", exc_info=True)
        return activations

    # Build new activations dict with blended scores
    from dataclasses import replace as dc_replace

    new_activations: dict[str, ActivationResult] = {}
    for rr in reranked:
        original = activations[rr.neuron_id]
        new_activations[rr.neuron_id] = dc_replace(original, activation_level=rr.blended_score)

    return new_activations
