"""The ``smem recall`` semantics as ONE engine function, shared by the host CLI and the shim.

``recall_like_cli`` is the body ``smem recall`` has always run: depth from ``QueryRouter`` (unless
given), ``ReflexPipeline.query`` with exactly four arguments (no engine session), the superseded
filter shared with ``recall_api`` (``engine.superseded_filter`` — only ``valid_until``, none of
``recall_api``'s expiry/trust/tier filters), the ``min_confidence`` threshold, freshness warnings and the
reranker-degradation warning — returning the dict ``smem recall --json`` prints. The recall-http
shim serves it on ``POST /v1/recall-cli`` so a thin ``smem`` client in a pod gets exactly what the
host CLI gets (same function, same arguments), while ``/v1/recall`` keeps its own semantics.

``persist_identified_trace`` writes the retrieval trace of such a recall through
``recall_api.persist_trace`` for any tor; the caller supplies the identity (the CLI resolves it
from its environment, the shim takes it from the request body). Never raises.

Kept in ``engine`` (not ``cli``): importing anything under ``cli`` pulls in the whole Typer app.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

from surreal_memory.engine import recall_api
from surreal_memory.engine.retrieval import DepthLevel, ReflexPipeline
from surreal_memory.engine.superseded_filter import filter_superseded
from surreal_memory.extraction.parser import QueryParser
from surreal_memory.extraction.router import QueryRouter
from surreal_memory.safety.freshness import evaluate_freshness, format_age
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)

MARKER: Final = "SMEM-SLAD-BLAD"

TraceStatus = Literal["sync", "sync_error", "off", "disabled", "identity_error"]


class IdentityError(ValueError):
    """Invalid caller identity. Carries the variable/field NAME and the reason, never the value."""

    def __init__(self, variable: str, reason: str) -> None:
        super().__init__(f"{variable}:{reason}")
        self.variable = variable
        self.reason = reason


@dataclass(frozen=True)
class TraceOutcome:
    status: TraceStatus
    trace_id: str | None = None
    error: str | None = None  # no identity values, no query text
    tor: str = recall_api.TOR_CLI

    def json_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {"trace_status": self.status}
        if self.trace_id is not None:
            out["trace_id"] = self.trace_id
        if self.error is not None:
            out["trace_error"] = self.error
        return out

    def stderr_line(self) -> str | None:
        if self.status in ("sync", "off", "disabled"):
            return None
        return f"{MARKER} tor={self.tor} status={self.status} powod={self.error}"


def trace_wanted(
    trace_cfg: Any, flag: bool | None, *, draw: Callable[[], float] = random.random
) -> bool:
    """``True``/``False`` flag wins; no flag = ``[trace]`` enabled + sample_rate (as persist_trace)."""
    if flag is not None:
        return flag
    if not trace_cfg.enabled:
        return False
    return not (trace_cfg.sample_rate < 1.0 and draw() >= trace_cfg.sample_rate)


async def persist_identified_trace(
    storage: Any,
    result: Any,
    *,
    brain: Any,
    query: str,
    depth: int,
    max_tokens: int,
    min_confidence: float,
    flag: bool | None,
    tor: str,
    identity: Callable[[], tuple[str, str | None]],
    config: Any | None = None,
) -> TraceOutcome:
    """Write the trace of one CLI-semantics recall. Never raises; never touches the pipeline.

    Order: flag off -> ``disabled``; ``[trace]`` missing -> ``sync_error``; not wanted -> ``off``;
    ``identity()`` raising :class:`IdentityError` -> ``identity_error`` (no trace); then the write.
    """
    if flag is False:
        return TraceOutcome("disabled", tor=tor)
    try:
        if config is None:
            from surreal_memory.unified_config import get_config

            config = get_config()
        trace_cfg = getattr(config, "trace", None)
        if trace_cfg is None:
            logger.warning("recall trace (%s): config has no [trace] section", tor)
            return TraceOutcome("sync_error", error="brak-sekcji-trace", tor=tor)
        if not trace_wanted(trace_cfg, flag):
            return TraceOutcome("off", tor=tor)
        try:
            agent_id, session_id = identity()
        except IdentityError as exc:
            logger.warning("recall trace (%s): invalid identity in %s", tor, exc.variable)
            return TraceOutcome("identity_error", error=str(exc), tor=tor)

        # The pipeline got the original query; only the trace copy is made encodable.
        q = query.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        args: dict[str, Any] = {
            "query": q,
            "depth": depth,
            "max_tokens": max_tokens,
            "session_id": session_id,
            "trace": True,
        }
        if min_confidence > 0.0:
            args["min_confidence"] = min_confidence
        sink: dict[str, Any] = {}
        st = await recall_api.persist_trace(
            sink,
            result,
            query=q,
            args=args,
            brain=brain,
            mode="associative",
            storage=storage,
            config=config,
            tor=tor,
            agent_id=agent_id,
            trace_tasks=None,
        )
        trace_id = sink.get("trace_id")
        if st == "sync" and trace_id:
            return TraceOutcome("sync", trace_id=str(trace_id), tor=tor)
        # Per-call persist returns "sync" or "sync_error"; "off" here means it raised internally.
        logger.warning("recall trace (%s) not persisted: status=%s", tor, st)
        return TraceOutcome(
            "sync_error",
            error=str(sink.get("trace_error") or f"persist_trace-status={st}"),
            tor=tor,
        )
    except Exception as exc:
        logger.warning("recall trace (%s) failed: %s", tor, type(exc).__name__)
        return TraceOutcome("sync_error", error=f"wyjatek-{type(exc).__name__}", tor=tor)


async def gather_freshness(storage: Any, fiber_ids: list[str]) -> tuple[list[str], int]:
    """Collect freshness warnings and the oldest age from matched fibers."""
    warnings: list[str] = []
    oldest_age = 0
    semaphore = asyncio.Semaphore(16)

    async def _fetch_one(fiber_id: str) -> Any:
        async with semaphore:
            return await storage.get_fiber(fiber_id)

    fibers = await asyncio.gather(*(_fetch_one(fiber_id) for fiber_id in fiber_ids))
    for fiber in fibers:
        if fiber:
            freshness = evaluate_freshness(fiber.created_at)
            if freshness.warning:
                warnings.append(freshness.warning)
            if freshness.age_days > oldest_age:
                oldest_age = freshness.age_days
    return warnings, oldest_age


TraceHook = Callable[[Any, int], Awaitable[TraceOutcome]]


async def recall_like_cli(
    storage: Any,
    brain: Any,
    *,
    query: str,
    depth: int | None,
    max_tokens: int,
    min_confidence: float,
    show_routing: bool,
    show_age: bool,
    po_zapytaniu: TraceHook,
) -> tuple[dict[str, Any], TraceOutcome]:
    """Run ``smem recall`` semantics; return (the ``--json`` dict, the trace outcome).

    ``po_zapytaniu(result, depth)`` is awaited right after the pipeline (before the threshold), so
    every pipeline execution gets exactly one trace decision.
    """
    parser = QueryParser()
    router = QueryRouter()
    stimulus = parser.parse(query, reference_time=utcnow())
    route = router.route(stimulus)

    depth_level = (
        DepthLevel(depth) if depth is not None else DepthLevel(min(route.suggested_depth, 3))
    )
    pipeline = ReflexPipeline(storage, brain.config)
    result = await pipeline.query(
        query=query,
        depth=depth_level,
        max_tokens=max_tokens,
        reference_time=utcnow(),
    )
    # Superseded facts (typed_memory.valid_until set) leave the list AND the prose, the same
    # semantics as recall_api (MCP, /v1/recall); the trace records the filtered list.
    from surreal_memory.unified_config import get_config

    filtr = await filter_superseded(
        result,
        storage,
        max_tokens=max_tokens,
        brain_id=getattr(storage, "brain_id", None) or getattr(brain, "id", "") or "",
        config=get_config(),
    )
    result = filtr.result
    slad = await po_zapytaniu(result, depth_level.value)

    if result.confidence < min_confidence:
        return {
            "answer": f"No memories found with confidence >= {min_confidence:.2f}",
            "confidence": result.confidence,
            "neurons_activated": result.neurons_activated,
            "below_threshold": True,
            **slad.json_fields(),
        }, slad

    freshness_warnings, oldest_age = await gather_freshness(storage, result.fibers_matched or [])

    response: dict[str, Any] = {
        "answer": result.context or "No relevant memories found.",
        "confidence": result.confidence,
        "depth_used": result.depth_used.value,
        "neurons_activated": result.neurons_activated,
        "fibers_matched": result.fibers_matched,
        "latency_ms": result.latency_ms,
    }
    if filtr.excluded_fiber_ids:
        response["superseded_excluded_count"] = len(filtr.excluded_fiber_ids)

    if show_routing:
        response["routing"] = {
            "query_type": route.primary.value,
            "confidence": route.confidence.name.lower(),
            "suggested_depth": route.suggested_depth,
            "use_embeddings": route.use_embeddings,
            "time_weighted": route.time_weighted,
            "signals": list(route.signals)[:5],
        }
    if show_age and oldest_age > 0:
        response["oldest_memory_age"] = format_age(oldest_age)
    if freshness_warnings:
        response["freshness_warnings"] = list(dict.fromkeys(freshness_warnings))[:3]

    # Never let reranking fail silently: raw SA ordering is indistinguishable
    # from reranked output, so say it out loud when the reranker did not run.
    rerank_degraded = (result.metadata or {}).get("rerank_degraded")
    if rerank_degraded:
        response["rerank_degraded_warning"] = (
            f"[!] Results NOT reranked (reranker enabled but unavailable): {rerank_degraded}"
        )
    response.update(slad.json_fields())
    return response, slad
