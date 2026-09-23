"""Minimal HTTP shim over the recall engine for callers that are not MCP clients.

Two routes only: ``GET /health`` and ``POST /v1/recall``. Every recall goes through
``engine.recall_api.recall`` — the same function the MCP tool uses — with ``extras=None``
(no passive capture, no knowledge surface, no session context of another client) and a
forced retrieval trace carrying ``tor`` and ``agent_id``.

Deliberately a top-level module and NOT part of ``surreal_memory.server``: that package
imports ``server.app``, whose lifespan starts the consolidation and decay daemons. This
module starts no background work at all; its lifespan only drains trace tasks on shutdown.

Auth: ``Authorization: Bearer <key>`` compared with ``hmac.compare_digest`` BEFORE routing
and body parsing, so an unauthenticated request never reaches the engine (401).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from surreal_memory.engine import recall_api

logger = logging.getLogger(__name__)

TraceMode = Literal["force", "config"]
_OPEN_PATHS = frozenset({"/health"})


class RecallIn(BaseModel):
    """Request body of ``POST /v1/recall``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=5, ge=1, le=20)
    depth: int = Field(default=1, ge=0, le=3)
    max_tokens: int = Field(default=500, ge=1, le=10_000)
    session_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9._:@|-]+$")
    agent_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._:@-]+$")
    tor: str = Field(pattern=r"^http:[a-z0-9][a-z0-9-]{0,31}$")

    @field_validator("query")
    @classmethod
    def _no_lone_surrogates(cls, v: str) -> str:
        # A lone surrogate (valid JSON "\ud800") cannot be encoded to UTF-8 downstream.
        return v.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


class MemoryOut(BaseModel):
    id: str
    neuron_id: str | None
    type: str | None
    content: str
    score: float | None
    rank: int


class RecallOut(BaseModel):
    answer: str | None
    confidence: float
    sufficient: bool
    memories: list[MemoryOut]
    neurons_activated: int
    trace_id: str | None
    trace_error: str | None
    elapsed_ms: float
    engine_latency_ms: float | None
    path: str
    tor: str
    score_kind: Literal["anchor_activation"]
    rerank_degraded: bool


def _new_counters() -> dict[str, int]:
    return {"ok": 0, "401": 0, "422": 0, "503": 0, "5xx": 0}


def create_app(
    *,
    key: str,
    max_concurrency: int = 4,
    queue_timeout_s: float = 2.0,
    trace_mode: TraceMode = "force",
    reconsolidate: bool = True,
) -> FastAPI:
    """Build the shim app. ``key`` is the bearer secret (never logged)."""
    if len(key) < 32:
        raise ValueError("recall-http key must be at least 32 characters")
    key_bytes = key.encode("utf-8")
    counters = _new_counters()
    trace_tasks: set[asyncio.Task[None]] = set()
    sem = asyncio.Semaphore(max_concurrency)

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if trace_tasks:
            await asyncio.gather(*list(trace_tasks), return_exceptions=True)

    app = FastAPI(
        title="smem recall-http",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.counters = counters
    app.state.trace_tasks = trace_tasks

    @app.middleware("http")
    async def _bearer(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path in _OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        given = header[7:].encode("utf-8") if header[:7].lower() == "bearer " else b""
        if not given or not hmac.compare_digest(given, key_bytes):
            counters["401"] += 1
            return JSONResponse(
                {"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Never echo the input back: the default handler renders ``input`` (the pod's query)
        # and crashes with UnicodeEncodeError on a lone surrogate — a 422 turned into a 500.
        counters["422"] += 1
        detail = [
            {
                "loc": [str(p) for p in e.get("loc", ())],
                "msg": str(e.get("msg", "")),
                "type": e.get("type"),
            }
            for e in exc.errors()
        ]
        return JSONResponse({"error": "invalid payload", "detail": detail}, status_code=422)

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        counters["5xx"] += 1
        logger.error("recall-http: unexpected %s", type(exc).__name__, exc_info=exc)
        return JSONResponse({"error": "internal error"}, status_code=500)

    @app.get("/health")
    async def health() -> JSONResponse:
        from surreal_memory.unified_config import get_shared_storage

        try:
            storage = await asyncio.wait_for(get_shared_storage(), timeout=2.0)
            brain = await asyncio.wait_for(storage.get_brain(storage.brain_id or ""), timeout=2.0)
        except Exception as exc:
            logger.warning("recall-http: health storage check failed: %s", type(exc).__name__)
            brain = None
        if brain is None:
            return JSONResponse(
                {"status": "storage_unavailable", "liczniki": dict(counters)}, status_code=503
            )
        return JSONResponse({"status": "ok", "liczniki": dict(counters)})

    @app.post("/v1/recall", response_model=RecallOut)
    async def recall(req: RecallIn) -> Any:
        from surreal_memory.unified_config import get_config, get_shared_storage

        try:
            await asyncio.wait_for(sem.acquire(), timeout=queue_timeout_s)
        except TimeoutError:
            counters["503"] += 1
            logger.warning(
                "recall-http: busy (queue timeout %.1fs) tor=%s", queue_timeout_s, req.tor
            )
            return JSONResponse({"error": "busy"}, status_code=503)
        t0 = time.perf_counter()
        try:
            storage = await get_shared_storage()
            config = get_config()
            args: dict[str, Any] = {
                "query": req.query,
                "depth": req.depth,
                "max_tokens": req.max_tokens,
                "session_id": req.session_id,
                "reconsolidate": reconsolidate,
            }
            if trace_mode == "force":
                args["trace"] = True
            t_api = time.perf_counter()
            outcome = await recall_api.recall(
                storage,
                args,
                config=config,
                tor=req.tor,
                agent_id=req.agent_id,
                engine_session_id=f"{req.tor}|{req.session_id or req.agent_id}",
                hooks=None,
                extras=None,
                trace_tasks=trace_tasks,
            )
            if outcome.path == "error":
                msg = str(outcome.response.get("error", "error"))
                code = 503 if msg == "No brain configured" else 422
                counters["503" if code == 503 else "422"] += 1
                return JSONResponse({"error": msg}, status_code=code)
            api_ms = (time.perf_counter() - t_api) * 1000.0
            t_mat = time.perf_counter()
            memories = await recall_api.materialize_memories(
                storage, outcome.response, outcome.result, config=config, limit=req.limit
            )
            mat_ms = (time.perf_counter() - t_mat) * 1000.0
        finally:
            sem.release()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body = recall_api.response_body(outcome, memories, tor=req.tor, elapsed_ms=elapsed_ms)
        counters["ok"] += 1
        if body["trace_error"]:
            logger.warning("recall-http: trace not persisted tor=%s", req.tor)
        logger.info(
            "recall tor=%s agent=%s q=%s ms=%.0f engine=%.0f api=%.0f mat=%.0f path=%s trace=%s n_mem=%d",
            req.tor,
            req.agent_id,
            hashlib.sha256(req.query.encode("utf-8")).hexdigest()[:8],
            elapsed_ms,
            body["engine_latency_ms"] or -1.0,
            api_ms,
            mat_ms,
            outcome.path,
            outcome.trace,
            len(memories),
        )
        return body

    return app
