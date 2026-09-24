"""Minimal HTTP shim over the memory engine for callers that are not MCP clients.

Routes: ``GET /health``; ``POST /v1/recall`` (hermes pods) — ``engine.recall_api.recall``, the
function the MCP tool uses, with ``extras=None`` (no passive capture, no knowledge surface, no
session context of another client) and a forced retrieval trace carrying ``tor`` and ``agent_id``;
``POST /v1/recall-cli`` (thin ``smem`` client in Claude Code pods) —
``engine.cli_recall_api.recall_like_cli``, exactly the ``smem recall`` semantics of the host CLI;
``POST /v1/remember`` — ``engine.remember_api``, the ``smem remember`` write, attributed to the pod.

Deliberately a top-level module and NOT part of ``surreal_memory.server``: that package
imports ``server.app``, whose lifespan starts the consolidation and decay daemons. This
module starts no background work at all; its lifespan only drains trace tasks on shutdown.

Auth: ``Authorization: Bearer <key>`` compared with ``hmac.compare_digest`` BEFORE routing
and body parsing, so an unauthenticated request never reaches the engine (401). Two disjoint
scopes: the read key opens only the read routes, the write key only ``/v1/remember`` (any other
combination = 403). Writing is off unless the app is built with ``remember_key`` (CLI ``--zapis``);
then the read key on ``/v1/remember`` gets 403 ``zapis_wylaczony``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from surreal_memory.engine import recall_api

logger = logging.getLogger(__name__)

TraceMode = Literal["force", "config"]
Skutki = Literal["inline", "odroczone"]
_OPEN_PATHS = frozenset({"/health"})
Zakres = Literal["odczyt", "zapis"]
_ZAKRES_TRASY: dict[str, Zakres] = {
    "/v1/recall": "odczyt",
    "/v1/recall-cli": "odczyt",
    "/v1/remember": "zapis",
}
TOR_HTTP_PATTERN = r"^http:[a-z0-9][a-z0-9-]{0,31}$"
AGENT_ID_PATTERN = r"^[A-Za-z0-9._:@-]+$"
SESSION_ID_PATTERN = r"^[A-Za-z0-9._:@|-]+$"
# Memory types a pod may write (HIPOTEZA polityki D-U2.c): no boundary/instruction/preference and
# no cognitive-layer types — those steer every agent and Robert's sessions.
POD_MEMORY_TYPES = (
    "fact",
    "decision",
    "insight",
    "context",
    "error",
    "workflow",
    "reference",
    "todo",
)
PodMemoryType = Literal[
    "fact", "decision", "insight", "context", "error", "workflow", "reference", "todo"
]
Tag = Annotated[
    str, StringConstraints(min_length=1, max_length=100, pattern=r"^[^\x00-\x1f\x7f]+$")
]


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
    trace_status: str = "sync"


def _encodable(v: str) -> str:
    # A lone surrogate (valid JSON "\ud800") cannot be encoded to UTF-8 downstream.
    return v.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


class CliRecallIn(BaseModel):
    """Request body of ``POST /v1/recall-cli`` (``smem recall`` semantics)."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=4000)
    depth: int | None = Field(default=None, ge=0, le=3)  # None = QueryRouter, as the host CLI
    max_tokens: int = Field(default=500, ge=1, le=10_000)
    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    agent_id: str = Field(min_length=1, max_length=120, pattern=AGENT_ID_PATTERN)
    tor: str = Field(pattern=TOR_HTTP_PATTERN)
    session_id: str | None = Field(default=None, max_length=128, pattern=SESSION_ID_PATTERN)

    @field_validator("query")
    @classmethod
    def _no_lone_surrogates(cls, v: str) -> str:
        return _encodable(v)


class CliRecallOut(BaseModel):
    """``smem recall --json`` of the host CLI (same keys; absent keys are omitted)."""

    answer: str
    confidence: float
    depth_used: int | None = None
    neurons_activated: int
    fibers_matched: list[str] | None = None
    latency_ms: float | None = None
    below_threshold: bool | None = None
    oldest_memory_age: str | None = None
    freshness_warnings: list[str] | None = None
    rerank_degraded_warning: str | None = None
    trace_status: str
    trace_id: str | None = None
    trace_error: str | None = None


class RememberIn(BaseModel):
    """Request body of ``POST /v1/remember``."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=8000)
    type: PodMemoryType
    tags: list[Tag] = Field(default_factory=list, max_length=50)
    priority: int | None = Field(default=None, ge=0, le=10)
    agent_id: str = Field(min_length=1, max_length=120, pattern=AGENT_ID_PATTERN)
    tor: str = Field(pattern=TOR_HTTP_PATTERN)
    session_id: str | None = Field(default=None, max_length=128, pattern=SESSION_ID_PATTERN)

    @field_validator("content")
    @classmethod
    def _no_lone_surrogates(cls, v: str) -> str:
        return _encodable(v)


class RememberOut(BaseModel):
    """``smem remember --json`` of the host CLI plus shim fields."""

    message: str
    fiber_id: str
    memory_type: str
    priority: str
    neurons_created: int
    neurons_linked: int
    synapses_created: int
    expires_in_days: int | None = None
    anchor_neuron_id: str
    content_sha256: str
    oczyszczone: bool
    agent_id: str
    tor: str
    elapsed_ms: float


class BladSzczegol(BaseModel):
    loc: list[str]
    msg: str
    type: str | None = None


class BladOut(BaseModel):
    """Every error body. Never carries the input, content, a sensitive match or a header value."""

    error: str
    powod: str | None = None
    detail: list[BladSzczegol] | None = None
    typy: list[str] | None = None
    liczba: int | None = None


def _new_counters() -> dict[str, int]:
    return {
        "ok": 0,
        "401": 0,
        "422": 0,
        "503": 0,
        "5xx": 0,
        "odroczone_ok": 0,
        "odroczone_err": 0,
        "bariera_timeout": 0,
        "403": 0,
        "411": 0,
        "413": 0,
        "remember_ok": 0,
        "remember_odrzucone": 0,
        "remember_err": 0,
        "recall_cli_ok": 0,
    }


def _bledy(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {c: {"model": BladOut} for c in codes}


def create_app(
    *,
    key: str,
    remember_key: str | None = None,
    max_concurrency: int = 4,
    queue_timeout_s: float = 2.0,
    trace_mode: TraceMode = "force",
    reconsolidate: bool = True,
    skutki: Skutki = "inline",
    bariera_s: float = 10.0,
    remember_max_bytes: int = 131_072,
) -> FastAPI:
    """Build the shim app. ``key`` = read bearer, ``remember_key`` = write bearer (None = writing
    off). Secrets are never logged."""
    if len(key) < 32:
        raise ValueError("recall-http key must be at least 32 characters")
    key_bytes = key.encode("utf-8")
    zapis_bytes: bytes | None = None
    if remember_key is not None:
        if len(remember_key) < 32:
            raise ValueError("remember key must be at least 32 characters")
        zapis_bytes = remember_key.encode("utf-8")
        if hmac.compare_digest(zapis_bytes, key_bytes):
            raise ValueError("remember key must differ from the read key")
    zapis_lock = asyncio.Lock()
    counters = _new_counters()
    trace_tasks: set[asyncio.Task[None]] = set()
    # skutki="odroczone": post-answer side effects + trace of recalls already answered. The
    # barrier awaits them before the next recall reads the brain, so the sequence of states is
    # the one an inline caller produces (K2: shim ≡ MCP).
    odroczone: set[asyncio.Task[Any]] = set()
    sem = asyncio.Semaphore(max_concurrency)

    def _odroczone_koniec(task: asyncio.Task[Any]) -> None:
        odroczone.discard(task)
        if task.cancelled() or task.exception() is not None:
            counters["odroczone_err"] += 1
            exc = None if task.cancelled() else task.exception()
            logger.warning(
                "recall-http: odroczone skutki nie powiodly sie: %s",
                type(exc).__name__ if exc else "cancelled",
            )
        else:
            counters["odroczone_ok"] += 1

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if odroczone:
            await asyncio.gather(*list(odroczone), return_exceptions=True)
        if trace_tasks:
            await asyncio.gather(*list(trace_tasks), return_exceptions=True)

    app = FastAPI(
        title="smem recall-http",
        version="0.2.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )
    app.state.counters = counters
    app.state.trace_tasks = trace_tasks
    app.state.odroczone = odroczone

    def _odmowa(code: int, powod: str, path: str, zakres: str) -> JSONResponse:
        counters[str(code)] += 1
        logger.warning(
            "recall-http: %d sciezka=%s powod=%s zakres_klucza=%s", code, path, powod, zakres
        )
        return JSONResponse({"error": "forbidden", "powod": powod}, status_code=code)

    @app.middleware("http")
    async def _bearer(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path in _OPEN_PATHS:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        given = header[7:].encode("utf-8") if header[:7].lower() == "bearer " else b""
        # Both comparisons always run (constant work regardless of which key was sent).
        ok_odczyt = bool(given) and hmac.compare_digest(given, key_bytes)
        ok_zapis = (
            bool(given) and zapis_bytes is not None and hmac.compare_digest(given, zapis_bytes)
        )
        if not (ok_odczyt or ok_zapis):
            counters["401"] += 1
            return JSONResponse(
                {"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
        zakres: Zakres = "zapis" if ok_zapis else "odczyt"
        potrzebny = _ZAKRES_TRASY.get(path)
        if potrzebny == "zapis" and zapis_bytes is None:
            return _odmowa(403, "zapis_wylaczony", path, zakres)
        if potrzebny is not None and potrzebny != zakres:
            return _odmowa(403, "zakres", path, zakres)
        if path == "/v1/remember" and request.method == "POST":
            dlugosc = request.headers.get("content-length")
            if dlugosc is None:
                counters["411"] += 1
                return JSONResponse({"error": "length_required"}, status_code=411)
            try:
                za_duze = int(dlugosc) > remember_max_bytes
            except ValueError:
                za_duze = True
            if za_duze:
                counters["413"] += 1
                return JSONResponse({"error": "payload_too_large"}, status_code=413)
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
                {
                    "status": "storage_unavailable",
                    "liczniki": dict(counters),
                    "oczekujace": len(odroczone),
                },
                status_code=503,
            )
        return JSONResponse(
            {
                "status": "ok",
                "liczniki": dict(counters),
                "oczekujace": len(odroczone),
                "zapis": "wlaczony" if zapis_bytes is not None else "wylaczony",
            }
        )

    @app.post("/v1/recall", response_model=RecallOut, responses=_bledy(401, 403, 422, 503, 500))
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
        bariera_ms = 0.0
        try:
            if odroczone:
                t_b = time.perf_counter()
                _done, niedokonczone = await asyncio.wait(set(odroczone), timeout=bariera_s)
                bariera_ms = (time.perf_counter() - t_b) * 1000.0
                if niedokonczone:
                    counters["bariera_timeout"] += 1
                    logger.warning(
                        "recall-http: bariera przekroczona (%d zadan skutkow po %.1fs)",
                        len(niedokonczone),
                        bariera_s,
                    )
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
                skutki=skutki,
                dekoracje=skutki == "inline",
            )
            if outcome.pending is not None:
                odroczone.add(outcome.pending)
                outcome.pending.add_done_callback(_odroczone_koniec)
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
        body["trace_status"] = outcome.trace
        counters["ok"] += 1
        if body["trace_error"]:
            logger.warning("recall-http: trace not persisted tor=%s", req.tor)
        logger.info(
            "recall tor=%s agent=%s q=%s ms=%.0f engine=%.0f api=%.0f mat=%.0f bariera=%.0f path=%s trace=%s n_mem=%d",
            req.tor,
            req.agent_id,
            hashlib.sha256(req.query.encode("utf-8")).hexdigest()[:8],
            elapsed_ms,
            body["engine_latency_ms"] or -1.0,
            api_ms,
            mat_ms,
            bariera_ms,
            outcome.path,
            outcome.trace,
            len(memories),
        )
        return body

    async def _bariera() -> float:
        if not odroczone:
            return 0.0
        t_b = time.perf_counter()
        _done, niedokonczone = await asyncio.wait(set(odroczone), timeout=bariera_s)
        if niedokonczone:
            counters["bariera_timeout"] += 1
            logger.warning(
                "recall-http: bariera przekroczona (%d zadan skutkow po %.1fs)",
                len(niedokonczone),
                bariera_s,
            )
        return (time.perf_counter() - t_b) * 1000.0

    @app.post(
        "/v1/recall-cli",
        response_model=CliRecallOut,
        response_model_exclude_none=True,
        responses=_bledy(401, 403, 422, 503, 500),
    )
    async def recall_cli(req: CliRecallIn) -> Any:
        from surreal_memory.engine import cli_recall_api
        from surreal_memory.unified_config import get_shared_storage

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
            bariera_ms = await _bariera()
            storage = await get_shared_storage()
            brain = await storage.get_brain(storage.brain_id or "")
            if brain is None:
                counters["503"] += 1
                return JSONResponse({"error": "No brain configured"}, status_code=503)

            async def _po_zapytaniu(res: Any, depth_value: int) -> Any:
                return await cli_recall_api.persist_identified_trace(
                    storage,
                    res,
                    brain=brain,
                    query=req.query,
                    depth=depth_value,
                    max_tokens=req.max_tokens,
                    min_confidence=req.min_confidence,
                    flag=True if trace_mode == "force" else None,
                    tor=req.tor,
                    identity=lambda: (req.agent_id, req.session_id),
                )

            body, slad = await cli_recall_api.recall_like_cli(
                storage,
                brain,
                query=req.query,
                depth=req.depth,
                max_tokens=req.max_tokens,
                min_confidence=req.min_confidence,
                show_routing=False,
                show_age=True,
                po_zapytaniu=_po_zapytaniu,
            )
        finally:
            sem.release()
        counters["recall_cli_ok"] += 1
        if slad.status not in ("sync", "off", "disabled"):
            logger.warning(
                "recall-http: trace not persisted tor=%s status=%s", req.tor, slad.status
            )
        logger.info(
            "recall-cli tor=%s agent=%s q=%s ms=%.0f bariera=%.0f trace=%s n_fib=%d",
            req.tor,
            req.agent_id,
            hashlib.sha256(req.query.encode("utf-8")).hexdigest()[:8],
            (time.perf_counter() - t0) * 1000.0,
            bariera_ms,
            slad.status,
            len(body.get("fibers_matched") or []),
        )
        return body

    @app.post(
        "/v1/remember",
        response_model=RememberOut,
        response_model_exclude_none=True,
        responses=_bledy(401, 403, 411, 413, 422, 503, 500),
    )
    async def remember(req: RememberIn) -> Any:
        from surreal_memory.core.memory_types import MemoryType
        from surreal_memory.engine import remember_api
        from surreal_memory.safety.input_firewall import sanitize_explicit_content
        from surreal_memory.unified_config import get_shared_storage

        t0 = time.perf_counter()
        # Content policy BEFORE any lock: a refusal never holds a slot.
        tresc = sanitize_explicit_content(req.content)
        oczyszczone = tresc != req.content
        if not tresc:
            counters["422"] += 1
            counters["remember_odrzucone"] += 1
            return JSONResponse({"error": "empty_after_sanitize"}, status_code=422)
        try:
            chk = remember_api.check_content(tresc, force=False, redact=False)
        except remember_api.SensitiveContentError as exc:
            counters["422"] += 1
            counters["remember_odrzucone"] += 1
            logger.warning(
                "remember tor=%s agent=%s wynik=odrzucone powod=sensitive_content typy=%s",
                req.tor,
                req.agent_id,
                ",".join(exc.types),
            )
            return JSONResponse(
                {"error": "sensitive_content", "typy": exc.types, "liczba": len(exc.matches)},
                status_code=422,
            )
        mem_type = MemoryType(req.type)
        expiry_days = remember_api.resolve_expiry_days(mem_type, None, ephemeral=False)
        mem_priority, jawny = remember_api.resolve_priority(req.priority)

        try:
            await asyncio.wait_for(zapis_lock.acquire(), timeout=queue_timeout_s)
        except TimeoutError:
            counters["503"] += 1
            return JSONResponse({"error": "busy"}, status_code=503)
        try:
            try:
                await asyncio.wait_for(sem.acquire(), timeout=queue_timeout_s)
            except TimeoutError:
                counters["503"] += 1
                return JSONResponse({"error": "busy"}, status_code=503)
            try:
                bariera_ms = await _bariera()
                storage = await get_shared_storage()
                brain = await storage.get_brain(storage.brain_id or "")
                if brain is None:
                    counters["503"] += 1
                    return JSONResponse({"error": "No brain configured"}, status_code=503)
                stored_by = {"agent_id": req.agent_id, "tor": req.tor, "kanal": "recall-http"}
                if req.session_id:
                    stored_by["session_id"] = req.session_id
                try:
                    stored = await remember_api.encode_and_store(
                        storage,
                        brain.config,
                        chk.content,
                        tags=set(req.tags) if req.tags else None,
                        mem_type=mem_type,
                        mem_priority=mem_priority,
                        expiry_days=expiry_days,
                        project_id=None,
                        priority_was_explicit=jawny,
                        attribution=remember_api.Attribution(
                            source=req.tor, created_by=req.agent_id, stored_by=stored_by
                        ),
                    )
                except Exception:
                    counters["remember_err"] += 1
                    raise
            finally:
                sem.release()
        finally:
            zapis_lock.release()

        body = remember_api.response_dict(
            stored,
            content=chk.content,
            mem_type=mem_type,
            mem_priority=mem_priority,
            ephemeral=False,
            project=None,
            forced_matches=0,
        )
        sha = hashlib.sha256(chk.content.encode("utf-8")).hexdigest()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        body.update(
            {
                "anchor_neuron_id": stored.anchor_neuron_id,
                "content_sha256": sha,
                "oczyszczone": oczyszczone,
                "agent_id": req.agent_id,
                "tor": req.tor,
                "elapsed_ms": elapsed_ms,
            }
        )
        counters["remember_ok"] += 1
        logger.info(
            "remember tor=%s agent=%s sesja=%s tresc=%s dl=%d typ=%s tagi=%d fiber=%s kotwica=%s "
            "neurony=%d synapsy=%d oczyszczone=%s ms=%.0f bariera=%.0f wynik=ok",
            req.tor,
            req.agent_id,
            req.session_id or "-",
            sha[:8],
            len(chk.content),
            req.type,
            len(req.tags),
            stored.fiber_id,
            stored.anchor_neuron_id,
            stored.neurons_created,
            stored.synapses_created,
            oczyszczone,
            elapsed_ms,
            bariera_ms,
        )
        return body

    def _openapi() -> dict[str, Any]:
        cached: dict[str, Any] | None = app.openapi_schema
        if cached:
            return cached
        schema: dict[str, Any] = get_openapi(
            title=app.title, version=app.version, routes=app.routes
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer"}
        }
        for path, zakres in _ZAKRES_TRASY.items():
            for op in schema.get("paths", {}).get(path, {}).values():
                op["security"] = [{"bearer": []}]
                op["x-zakres"] = zakres
        app.openapi_schema = schema
        return schema

    app.openapi = _openapi  # type: ignore[method-assign, unused-ignore]

    return app
