"""engine/recall_api: one recall + trace call shared by every tor (MCP, HTTP shim).

Covers what the MCP snapshot cannot: the engine-only path (``extras=None``) that the
hermes-pod shim uses, tor/agent_id validation and their landing in the trace, and the
order of MCP hooks when extras ARE given.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine import recall_api
from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.unified_config import TraceConfig


class _Storage:
    def __init__(self) -> None:
        self.brain_id = "b1"
        self._current_brain_id = "b1"
        self.traces: list[Any] = []
        self.calls: list[str] = []

    async def get_brain(self, _bid: str) -> Any:
        return SimpleNamespace(id="b1", name="b1", config=BrainConfig())

    async def get_typed_memory(self, _fid: str) -> Any:
        return None

    async def get_fiber(self, _fid: str) -> Any:
        return None

    async def add_retrieval_trace(self, trace: Any) -> str:
        self.traces.append(trace)
        return str(trace.id)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        calls = self.__dict__["calls"]

        async def _stub(*_a: Any, **_k: Any) -> None:
            calls.append(name)
            return None

        return _stub


def _config(trace: TraceConfig | None = None) -> Any:
    cfg = MagicMock()
    cfg.trace = trace or TraceConfig()
    cfg.auto.enabled = True
    cfg.encryption.enabled = False
    return cfg


def _result() -> RetrievalResult:
    return RetrievalResult(
        answer="a",
        confidence=0.5,
        depth_used=DepthLevel.CONTEXT,
        neurons_activated=3,
        fibers_matched=["f-1", "f-2"],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=["n-1"]),
        context="ctx",
        latency_ms=1.0,
    )


class _Recorder:
    def __init__(self) -> None:
        self.order: list[str] = []

    async def active_session(self, storage: Any) -> dict[str, Any] | None:
        self.order.append("active_session")
        return None

    def surface_depth(self, query: str) -> tuple[dict[str, Any] | None, int | None]:
        self.order.append("surface_depth")
        return None, None

    async def after_query(self, query: str) -> None:
        self.order.append("after_query")

    async def decorate(
        self, response: Any, result: Any, query: str, brain: Any, storage: Any
    ) -> None:
        self.order.append("decorate")


async def _call(
    args: dict[str, Any], **kw: Any
) -> tuple[recall_api.RecallOutcome, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    async def _query(**q: Any) -> RetrievalResult:
        seen.append(q)
        return _result()

    storage = kw.pop("storage", None) or _Storage()
    with patch("surreal_memory.engine.retrieval.ReflexPipeline") as pcls:
        pcls.return_value = SimpleNamespace(query=_query)
        out = await recall_api.recall(
            storage,
            args,
            config=kw.pop("config", None) or _config(),
            tor=kw.pop("tor", "http:test"),
            engine_session_id=kw.pop("engine_session_id", "http:test|s1"),
            **kw,
        )
    return out, seen


@pytest.mark.asyncio
async def test_engine_only_path_runs_pipeline_without_mcp_effects() -> None:
    storage = _Storage()
    out, seen = await _call({"query": "q", "trace": True}, storage=storage, agent_id="agent:x_1")
    assert out.path == "pipeline"
    assert out.trace == "sync"
    assert out.response["fibers_matched"] == ["f-1", "f-2"]
    assert seen[0]["session_id"] == "http:test|s1"
    # No MCP-only writes: no action events, no brain save, no passive capture.
    assert not {"record_action", "save_brain", "add_neuron", "add_fiber"} & set(storage.calls)


@pytest.mark.asyncio
async def test_tor_and_agent_id_land_in_trace() -> None:
    storage = _Storage()
    out, _ = await _call(
        {"query": "q", "trace": True},
        storage=storage,
        tor="http:hermes-pod",
        agent_id="agent:a_b_1",
    )
    assert out.response["trace_id"] == storage.traces[0].id
    t = storage.traces[0]
    assert (t.tor, t.agent_id) == ("http:hermes-pod", "agent:a_b_1")


@pytest.mark.asyncio
@pytest.mark.parametrize("tor", ["", "http", "http:", "HTTP:x", "http:bad tor", "cli", "mcp2"])
async def test_invalid_tor_is_rejected(tor: str) -> None:
    with pytest.raises(ValueError):
        await _call({"query": "q"}, tor=tor)


@pytest.mark.asyncio
async def test_too_long_agent_id_is_rejected() -> None:
    with pytest.raises(ValueError):
        await _call({"query": "q"}, agent_id="x" * (recall_api.AGENT_ID_MAX + 1))


@pytest.mark.asyncio
async def test_mcp_hooks_run_in_original_order() -> None:
    rec = _Recorder()
    out, _ = await _call({"query": "q"}, tor="mcp", extras=rec)
    assert out.path == "pipeline"
    assert rec.order == ["active_session", "surface_depth", "after_query", "decorate"]


@pytest.mark.asyncio
async def test_explicit_depth_skips_surface_routing() -> None:
    rec = _Recorder()
    await _call({"query": "q", "depth": 2}, tor="mcp", extras=rec)
    assert rec.order == ["active_session", "after_query", "decorate"]


@pytest.mark.asyncio
async def test_paths_without_pipeline_are_named_and_write_no_trace() -> None:
    storage = _Storage()
    out, seen = await _call({"query": ""}, storage=storage)
    assert (out.path, out.trace, seen) == ("error", "skipped", [])
    out, _ = await _call({"query": "q", "min_confidence": 0.99}, storage=storage)
    assert (out.path, out.trace) == ("min_confidence", "skipped")
    assert storage.traces == []


@pytest.mark.asyncio
async def test_background_trace_is_held_in_callers_task_set() -> None:
    storage = _Storage()
    tasks: set[Any] = set()
    out, _ = await _call(
        {"query": "q"},
        storage=storage,
        config=_config(TraceConfig(enabled=True, sample_rate=1.0)),
        trace_tasks=tasks,
    )
    assert out.trace == "background"
    for t in list(tasks):
        await t
    assert storage.traces and storage.traces[0].tor == "http:test"


@pytest.mark.asyncio
async def test_trace_off_by_config_reports_off() -> None:
    out, _ = await _call({"query": "q"})
    assert out.trace == "off"
    assert "trace_id" not in out.response


@pytest.mark.asyncio
async def test_reconsolidate_flag_passes_to_pipeline() -> None:
    _, seen = await _call({"query": "q", "reconsolidate": False})
    assert seen[0]["reconsolidate"] is False


@pytest.mark.asyncio
async def test_materialize_keeps_rank_of_unreadable_fiber() -> None:
    class _St:
        brain_id = "b1"

        async def get_fiber(self, fid: str) -> Any:
            return (
                None
                if fid == "f-2"
                else SimpleNamespace(anchor_neuron_id="n-" + fid, summary="s", metadata={})
            )

        async def get_neuron(self, nid: str) -> Any:
            return SimpleNamespace(content="c-" + nid, type=SimpleNamespace(value="fact"))

        async def get_typed_memory(self, _fid: str) -> Any:
            return None

    res = SimpleNamespace(metadata={"activation_levels": {"n-f-1": 0.9}})
    mem = await recall_api.materialize_memories(
        _St(), {"fibers_matched": ["f-1", "f-2", "f-3"]}, res, config=_config(), limit=10
    )
    assert [(m["id"], m["rank"]) for m in mem] == [("f-1", 1), ("f-2", 2), ("f-3", 3)]
    assert mem[1]["content"] == "" and mem[1]["neuron_id"] is None
    assert (mem[0]["score"], mem[2]["score"]) == (0.9, None)


@pytest.mark.asyncio
async def test_materialize_on_non_pipeline_response_is_empty() -> None:
    assert (
        await recall_api.materialize_memories(
            _Storage(), {"error": "x"}, None, config=_config(), limit=5
        )
        == []
    )
