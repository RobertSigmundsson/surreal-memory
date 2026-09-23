"""Characterization snapshot of the MCP ``smem_recall`` response at f8194084.

The fixture ``fixtures/recall_response_f8194084.json`` was generated from the
pre-refactor ``RecallHandler._recall`` (commit that adds this file, BEFORE the
recall body moved to ``engine/recall_api.py``). After the move the MCP path
must produce the same response bytes, the same order of MCP side-effect calls
and the same ``ReflexPipeline.query`` arguments for every scenario.

Regenerate ONLY on the pre-refactor tree:
    SMEM_RECALL_SNAPSHOT_REGEN=1 pytest tests/unit/test_recall_response_snapshot.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine.retrieval_types import (
    DepthLevel,
    RetrievalResult,
    ScoreBreakdown,
    Subgraph,
)
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig, TraceConfig

FIXTURE = Path(__file__).parent / "fixtures" / "recall_response_f8194084.json"
REGEN = os.environ.get("SMEM_RECALL_SNAPSHOT_REGEN") == "1"

_FIBERS = {
    "f-1": SimpleNamespace(
        id="f-1",
        anchor_neuron_id="n-1",
        summary="summary one",
        tags={"b", "a"},
        metadata={},
        created_at=datetime(2026, 9, 1, 10, 0, 0),
        time_end=datetime(2026, 9, 2, 10, 0, 0),
    ),
    "f-2": SimpleNamespace(
        id="f-2",
        anchor_neuron_id="n-2",
        summary="summary two",
        tags={"c"},
        metadata={},
        created_at=datetime(2026, 9, 3, 10, 0, 0),
        time_end=None,
    ),
}
_NEURONS = {
    "n-1": SimpleNamespace(id="n-1", content="Emma lives in Bergen.", metadata={}),
    "n-2": SimpleNamespace(id="n-2", content="Bergen is rainy.", metadata={"_structure": "s"}),
    "n-9": SimpleNamespace(id="n-9", content="Emma lived in Oslo.", metadata={"_superseded": True}),
}


class _FakeStorage:
    """Strict, deterministic storage double. Every call is recorded in order."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls
        self.brain_id = "test-brain"
        self._current_brain_id = "test-brain"
        self.traces: list[Any] = []

    async def get_brain(self, brain_id: str) -> Any:
        self._calls.append(f"storage.get_brain:{brain_id}")
        return SimpleNamespace(id="test-brain", name="test-brain", config=BrainConfig())

    async def get_fiber(self, fid: str) -> Any:
        self._calls.append(f"storage.get_fiber:{fid}")
        return _FIBERS.get(fid)

    async def get_neuron(self, nid: str) -> Any:
        self._calls.append(f"storage.get_neuron:{nid}")
        return _NEURONS.get(nid)

    async def get_neurons_batch(self, ids: list[str]) -> dict[str, Any]:
        self._calls.append(f"storage.get_neurons_batch:{','.join(ids)}")
        return {i: _NEURONS.get(i) for i in ids}

    async def get_typed_memory(self, fid: str) -> Any:
        self._calls.append(f"storage.get_typed_memory:{fid}")
        return None

    async def get_expiring_memories_for_fibers(self, fiber_ids: list[str], within_days: int) -> Any:
        self._calls.append(f"storage.get_expiring:{','.join(fiber_ids)}:{within_days}")
        return []

    async def get_synapses(self, source_id: str | None = None, **kw: Any) -> list[Any]:
        self._calls.append(f"storage.get_synapses:{source_id}:{sorted(kw.items())}")
        return []

    async def add_retrieval_trace(self, trace: Any) -> str:
        self._calls.append("storage.add_retrieval_trace")
        self.traces.append(trace)
        return str(trace.id)

    def __getattr__(self, name: str) -> Any:
        # Any other storage access is recorded and answered with None (deterministic).
        if name.startswith("__"):
            raise AttributeError(name)
        calls = self.__dict__["_calls"]

        async def _stub(*_a: Any, **_k: Any) -> None:
            calls.append(f"storage.other:{name}")
            return None

        return _stub


def _result() -> RetrievalResult:
    return RetrievalResult(
        answer="Emma lives in Bergen.",
        confidence=0.8125,
        depth_used=DepthLevel.CONTEXT,
        neurons_activated=7,
        fibers_matched=["f-1", "f-2"],
        subgraph=Subgraph(neuron_ids=["n-1", "n-2"], synapse_ids=[], anchor_ids=["n-1"]),
        context="Emma lives in Bergen. Bergen is rainy.",
        latency_ms=3.0,
        tokens_used=11,
        metadata={
            "disputed_ids": ["n-9"],
            "session_topics": ["emma"],
            "session_query_count": 3,
            "activation_levels": {"n-1": 0.7, "n-2": 0.4},
        },
        score_breakdown=ScoreBreakdown(
            base_activation=0.61234,
            intersection_boost=0.1,
            freshness_boost=0.05,
            frequency_boost=0.02,
            emotional_resonance=0.0,
        ),
    )


SCENARIOS: dict[str, dict[str, Any]] = {
    "assoc_krotkie_z_sesja": {"args": {"query": "gdzie mieszka emma"}, "session": True},
    "assoc_dlugie_passive_depth": {
        "args": {
            "query": "gdzie teraz mieszka Emma i czy nadal pracuje w tym samym miejscu w Bergen",
            "depth": 2,
        }
    },
    "exact": {"args": {"query": "emma", "mode": "exact"}},
    "budzet": {"args": {"query": "emma bergen", "recall_token_budget": 300}},
    "recent_uncertainty_conflicts": {
        "args": {
            "query": "emma",
            "prefer_recent": True,
            "include_uncertainty": True,
            "warn_expiry_days": 5,
            "include_conflicts": True,
            "tier": "warm",
            "min_trust": 0.1,
        }
    },
    "trace_per_call": {"args": {"query": "emma", "trace": True}},
    "min_confidence": {"args": {"query": "emma", "min_confidence": 0.99}},
    "surface_sufficient": {"args": {"query": "emma"}, "surface": ({"answer": "S"}, None)},
    "surface_depth_override": {"args": {"query": "emma"}, "surface": (None, 3)},
    "exact_fiber": {"args": {"query": "fiber:f-1"}},
    "error_query": {"args": {"query": ""}},
    "error_depth": {"args": {"query": "emma", "depth": 9}},
}


def _server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
            trace=TraceConfig(),
        )
        cfg.write_gate.enabled = False
        cfg.auto.enabled = True
        cfg.encryption.enabled = False
        cfg.budget.system_overhead = 50
        cfg.budget.per_fiber_overhead = 10
        mock_get_config.return_value = cfg
        return MCPServer()


def _norm(obj: Any) -> Any:
    return f"<{type(obj).__name__}>"


async def _run(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    calls: list[str] = []
    server = _server()
    storage = _FakeStorage(calls)
    session = {"feature": "feat-a", "task": "task-b"} if spec.get("session") else None
    surface = spec.get("surface", (None, None))

    def rec(label: str, ret: Any = None, is_async: bool = True) -> Any:
        async def _a(*a: Any, **_k: Any) -> Any:
            calls.append(f"{label}:{a[0] if a and isinstance(a[0], str) else ''}"[:120])
            return ret

        def _s(*a: Any, **_k: Any) -> Any:
            calls.append(f"{label}:{a[0] if a and isinstance(a[0], str) else ''}"[:120])
            return ret

        return _a if is_async else _s

    async def _emit(event: Any, payload: dict[str, Any]) -> None:
        calls.append(f"hooks.emit:{getattr(event, 'value', event)}")

    pipeline_calls: list[dict[str, Any]] = []

    async def _query(**kw: Any) -> RetrievalResult:
        sid = kw.get("session_id")
        assert sid == f"mcp-{id(server)}", sid
        kw = {**kw, "session_id": "mcp-<id(server)>", "reference_time": "<now>"}
        kw["depth"] = getattr(kw["depth"], "value", kw["depth"])
        pipeline_calls.append({k: kw[k] for k in sorted(kw)})
        calls.append("pipeline.query")
        return _result()

    server.config.trace = TraceConfig()
    with (
        patch.object(server, "get_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline") as pcls,
        patch.object(server, "_get_active_session", rec("mcp.active_session", session)),
        patch.object(server, "_check_surface_depth", rec("mcp.surface_depth", surface, False)),
        patch.object(server, "_passive_capture", rec("mcp.passive_capture")),
        patch.object(server, "_fire_eternal_trigger", rec("mcp.eternal", None, False)),
        patch.object(server, "_record_tool_action", rec("mcp.record_tool_action")),
        patch.object(server, "_check_maintenance", rec("mcp.check_maintenance", "PULSE")),
        patch.object(server, "_get_maintenance_hint", rec("mcp.maint_hint", "hint-x", False)),
        patch.object(server, "get_update_hint", rec("mcp.update_hint", None, False)),
        patch.object(server, "_check_onboarding", rec("mcp.onboarding", None)),
        patch.object(server, "_surface_pending_alerts", rec("mcp.alerts", {"pending_alerts": 1})),
        patch.object(server.hooks, "emit", _emit),
    ):
        pcls.return_value = SimpleNamespace(query=_query)
        response = await server._recall(dict(spec["args"]))
        for task in list(getattr(server, "_trace_tasks", set())):
            await task

    trace_view = None
    if storage.traces:
        t = storage.traces[0]
        assert response.get("trace_id") == t.id
        response = {**response, "trace_id": "<trace-id>"}
        trace_view = {
            "query": t.query,
            "mode": t.mode,
            "fiber_ids": list(t.fiber_ids),
            "confidence": t.confidence,
            "session_id": t.session_id,
            "filters": t.filters,
            "config_snapshot": t.config_snapshot,
            "brain_id": t.brain_id,
        }
    return {
        "response": json.loads(json.dumps(response, default=_norm)),
        "response_key_order": list(response.keys()),
        "calls": calls,
        "pipeline_query": json.loads(json.dumps(pipeline_calls, default=_norm)),
        "trace": trace_view,
    }


@pytest.mark.asyncio
async def test_recall_response_snapshot_matches_f8194084() -> None:
    got = {name: await _run(name, spec) for name, spec in SCENARIOS.items()}
    if REGEN:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(got, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        pytest.skip("snapshot regenerated")
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert sorted(got) == sorted(expected)
    for name in SCENARIOS:
        assert json.dumps(got[name], ensure_ascii=False, sort_keys=False) == json.dumps(
            expected[name], ensure_ascii=False, sort_keys=False
        ), name
