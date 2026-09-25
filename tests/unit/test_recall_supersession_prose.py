"""MCP recall: the prose of a hard-filtered (superseded) fiber is gone from EVERY section.

Complements ``test_recall_supersession.py`` (which checks the fiber list). Here the REAL
``format_context`` / ``format_context_budgeted`` build the answer, so the "Related Information"
section (top activations — the anchor of the dropped fiber is one of them) is exercised, and the
``recall_token_budget`` path, which used to format the prose BEFORE the post-filter, is covered.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval_types import (
    CoActivation,
    DepthLevel,
    RetrievalResult,
    Subgraph,
)
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig
from surreal_memory.utils.timeutils import utcnow

OSLO = "Emma lives in Oslo since 2019"
BERGEN = "Emma moved to Bergen and lives there now"


def _tms() -> dict[str, TypedMemory]:
    now = utcnow()
    old = now - timedelta(days=30)
    base = {
        "memory_type": MemoryType.FACT,
        "priority": Priority.from_int(5),
        "provenance": Provenance(source="test"),
    }
    return {
        "f-oslo": TypedMemory(
            fiber_id="f-oslo",
            created_at=old,
            valid_from=old,
            valid_until=now,
            superseded_by="f-bergen",
            **base,
        ),
        "f-bergen": TypedMemory(fiber_id="f-bergen", created_at=now, valid_from=now, **base),
    }


_NEURONS = {
    "anchor-f-oslo": Neuron.create(
        type=NeuronType.CONCEPT, content=OSLO, neuron_id="anchor-f-oslo"
    ),
    "anchor-f-bergen": Neuron.create(
        type=NeuronType.CONCEPT, content=BERGEN, neuron_id="anchor-f-bergen"
    ),
}


def _fiber(fid: str) -> MagicMock:
    f = MagicMock()
    f.id = fid
    f.anchor_neuron_id = f"anchor-{fid}"
    f.summary = None
    f.metadata = {}
    f.created_at = utcnow()
    return f


def _result() -> RetrievalResult:
    return RetrievalResult(
        answer=None,
        confidence=0.9,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=2,
        fibers_matched=["f-oslo", "f-bergen"],
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
        context=f"## Relevant Memories\n\n- {OSLO}\n- {BERGEN}",
        latency_ms=1.0,
        tokens_used=10,
        co_activations=[
            CoActivation(
                neuron_ids=frozenset({"anchor-f-oslo", "anchor-f-bergen"}),
                temporal_window_ms=0,
                binding_strength=0.8,
            )
        ],
        metadata={"activation_levels": {"anchor-f-oslo": 0.9, "anchor-f-bergen": 0.8}},
    )


def _server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as gc:
        cfg = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test-brain.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
        )
        cfg.write_gate.enabled = False
        cfg.encryption.enabled = False
        cfg.budget.system_overhead = 50
        cfg.budget.per_fiber_overhead = 10
        gc.return_value = cfg
        return MCPServer()


async def _recall(server: MCPServer, args: dict[str, Any]) -> dict[str, Any]:
    tms = _tms()
    storage = AsyncMock()
    storage.get_brain = AsyncMock(return_value=MagicMock(id="test-brain", config=MagicMock()))
    storage._current_brain_id = "test-brain"
    storage.brain_id = "test-brain"
    storage.get_typed_memory = AsyncMock(side_effect=lambda fid: tms.get(fid))
    storage.get_fiber = AsyncMock(side_effect=_fiber)
    storage.get_neuron = AsyncMock(side_effect=lambda nid: _NEURONS.get(nid))
    storage.get_neurons_batch = AsyncMock(
        side_effect=lambda ids: {i: _NEURONS[i] for i in ids if i in _NEURONS}
    )
    with (
        patch.object(server, "get_storage", return_value=storage),
        patch("surreal_memory.engine.retrieval.ReflexPipeline") as pipeline_cls,
        patch.object(server, "_check_maintenance", return_value=MagicMock(hints=())),
        patch.object(server, "_fire_eternal_trigger"),
        patch.object(server, "_record_tool_action", new_callable=AsyncMock),
        patch.object(server, "_passive_capture", new_callable=AsyncMock),
    ):
        pipeline = AsyncMock()
        pipeline.query = AsyncMock(return_value=_result())
        pipeline_cls.return_value = pipeline
        return await server.call_tool(
            "smem_recall", {"query": "where does emma live", "include_citations": False, **args}
        )


@pytest.fixture(autouse=True)
def _filter_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER", raising=False)


async def test_related_information_drops_the_superseded_anchor() -> None:
    res = await _recall(_server(), {})
    assert res["fibers_matched"] == ["f-bergen"]
    assert OSLO not in res["answer"]
    assert BERGEN in res["answer"]


async def test_recall_token_budget_prose_is_filtered_too() -> None:
    res = await _recall(_server(), {"recall_token_budget": 800})
    assert res["fibers_matched"] == ["f-bergen"]
    assert OSLO not in res["answer"]
    assert BERGEN in res["answer"]


async def test_failed_budget_pass_keeps_the_filtered_prose() -> None:
    """The budget pass is non-critical: when it raises, the FILTERED prose must remain."""
    with patch(
        "surreal_memory.engine.retrieval_context.format_context_budgeted",
        AsyncMock(side_effect=RuntimeError("budget boom")),
    ):
        res = await _recall(_server(), {"recall_token_budget": 800})
    assert res["fibers_matched"] == ["f-bergen"]
    assert OSLO not in res["answer"]


async def test_control_escape_hatch_keeps_the_old_prose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER", "1")
    res = await _recall(_server(), {})
    assert res["fibers_matched"] == ["f-oslo", "f-bergen"]
    assert OSLO in res["answer"]
