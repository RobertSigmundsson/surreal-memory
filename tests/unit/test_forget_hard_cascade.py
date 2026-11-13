"""W-2 (smem-higiena-autocapture): ``smem_forget(hard=true)`` removes the WHOLE memory.

Before W-2 the hard branch deleted only the ``typed_memory`` row and the fiber:
anchor, member neurons, their synapses/state and the derived rows (maturation,
backups, schedules, entity refs, co-activations) stayed behind as orphans -- the
61 anchors without a fiber and the 3 neurons of fiber ``f1270df5`` measured on
production 13.11. These tests pin the contract on the in-memory backend; the live
twin (``test_forget_hard_cascade_live.py``) pins it on SurrealDB.

MUTANT (must turn this file red): make ``delete_memory_hard`` skip the neuron
deletion (``own = []``) -- the "0 neurons" assertions fail.

Changed FROM INTENT (listed in the commit): ``test_delete_neuron_fiber_cascade.py``
(anchor delete used to tombstone the fiber, now cascades), ``test_edit_forget.py``
hard-delete tests (the handler now calls ``delete_memory_hard`` / checks anchors).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.review_schedule import ReviewSchedule
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.mcp.server import MCPServer
from surreal_memory.storage.base import HardDeleteIncompleteError, HardDeleteRefusedError
from surreal_memory.storage.memory_store import InMemoryStorage


async def _store() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="forget-hard")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


async def _neuron(
    storage: InMemoryStorage, nid: str, metadata: dict[str, Any] | None = None
) -> Neuron:
    n = Neuron.create(type=NeuronType.CONCEPT, content=f"content {nid}", neuron_id=nid)
    if metadata:
        n = Neuron.create(
            type=NeuronType.CONCEPT, content=f"content {nid}", neuron_id=nid, metadata=metadata
        )
    await storage.add_neuron(n)
    return n


async def _memory(
    storage: InMemoryStorage,
    fid: str,
    anchor: str,
    members: list[str],
    *,
    pinned: bool = False,
) -> None:
    """Fiber + typed_memory over already-created neurons."""
    fiber = Fiber.create(
        neuron_ids={anchor, *members},
        synapse_ids=set(),
        anchor_neuron_id=anchor,
        fiber_id=fid,
    )
    if pinned:
        from dataclasses import replace

        fiber = replace(fiber, pinned=True)
    await storage.add_fiber(fiber)
    await storage.add_typed_memory(
        TypedMemory.create(
            fiber_id=fid, memory_type=MemoryType.CONTEXT, priority=Priority.NORMAL, source="test"
        )
    )


def _server(storage: InMemoryStorage) -> MCPServer:
    server = MCPServer.__new__(MCPServer)
    config = MagicMock()
    config.encryption = MagicMock(enabled=False, auto_encrypt_sensitive=False)
    config.safety = MagicMock(auto_redact_min_severity=3)
    config.auto = MagicMock(enabled=False)
    config.dedup = MagicMock(enabled=False)
    config.tool_tier = MagicMock(tier="full")
    for name, value in {
        "_config": config,
        "_storage": None,
        "_hooks": None,
        "_eternal_trigger_count": 0,
        "get_storage": AsyncMock(return_value=storage),
    }.items():
        setattr(server, name, value)
    return server


async def _alive(storage: InMemoryStorage, *ids: str) -> dict[str, bool]:
    return {i: await storage.get_neuron(i) is not None for i in ids}


@pytest.mark.asyncio
async def test_forget_hard_removes_the_whole_memory_and_its_derived_rows() -> None:
    s = await _store()
    for nid in ("a1", "m1", "m2", "alias1"):
        await _neuron(s, nid, {"_dedup_alias_of": "a1"} if nid == "alias1" else None)
    await _memory(s, "f1", "a1", ["m1", "m2"])
    # alias neuron: no fiber of its own, linked to the anchor by an ALIAS edge
    await s.add_synapse(
        Synapse.create(source_id="alias1", target_id="a1", type=SynapseType.ALIAS, weight=1.0)
    )
    await s.add_synapse(
        Synapse.create(source_id="a1", target_id="m1", type=SynapseType.RELATED_TO, weight=0.5)
    )
    await s.add_entity_ref("Alibaba", "a1")
    await s.record_co_activation("m1", "m2", 0.5)
    await s.add_review_schedule(ReviewSchedule.create(fiber_id="f1", brain_id=s.brain_id or ""))
    await s.save_compression_backup("f1", "original", 1, 10, 5)

    result = await _server(s).call_tool("smem_forget", {"memory_id": "f1", "hard": True})

    assert result["status"] == "hard_deleted", result
    assert result["deleted"]["typed_memory"] == 1 and result["deleted"]["fiber"] == 1
    assert result["deleted"]["neuron"] == 4
    assert await s.get_typed_memory("f1") is None
    assert await s.get_fiber("f1") is None
    assert await _alive(s, "a1", "m1", "m2", "alias1") == dict.fromkeys(
        ("a1", "m1", "m2", "alias1"), False
    )
    assert await s.get_synapses() == [], "synapses of deleted neurons go with them"
    assert await s.count_entity_refs("Alibaba") == 0
    assert await s.get_review_schedule("f1") is None
    assert await s.get_compression_backup("f1") is None
    assert await s.get_co_activation_counts() == []
    for key in ("review_schedules", "compression_backups", "entity_refs", "co_activations"):
        assert result["deleted"][key] == 1, key


@pytest.mark.asyncio
async def test_neuron_shared_with_another_fiber_stays() -> None:
    s = await _store()
    for nid in ("a1", "own1", "shared", "a2", "own2"):
        await _neuron(s, nid)
    await _memory(s, "f1", "a1", ["own1", "shared"])
    await _memory(s, "f2", "a2", ["own2", "shared"])

    result = await s.delete_memory_hard("f1")

    assert result["neuron"] == 2, "anchor and own member only"
    assert await _alive(s, "a1", "own1", "shared", "a2", "own2") == {
        "a1": False,
        "own1": False,
        "shared": True,
        "a2": True,
        "own2": True,
    }
    f2 = await s.get_fiber("f2")
    assert f2 is not None and f2.neuron_ids == {"a2", "own2", "shared"}
    assert await s.get_typed_memory("f2") is not None


@pytest.mark.asyncio
async def test_alias_that_anchors_another_fiber_stays() -> None:
    s = await _store()
    await _neuron(s, "a1")
    await _neuron(s, "alias2", {"_dedup_alias_of": "a1"})
    await _neuron(s, "loose-alias", {"_dedup_alias_of": "a1"})
    await _memory(s, "f1", "a1", [])
    await _memory(s, "f2", "alias2", [])  # the alias IS the anchor of the second memory
    for src in ("alias2", "loose-alias"):
        await s.add_synapse(
            Synapse.create(source_id=src, target_id="a1", type=SynapseType.ALIAS, weight=1.0)
        )

    await s.delete_memory_hard("f1")

    assert await _alive(s, "a1", "alias2", "loose-alias") == {
        "a1": False,
        "alias2": True,
        "loose-alias": False,
    }
    assert await s.get_fiber("f2") is not None
    assert await s.get_typed_memory("f2") is not None


@pytest.mark.asyncio
async def test_reasoning_pattern_and_task_result_neurons_stay() -> None:
    s = await _store()
    await _neuron(s, "a1")
    await _neuron(s, "pattern", {"_reasoning_pattern_concept": True})
    await _neuron(s, "result", {"type": "task_result"})
    await _neuron(s, "plain")
    await _memory(s, "f1", "a1", ["pattern", "result", "plain"])

    await s.delete_memory_hard("f1")

    assert await _alive(s, "a1", "pattern", "result", "plain") == {
        "a1": False,
        "pattern": True,
        "result": True,
        "plain": False,
    }


@pytest.mark.asyncio
async def test_pinned_fiber_is_refused_and_nothing_is_deleted() -> None:
    s = await _store()
    await _neuron(s, "a1")
    await _neuron(s, "m1")
    await _memory(s, "f1", "a1", ["m1"], pinned=True)

    with pytest.raises(HardDeleteRefusedError):
        await s.delete_memory_hard("f1")
    result = await _server(s).call_tool("smem_forget", {"memory_id": "f1", "hard": True})

    assert "error" in result and "pinned" in result["error"].lower(), result
    assert await s.get_fiber("f1") is not None
    assert await s.get_typed_memory("f1") is not None
    assert await _alive(s, "a1", "m1") == {"a1": True, "m1": True}


@pytest.mark.asyncio
async def test_forget_hard_by_anchor_id_of_a_live_fiber_is_refused() -> None:
    s = await _store()
    await _neuron(s, "a1")
    await _neuron(s, "m1")
    await _memory(s, "f1", "a1", ["m1"])

    result = await _server(s).call_tool("smem_forget", {"memory_id": "a1", "hard": True})

    assert "error" in result, result
    assert "f1" in str(result), "the refusal must name the fiber that owns the anchor"
    assert await s.get_neuron("a1") is not None, "the anchor must be untouched"
    fiber = await s.get_fiber("f1")
    assert fiber is not None and fiber.anchor_neuron_id == "a1"


@pytest.mark.asyncio
async def test_forget_hard_by_member_neuron_id_still_deletes_just_the_neuron() -> None:
    """Zero control: the refusal is for ANCHORS only; a plain neuron is deleted as before."""
    s = await _store()
    await _neuron(s, "lonely")

    result = await _server(s).call_tool("smem_forget", {"memory_id": "lonely", "hard": True})

    assert result["status"] == "hard_deleted"
    assert await s.get_neuron("lonely") is None


@pytest.mark.asyncio
async def test_delete_neuron_of_an_anchor_cascades_to_the_whole_memory() -> None:
    s = await _store()
    for nid in ("a1", "m1", "other"):
        await _neuron(s, nid)
    await _memory(s, "f1", "a1", ["m1"])
    await _memory(s, "f9", "other", [])

    assert await s.delete_neuron("a1") is True

    assert await s.get_fiber("f1") is None
    assert await s.get_typed_memory("f1") is None
    assert await s.get_neuron("m1") is None, "own member goes with the memory"
    assert await s.get_fiber("f9") is not None and await s.get_neuron("other") is not None


@pytest.mark.asyncio
async def test_delete_neuron_of_an_anchor_shared_by_two_fibers_removes_both_memories() -> None:
    s = await _store()
    for nid in ("a1", "m1", "m2"):
        await _neuron(s, nid)
    await _memory(s, "f1", "a1", ["m1"])
    await _memory(s, "f2", "a1", ["m2"])

    assert await s.delete_neuron("a1") is True

    for fid in ("f1", "f2"):
        assert await s.get_fiber(fid) is None and await s.get_typed_memory(fid) is None
    assert await _alive(s, "m1", "m2") == {"m1": False, "m2": False}


@pytest.mark.asyncio
async def test_delete_neuron_of_the_anchor_of_a_pinned_fiber_keeps_the_pinned_memory() -> None:
    """Pinned protection beats the cascade: tombstone (old behaviour), memory survives."""
    s = await _store()
    for nid in ("a1", "m1"):
        await _neuron(s, nid)
    await _memory(s, "f1", "a1", ["m1"], pinned=True)

    assert await s.delete_neuron("a1") is True

    fiber = await s.get_fiber("f1")
    assert fiber is not None and fiber.metadata.get("_anchor_deleted")
    assert await s.get_neuron("m1") is not None
    typed = await s.get_typed_memory("f1")
    assert typed is not None and typed.expires_at is not None


@pytest.mark.asyncio
async def test_failure_midway_raises_with_the_list_of_what_was_deleted() -> None:
    s = await _store()
    for nid in ("a1", "m1"):
        await _neuron(s, nid)
    await _memory(s, "f1", "a1", ["m1"])
    real = s.delete_neuron
    calls = {"n": 0}

    async def flaky(neuron_id: str) -> bool:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return await real(neuron_id)

    s.delete_neuron = flaky  # type: ignore[method-assign]

    with pytest.raises(HardDeleteIncompleteError) as exc:
        await s.delete_memory_hard("f1")

    assert exc.value.deleted["typed_memory"] == 1 and exc.value.deleted["fiber"] == 1
    assert exc.value.deleted["neuron"] == 1
    assert isinstance(exc.value.__cause__, RuntimeError)

    result = await _server(s).call_tool("smem_forget", {"memory_id": "f1", "hard": True})
    assert "error" in result, "a vanished fiber is 'not found', never a silent success"


@pytest.mark.asyncio
async def test_handler_reports_an_incomplete_delete_as_an_error_with_counters() -> None:
    s = await _store()
    await _neuron(s, "a1")
    await _memory(s, "f1", "a1", [])

    async def boom(fiber_id: str, *, pomin_neuron: str | None = None) -> dict[str, int]:
        raise HardDeleteIncompleteError(
            fiber_id, {"typed_memory": 1, "fiber": 1}, RuntimeError("x")
        )

    s.delete_memory_hard = boom  # type: ignore[method-assign]

    result = await _server(s).call_tool("smem_forget", {"memory_id": "f1", "hard": True})

    assert "error" in result and result.get("deleted") == {"typed_memory": 1, "fiber": 1}
