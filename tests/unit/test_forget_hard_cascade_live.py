"""W-2 live twin of ``test_forget_hard_cascade.py`` -- real SurrealDB, real rows.

STANOWISKO: kopia bazy (pula 18680-18689), wlasna przestrzen nazw ``higiena_test``;
skipped without ``SURREALDB_URL``, asserts (does not skip) on a production port.

Every "gone" assertion is paired with the same measurement taken BEFORE the delete
(positive control: the instrument can see the rows it later declares gone), and rows
are counted after the operation, not inferred from a status.

MUTANT (must turn this file red): ``delete_memory_hard`` that skips the neuron
deletion (``owned = []``) -- neurons, their synapses/state and the derived rows of the
anchor survive.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.review_schedule import ReviewSchedule
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.memory_stages import MaturationRecord
from surreal_memory.mcp.server import MCPServer
from surreal_memory.storage.base import HardDeleteRefusedError
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from tests.unit._higiena_live import (
    add_memory,
    close_live_store,
    open_live_store,
    requires_live,
)

pytestmark = requires_live

_FORMS = ("dashes", "underscores", "record_underscores", "record_quoted_dashes")


def _spell(dashed: str, form: str) -> str:
    return {
        "dashes": dashed,
        "underscores": _to_surreal_id(dashed),
        "record_underscores": f"neuron:{_to_surreal_id(dashed)}",
        "record_quoted_dashes": f"neuron:⟨{dashed}⟩",
    }[form]


@pytest.fixture
async def store() -> Any:
    st, brain = await open_live_store("w2-forget-hard-live")
    try:
        yield st
    finally:
        await close_live_store(st, brain)


def _server(store: Any) -> MCPServer:
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
        "get_storage": AsyncMock(return_value=store),
    }.items():
        setattr(server, name, value)
    return server


async def _count(store: Any, sql: str, **params: Any) -> int:
    rows = await store._query(sql, brain_id=store._get_brain_id(), **params)
    return int(rows[0]["c"]) if rows else 0


async def _footprint(store: Any, fid: str, neurons: list[str], anchor: str) -> dict[str, int]:
    """Rows the memory occupies, per table (every one is positive before the delete)."""
    forms = [x for n in neurons for x in (n, _to_surreal_id(n))]
    sids = [_to_surreal_id(n) for n in neurons]
    fid_forms = [fid, _to_surreal_id(fid)]
    alive = 0
    for n in neurons:
        alive += 1 if await store.get_neuron(n) is not None else 0
    synapses = 0
    for s in sids:
        synapses += await _count(
            store,
            "SELECT count() AS c FROM synapse WHERE brain_id = $brain_id"
            " AND in = type::record('neuron', $s) GROUP ALL",
            s=s,
        )
        synapses += await _count(
            store,
            "SELECT count() AS c FROM synapse WHERE brain_id = $brain_id"
            " AND out = type::record('neuron', $s) GROUP ALL",
            s=s,
        )
    return {
        "typed_memory": await _count(
            store,
            "SELECT count() AS c FROM typed_memory WHERE brain_id = $brain_id"
            " AND fiber_id IN $f GROUP ALL",
            f=fid_forms,
        ),
        "fiber": 1 if await store.get_fiber(fid) is not None else 0,
        "neuron": alive,
        "neuron_state": await _count(
            store,
            "SELECT count() AS c FROM neuron_state WHERE brain_id = $brain_id"
            " AND neuron_id IN $f GROUP ALL",
            f=forms,
        ),
        "synapse": synapses,
        "maturation": 1 if await store.get_maturation(fid) is not None else 0,
        "compression_backups": 1 if await store.get_compression_backup(fid) is not None else 0,
        "review_schedules": 1 if await store.get_review_schedule(fid) is not None else 0,
        "entity_refs": await store.count_entity_refs("Alibaba"),
        "co_activations": await _count(
            store,
            "SELECT count() AS c FROM co_activations WHERE brain_id = $brain_id"
            " AND neuron_a IN $f GROUP ALL",
            f=[anchor, _to_surreal_id(anchor), *forms],
        ),
    }


async def _seed(store: Any, *, with_alias: bool = True) -> dict[str, Any]:
    ids = {k: str(uuid4()) for k in ("anchor", "m1", "m2", "alias", "fid")}
    await add_memory(
        store, anchor_id=ids["anchor"], member_ids=[ids["m1"], ids["m2"]], fiber_id=ids["fid"]
    )
    neurons = [ids["anchor"], ids["m1"], ids["m2"]]
    if with_alias:
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content="alias",
                neuron_id=ids["alias"],
                metadata={"_dedup_alias_of": ids["anchor"], "is_anchor": False},
            )
        )
        await store.add_synapse(
            Synapse.create(
                source_id=ids["alias"],
                target_id=ids["anchor"],
                type=SynapseType.ALIAS,
                weight=1.0,
            )
        )
        neurons.append(ids["alias"])
    await store.add_synapse(
        Synapse.create(
            source_id=ids["anchor"], target_id=ids["m1"], type=SynapseType.RELATED_TO, weight=0.5
        )
    )
    await store.save_maturation(MaturationRecord(fiber_id=ids["fid"], brain_id=store.brain_id))
    await store.save_compression_backup(ids["fid"], "original", 1, 10, 5)
    await store.add_review_schedule(ReviewSchedule.create(ids["fid"], store.brain_id))
    await store.add_entity_ref("Alibaba", ids["anchor"])
    await store.record_co_activation(ids["m1"], ids["m2"], 0.5)
    ids["neurons"] = neurons  # type: ignore[assignment]
    return ids


@pytest.mark.asyncio
async def test_forget_hard_leaves_no_row_of_the_memory(store: Any) -> None:
    ids = await _seed(store)
    before = await _footprint(store, ids["fid"], ids["neurons"], ids["anchor"])
    assert all(v >= 1 for v in before.values()), f"instrument blind spot: {before}"

    result = await _server(store).call_tool("smem_forget", {"memory_id": ids["fid"], "hard": True})

    assert result["status"] == "hard_deleted", result
    after = await _footprint(store, ids["fid"], ids["neurons"], ids["anchor"])
    assert after == dict.fromkeys(before, 0), f"leftovers: {after} (counters {result['deleted']})"
    assert result["deleted"]["neuron"] == 4 and result["deleted"]["typed_memory"] == 1


@pytest.mark.asyncio
async def test_shared_member_and_anchoring_alias_stay(store: Any) -> None:
    ids = await _seed(store)
    other_anchor, other_fid = ids["alias"], str(uuid4())  # the alias anchors a second memory
    shared = str(uuid4())
    await add_memory(store, anchor_id=str(uuid4()), member_ids=[shared], fiber_id=str(uuid4()))
    f1 = await store.get_fiber(ids["fid"])
    await store.update_fiber(replace(f1, neuron_ids={*f1.neuron_ids, shared}))
    # a second fiber anchored on the alias
    from surreal_memory.core.fiber import Fiber

    await store.add_fiber(
        Fiber.create(
            neuron_ids={other_anchor},
            synapse_ids=set(),
            anchor_neuron_id=other_anchor,
            fiber_id=other_fid,
        )
    )

    await store.delete_memory_hard(ids["fid"])

    assert await store.get_neuron(shared) is not None, "a neuron another fiber lists stays"
    assert await store.get_neuron(other_anchor) is not None, "an alias that anchors a fiber stays"
    assert await store.get_fiber(other_fid) is not None
    assert (
        await store.get_neuron(ids["anchor"]) is None and await store.get_neuron(ids["m1"]) is None
    )


@pytest.mark.asyncio
async def test_pinned_fiber_is_refused_with_nothing_deleted(store: Any) -> None:
    anchor, member, fid = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=anchor, member_ids=[member], fiber_id=fid, pinned=True)

    with pytest.raises(HardDeleteRefusedError):
        await store.delete_memory_hard(fid)
    result = await _server(store).call_tool("smem_forget", {"memory_id": fid, "hard": True})

    assert "pinned" in result["error"].lower(), result
    assert await store.get_fiber(fid) is not None and await store.get_typed_memory(fid) is not None
    assert await store.get_neuron(anchor) is not None and await store.get_neuron(member) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("form", _FORMS)
async def test_forget_by_anchor_id_of_a_live_fiber_is_refused_for_every_form(
    store: Any, form: str
) -> None:
    anchor, member, fid = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=anchor, member_ids=[member], fiber_id=fid)

    result = await _server(store).call_tool(
        "smem_forget", {"memory_id": _spell(anchor, form), "hard": True}
    )

    assert "error" in result and fid in str(result["fiber_ids"]), (form, result)
    assert await store.get_neuron(anchor) is not None
    fiber = await store.get_fiber(fid)
    assert fiber is not None and fiber.anchor_neuron_id == anchor


@pytest.mark.asyncio
@pytest.mark.parametrize("form", _FORMS)
async def test_delete_neuron_of_an_anchor_cascades_for_every_form(store: Any, form: str) -> None:
    """CHANGED FROM INTENT (W-2, PROJEKT Q5 P3): upstream tombstoned the fiber; the memory
    now goes whole, so no fiber points at a dead anchor (invariant I2 = 0)."""
    anchor, member, fid = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=anchor, member_ids=[member], fiber_id=fid)

    assert await store.delete_neuron(_spell(anchor, form)) is True

    assert await store.get_fiber(fid) is None, f"form {form}: fiber outlived its anchor"
    assert await store.get_typed_memory(fid) is None
    assert await store.get_neuron(anchor) is None and await store.get_neuron(member) is None
    dangling = await _count(
        store,
        "SELECT count() AS c FROM fiber WHERE brain_id = $brain_id"
        " AND (anchor_neuron_id IN $f OR neuron_ids CONTAINSANY $f) GROUP ALL",
        f=[anchor, _to_surreal_id(anchor)],
    )
    assert dangling == 0


@pytest.mark.asyncio
async def test_anchor_shared_by_two_fibers_removes_both_memories(store: Any) -> None:
    anchor, m1, m2, f1, f2 = (str(uuid4()) for _ in range(5))
    await add_memory(store, anchor_id=anchor, member_ids=[m1], fiber_id=f1)
    from surreal_memory.core.fiber import Fiber

    await store.add_neuron(Neuron.create(type=NeuronType.CONCEPT, content="m2", neuron_id=m2))
    await store.add_fiber(
        Fiber.create(
            neuron_ids={anchor, m2}, synapse_ids=set(), anchor_neuron_id=anchor, fiber_id=f2
        )
    )

    assert await store.delete_neuron(f"neuron:{_to_surreal_id(anchor)}") is True

    for fid in (f1, f2):
        assert await store.get_fiber(fid) is None
    assert await store.get_neuron(m1) is None and await store.get_neuron(m2) is None
