"""Shared fixture helpers for the smem-higiena-autocapture live tests (W-1, W-2).

STANOWISKO: kopia bazy (pula portow 18680-18689) wylacznie; nigdy produkcja. The
tests run only when ``SURREALDB_URL`` is set, use their OWN namespace/database
(never ``uruboros``) and refuse the production ports outright -- a mistyped env
var must fail loudly instead of writing into the live brain.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

#: Ports that must never receive these tests (8000 = production SurrealDB, 8001 = its alias).
PRODUCTION_PORTS = frozenset({8000, 8001})
#: Namespace/database the tests use on the copy; deliberately not ``uruboros``/``neural_memory``.
TEST_NAMESPACE = "higiena_test"
TEST_DATABASE = "higiena_test"

SURREALDB_URL = os.getenv("SURREALDB_URL")

requires_live = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing at a COPY of the database (pool 18680-18689)",
)


def refuse_production(url: str) -> None:
    """Fail the test when ``url`` points at a production port (assertion, not skip)."""
    port = urlparse(url).port
    assert port not in PRODUCTION_PORTS, (
        f"SURREALDB_URL {url!r} points at production port {port}; "
        "these tests write data and run on a copy only"
    )


async def open_live_store(brain_name: str) -> tuple[Any, Brain]:
    """Initialise a store on the copy, in the test namespace, with a fresh brain."""
    assert SURREALDB_URL, "open_live_store requires SURREALDB_URL"
    refuse_production(SURREALDB_URL)
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL, namespace=TEST_NAMESPACE, database=TEST_DATABASE)
    await store.initialize()
    brain = Brain.create(name=brain_name)
    await store.save_brain(brain)
    store.set_brain(brain.id)
    return store, brain


async def close_live_store(store: Any, brain: Brain) -> None:
    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    finally:
        await store.close()


async def live_store_iter(brain_name: str) -> AsyncIterator[Any]:
    store, brain = await open_live_store(brain_name)
    try:
        yield store
    finally:
        await close_live_store(store, brain)


async def add_memory(
    store: Any,
    *,
    anchor_id: str,
    member_ids: list[str],
    fiber_id: str,
    pinned: bool = False,
    anchor_metadata: dict[str, Any] | None = None,
) -> Fiber:
    """Persist anchor + members, a fiber over them and its typed_memory; return the fiber."""
    anchor = Neuron.create(
        type=NeuronType.CONCEPT,
        content=f"anchor {anchor_id}",
        neuron_id=anchor_id,
        metadata=anchor_metadata or {"is_anchor": True},
    )
    await store.add_neuron(anchor)
    for mid in member_ids:
        await store.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content=f"member {mid}", neuron_id=mid)
        )
    fiber = Fiber.create(
        neuron_ids={anchor_id, *member_ids},
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        fiber_id=fiber_id,
    )
    if pinned:
        fiber = replace(fiber, pinned=True)
    await store.add_fiber(fiber)
    await store.add_typed_memory(
        TypedMemory.create(
            fiber_id=fiber_id,
            memory_type=MemoryType.CONTEXT,
            priority=Priority.NORMAL,
            source="test",
        )
    )
    return fiber
