"""F6 prune_shadow on the path SurrealDB actually takes: the keyset ``_prune`` (live server)."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from datetime import timedelta

import pytest
import pytest_asyncio
from surrealdb.errors import NotFoundError

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="f6-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _neuron(store: SurrealDBStorage, content: str, age_days: float) -> Neuron:
    neuron = replace(
        Neuron.create(type=NeuronType.ENTITY, content=content),
        created_at=utcnow() - timedelta(days=age_days),
    )
    await store.add_neuron(neuron)
    return neuron


async def _graph(store: SurrealDBStorage) -> dict[str, str]:
    young = await _neuron(store, "f6 young orphan", 1)
    accessed = await _neuron(store, "f6 accessed orphan", 40)
    await store.update_neuron_state(NeuronState(neuron_id=accessed.id, access_frequency=3))
    old = await _neuron(store, "f6 old orphan", 40)
    a = await _neuron(store, "f6 connected a", 1)
    b = await _neuron(store, "f6 connected b", 1)
    await store.add_synapse(
        Synapse.create(source_id=a.id, target_id=b.id, type=SynapseType.RELATED_TO)
    )
    member = await _neuron(store, "f6 fiber member", 1)
    await store.add_fiber(
        Fiber.create(neuron_ids={member.id}, synapse_ids=set(), anchor_neuron_id=member.id)
    )
    return {"young": young.id, "accessed": accessed.id, "old": old.id}


async def _shadow_rows(store: SurrealDBStorage) -> list[dict[str, object]]:
    # prune_shadow is SCHEMALESS and created by its first insert; a SELECT on a table that
    # does not exist raises NotFoundError on 3.2.x. Only THAT error means "no rows yet".
    try:
        rows = await store._query(
            "SELECT neuron_id, reason FROM prune_shadow WHERE brain_id = $b",
            b=store._get_brain_id(),
        )
    except NotFoundError as exc:
        if "prune_shadow" not in str(exc):
            raise
        return []
    return list(rows or [])


async def _neuron_count(store: SurrealDBStorage) -> int:
    return len(await store.find_neurons(limit=1000))


@pytest.fixture
def keyset_only(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _legacy(*_a: object, **_k: object) -> None:
        raise AssertionError("prune fell back to _prune_legacy; this test pins the keyset path")

    monkeypatch.setattr(ConsolidationEngine, "_prune_legacy", _legacy)


async def test_dry_run_records_guarded_orphans_on_keyset_path(store, keyset_only) -> None:
    ids = await _graph(store)
    before = await _neuron_count(store)
    engine = ConsolidationEngine(store, ConsolidationConfig(prune_shadow_enabled=True))

    report = await engine.run(strategies=[ConsolidationStrategy.PRUNE], dry_run=True)

    got = {(str(r["neuron_id"]), str(r["reason"])) for r in await _shadow_rows(store)}
    assert got == {(ids["young"], "young"), (ids["accessed"], "accessed")}
    assert report.neurons_pruned == 1  # only the old, never-accessed orphan is a candidate
    assert await _neuron_count(store) == before  # dry run: nothing deleted
    assert report.extra["prune_shadow"] == {"recorded": 2, "failed": 0}


async def test_disabled_records_nothing(store, keyset_only) -> None:
    await _graph(store)
    engine = ConsolidationEngine(store, ConsolidationConfig(prune_shadow_enabled=False))
    await engine.run(strategies=[ConsolidationStrategy.PRUNE], dry_run=True)
    assert await _shadow_rows(store) == []


async def test_real_prune_keeps_the_shadowed_and_deletes_only_the_candidate(
    store, keyset_only
) -> None:
    ids = await _graph(store)
    engine = ConsolidationEngine(store, ConsolidationConfig(prune_shadow_enabled=True))
    await engine.run(strategies=[ConsolidationStrategy.PRUNE], dry_run=False)
    assert await store.get_neuron(ids["old"]) is None
    assert await store.get_neuron(ids["young"]) is not None
    assert await store.get_neuron(ids["accessed"]) is not None
    assert len(await _shadow_rows(store)) == 2
