"""Live integration test: KNN reads omit the stored vectors but return the same ranking (smem-baza-odchudzenie-b B6).

Skipped unless SURREALDB_URL is set (live SurrealDB >= 3.2.0; see tests/integration/test_surrealdb_query_shapes.py).
Proves against a real HNSW index that (1) `find_neurons_by_embedding` / `find_fibers_by_embedding` return exactly the
ids and cosine similarities of the historical `SELECT *, vector::distance::knn()` form, (2) the returned neurons no
longer carry the vector in ``metadata["_embedding"]`` (the saving), and (3) the vector is still stored and readable
through the normal read path (nothing was lost).
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

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

_DIM = 4
_QUERY = [1.0, 0.2, 0.0, 0.0]


def _vec(i: int) -> list[float]:
    # distinct directions with a spread of cosines to the query
    return [1.0, 0.05 * i, 0.01 * (i % 7), 0.02 * (i % 5)]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
        embedding_dim=_DIM,
    )
    await storage.initialize()
    brain = Brain.create(name="knn-omit-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


@pytest.mark.timeout(180)
async def test_neuron_knn_same_ranking_without_vectors_and_vectors_still_stored(
    store: SurrealDBStorage,
) -> None:
    neurons = [
        Neuron.create(
            type=NeuronType.CONCEPT,
            content=f"memory {i}",
            neuron_id=f"knn-omit-{i:04d}",
            metadata={"_embedding": _vec(i)},
        )
        for i in range(60)
    ]
    await store.add_neurons_batch(neurons, record_change=False)

    rows = await store.find_neurons_by_embedding(_QUERY, limit=15)
    assert len(rows) == 15

    # historical form of the query, run directly against the same index
    raw = await store._query(
        "SELECT *, vector::distance::knn() AS score FROM neuron "  # noqa: S608 — brain_id: uuid z testu
        f"WHERE brain_id = '{store.brain_id}' AND embedding_vec <|15,400|> $vec",
        vec=_QUERY,
    )
    old_ids = [str(r["id"]).split(":")[-1].replace("_", "-") for r in raw]
    old_sims = [round(max(-1.0, min(1.0, 1.0 - float(r["score"]))), 9) for r in raw]
    new_ids = [n.id for n, _ in rows]
    new_sims = [round(s, 9) for _, s in rows]
    assert [i.replace("_", "-") for i in new_ids] == old_ids
    assert new_sims == old_sims

    # the saving: no vector rides along on the returned neurons
    assert all("_embedding" not in n.metadata for n, _ in rows)

    # nothing lost: the vector is still stored and readable through the normal path
    stored = await store.get_neuron(new_ids[0])
    assert stored is not None
    assert len(stored.metadata.get("_embedding") or []) == _DIM


@pytest.mark.timeout(180)
async def test_fiber_knn_same_ranking_without_vectors(store: SurrealDBStorage) -> None:
    anchors: list[str] = []
    for i in range(40):
        anchor = Neuron.create(type=NeuronType.CONCEPT, content=f"anchor {i}")
        await store.add_neuron(anchor)
        fiber = Fiber.create(neuron_ids={anchor.id}, synapse_ids=set(), anchor_neuron_id=anchor.id)
        await store.add_fiber(fiber)
        await store.update_fiber_embeddings([(fiber.id, _vec(i))])
        anchors.append(anchor.id)

    rows = await store.find_fibers_by_embedding(_QUERY, limit=10)
    assert len(rows) == 10
    raw = await store._query(
        "SELECT *, vector::distance::knn() AS score FROM fiber "  # noqa: S608 — brain_id: uuid z testu
        f"WHERE brain_id = '{store.brain_id}' AND fiber_vec <|10,400|> $vec",
        vec=_QUERY,
    )
    old = [
        (str(r["anchor_neuron_id"]), round(max(-1.0, min(1.0, 1.0 - float(r["score"]))), 9))
        for r in raw
    ]
    new = [(f.anchor_neuron_id, round(s, 9)) for f, s in rows]
    assert new == old
