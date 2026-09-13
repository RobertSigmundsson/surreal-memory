"""Live-DB pinning tests: the batched neuron_state lookup must not be driven by the brain index.

``get_neuron_states_batch`` runs about 14 times per recall. Its only index,
``idx_state_neuron``, is the composite UNIQUE ``(brain_id, neuron_id)``; for ``neuron_id IN $ids``
the planner can use only the ``brain_id`` prefix, which on a single-brain database selects every
row and then evaluates the IN list after decoding each one. Measured on a copy of the production
brain (18 657 rows): 85 ms per call with the index, 9.7 ms letting SurrealDB evaluate the
predicate before decoding (``pre_decode_filter``).

Same shape as ``test_surrealdb_fiber_lookup_plan.py``: the PLAN is pinned, not the clock.
Fails before the hint, passes after. Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
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
    brain = Brain.create(name="neuron-state-plan-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(store: SurrealDBStorage, n: int = 12) -> list[str]:
    """Neurons with a state row each; returns the ids we will look up."""
    ids: list[str] = []
    for i in range(n):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"state anchor {i}")
        await store.add_neuron(neuron)
        await store.update_neuron_state(NeuronState(neuron_id=neuron.id, activation_level=0.5))
        ids.append(neuron.id)
    return ids


def _capture_sql(store: SurrealDBStorage) -> list[str]:
    zebrane: list[str] = []
    oryg = store._query

    async def obs(sql: str, **params):
        zebrane.append(sql)
        return await oryg(sql, **params)

    store._query = obs  # type: ignore[method-assign]
    return zebrane


async def _plan(store: SurrealDBStorage, sql: str, **params) -> str:
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN", **params)
    return json.dumps(plan, ensure_ascii=False, default=str)


async def test_batched_state_lookup_plan_does_not_use_the_brain_index(
    store: SurrealDBStorage,
) -> None:
    ids = await _seed(store)
    zebrane = _capture_sql(store)

    await store.get_neuron_states_batch(ids[:5])

    (sql,) = [s for s in zebrane if "FROM neuron_state" in s and "IN $ids" in s]
    plan = await _plan(store, sql, brain_id=store._get_brain_id(), ids=ids[:5])

    assert "idx_state_neuron" not in plan, (
        "the batched neuron_state lookup is driven by idx_state_neuron's brain_id prefix, "
        f"which selects every row of a single-brain database. plan={plan[:400]}"
    )
    assert '"pre_decode_filter": "yes"' in plan, (
        f"the IN list is not evaluated before decoding the row. plan={plan[:400]}"
    )


async def test_the_hint_does_not_change_which_states_come_back(
    store: SurrealDBStorage,
) -> None:
    ids = await _seed(store)

    z_hintem = await store.get_neuron_states_batch(ids[:5])
    bez_hintu = await SurrealDBStorage._query(
        store,
        "SELECT neuron_id FROM neuron_state WHERE brain_id = $brain_id AND neuron_id IN $ids",
        brain_id=store._get_brain_id(),
        ids=ids[:5],
    )

    assert set(z_hintem) == {str(r["neuron_id"]) for r in bez_hintu}
    assert len(z_hintem) == 5
