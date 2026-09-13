"""Live-DB pinning tests: the neuron->fiber lookup must not be driven by the brain index.

``find_fibers(contains_neuron=...)`` is the hottest query in recall — roughly 58 calls per
``pipeline.query()``. Filtering on ``brain_id`` makes the planner choose ``idx_fiber_brain``;
on a database holding a single brain that index selects every row and then forces each one to
be decoded before ``$x IN neuron_ids`` can be evaluated. Since fibers carry a 1024-float
``fiber_vec``, decoding is the whole cost: measured 279 ms per call against 4.9 ms when the
planner is allowed to answer straight off the encoded row (``pre_decode_filter``).

These tests pin the PLAN, not the clock — a timing assertion would measure the machine. They
fail on the code before the hint (the plan reaches for ``idx_fiber_brain``) and pass after it.

Skipped unless SURREALDB_URL points at a running SurrealDB.
"""

from __future__ import annotations

import json
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
    brain = Brain.create(name="fiber-lookup-plan-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(store: SurrealDBStorage, n: int = 12) -> str:
    """A handful of fibers; returns the anchor neuron id of the one we look up."""
    target = ""
    for i in range(n):
        anchor = Neuron.create(type=NeuronType.CONCEPT, content=f"plan anchor {i}")
        await store.add_neuron(anchor)
        fiber = Fiber.create(neuron_ids={anchor.id}, synapse_ids=set(), anchor_neuron_id=anchor.id)
        await store.add_fiber(fiber)
        if i == 0:
            target = anchor.id
    return target


def _capture_sql(store: SurrealDBStorage) -> list[str]:
    """Record the SurQL `find_fibers` actually emits, without changing it."""
    zebrane: list[str] = []
    oryg = store._query

    async def obs(sql: str, **params):
        zebrane.append(sql)
        return await oryg(sql, **params)

    store._query = obs  # type: ignore[method-assign]
    return zebrane


async def _plan(store: SurrealDBStorage, sql: str, **params) -> str:
    """The query plan as JSON text.

    ``_query`` returns result *rows* and an EXPLAIN carries none, so it hands back ``[]``;
    ``_query_response`` is the one that carries the plan.
    """
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN", **params)
    return json.dumps(plan, ensure_ascii=False, default=str)


async def test_neuron_lookup_plan_does_not_use_the_brain_index(
    store: SurrealDBStorage,
) -> None:
    """The hot path must be answerable without decoding every fiber row."""
    target = await _seed(store)
    zebrane = _capture_sql(store)

    await store.find_fibers(contains_neuron=target, limit=10)

    (sql,) = [s for s in zebrane if "FROM fiber" in s and "neuron_ids" in s]
    plan = await _plan(store, sql, brain_id=store._get_brain_id(), contains_neuron=target)

    assert "idx_fiber_brain" not in plan, (
        "the neuron->fiber lookup is driven by idx_fiber_brain, which selects every row of a "
        f"single-brain database and forces a full decode of each. plan={plan[:400]}"
    )
    assert '"pre_decode_filter": "yes"' in plan, (
        f"the predicate is not evaluated before decoding the row. plan={plan[:400]}"
    )


async def test_the_hint_does_not_change_which_fibers_come_back(
    store: SurrealDBStorage,
) -> None:
    """A planner hint must be invisible in the results — same rows, hinted or not."""
    target = await _seed(store)

    z_hintem = await store.find_fibers(contains_neuron=target, limit=10)
    bez_hintu = await SurrealDBStorage._query(
        store,
        "SELECT * FROM fiber WHERE brain_id = $brain_id AND $contains_neuron IN neuron_ids "
        "LIMIT 10",
        brain_id=store._get_brain_id(),
        contains_neuron=target,
    )

    assert len(z_hintem) == 1
    assert {f.id for f in z_hintem} == {
        str(r["id"]).split(":", 1)[1].replace("_", "-") for r in bez_hintu
    }


async def test_queries_without_a_neuron_filter_keep_the_planners_choice(
    store: SurrealDBStorage,
) -> None:
    """The hint is scoped to the one predicate that benefits — not applied blanket.

    On a database with several brains `idx_fiber_brain` is the right index, and skipping it
    would scan other brains' fibers for nothing.
    """
    await _seed(store)
    zebrane = _capture_sql(store)

    await store.find_fibers(min_salience=0.0, limit=10)

    (sql,) = [s for s in zebrane if "FROM fiber" in s and "salience" in s]
    assert "WITH NOINDEX" not in sql, f"hint leaked onto a non-neuron query: {sql}"
