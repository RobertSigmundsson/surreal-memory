"""Regression: find_neurons(time_range=...) must select by value, not by type rank.

`created_at` is a datetime column, but the query bound `$time_start`/`$time_end` as
ISO strings. SurrealDB resolves a comparison between two different types by type
rank rather than by value, so `created_at >= $time_start AND created_at <= $time_end`
became a constant predicate and the filter matched nothing at all - including rows
squarely inside the window.

The in-memory backend compares Python objects and so cannot reproduce this; only a
live engine can. The test is skipped when SURREALDB_URL is unset so CI without
docker still passes.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="time-range-filter-live")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_time_range_returns_rows_inside_the_window(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """A window around now returns the neurons created now."""
    now = utcnow()
    for i in range(3):
        await surrealdb_storage.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content=f"time-range probe {i}")
        )

    found = await surrealdb_storage.find_neurons(
        time_range=(now - timedelta(hours=1), now + timedelta(hours=1)), limit=100
    )
    assert len(found) == 3, "a window containing the rows returned nothing"


@pytest.mark.asyncio
async def test_time_range_excludes_rows_outside_the_window(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """Windows in the far past and far future match nothing - the filter narrows."""
    now = utcnow()
    await surrealdb_storage.add_neuron(
        Neuron.create(type=NeuronType.CONCEPT, content="time-range probe outside")
    )

    future_start = now + timedelta(days=1)
    future = await surrealdb_storage.find_neurons(
        time_range=(future_start, future_start + timedelta(days=1)), limit=100
    )
    assert future == []

    past_start = now - timedelta(days=2)
    past = await surrealdb_storage.find_neurons(
        time_range=(past_start, past_start + timedelta(days=1)), limit=100
    )
    assert past == []


@pytest.mark.asyncio
async def test_time_neuron_lookup_can_reach_storage(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """The pipeline's TIME lookup can see a neuron through this filter at all.

    This asserts reachability, not de-duplication: `_find_similar_time_neuron`
    compares `created_at` (insert time) against a window around the hint's
    midpoint (referenced time), which are different quantities, so it only
    coincides when a neuron was inserted near the referenced moment. While the
    filter matched nothing, the lookup could never return anything at all.
    """
    from surreal_memory.engine.pipeline_steps import _find_similar_time_neuron

    now = utcnow()
    await surrealdb_storage.add_neuron(
        Neuron.create(type=NeuronType.TIME, content="this afternoon")
    )

    existing = await _find_similar_time_neuron(surrealdb_storage, now)
    assert existing is not None, "the TIME lookup could not see a neuron inserted just now"


@pytest.mark.asyncio
async def test_today_fibers_count_counts_only_today(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """`today_fibers_count` must not count fibers created before today.

    The same string-bound comparison appeared in get_enhanced_stats, but with a
    one-sided `>=`, where the constant predicate is always TRUE - so the counter
    reported every fiber the brain had ever held as created today. It surfaces in
    `smem info` and on the dashboard.
    """
    neuron = Neuron.create(type=NeuronType.CONCEPT, content="fiber anchor")
    await surrealdb_storage.add_neuron(neuron)

    fresh = Fiber.create(
        neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id, summary="today"
    )
    await surrealdb_storage.add_fiber(fresh)

    old = Fiber.create(
        neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id, summary="old"
    )
    old = dataclasses.replace(old, created_at=utcnow() - timedelta(days=10))
    await surrealdb_storage.add_fiber(old)

    stats = await surrealdb_storage.get_enhanced_stats(surrealdb_storage.current_brain_id)
    assert stats["today_fibers_count"] == 1, "the counter included a fiber created ten days ago"
