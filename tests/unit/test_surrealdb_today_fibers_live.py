"""Live-DB proof for the string-vs-datetime binding trap (see #today-fibers).

``test_surrealdb_datetime_binding.py`` models SurrealDB's cross-type comparison
rule in a fake ``_query``; this file checks that the model matches the real
engine, and that ``today_fibers_count`` / ``find_neurons(time_range=...)`` are
right end-to-end. Skipped unless ``SURREALDB_URL`` points at a running
SurrealDB.

Both assertions run against a throwaway brain seeded with fibers backdated
across several days, so "today" is a genuine subset — a brain where every fiber
is from today cannot distinguish the fix from the bug.
"""

from __future__ import annotations

import os
from dataclasses import replace
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
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)

BRAIN_NAME = "today-fibers-binding-live"

NOW = utcnow()
TODAY_MIDNIGHT = NOW.replace(hour=0, minute=0, second=0, microsecond=0)
#: Two fibers today (one exactly at midnight, pinning the inclusive boundary),
#: three strictly before it.
OFFSETS = [
    timedelta(0),
    timedelta(hours=1),
    timedelta(microseconds=-1),
    timedelta(days=-1),
    timedelta(days=-9),
]
EXPECTED_TODAY = 2


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name=BRAIN_NAME)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    # Backdate the neuron as well as the fiber. ``Neuron.create()`` stamps
    # created_at with "now", so seeding only the fibers leaves every neuron
    # dated today and the find_neurons window assertions can never fail.
    for i, offset in enumerate(OFFSETS):
        when = TODAY_MIDNIGHT + offset
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"binding-live-{i}")
        await store.add_neuron(replace(neuron, created_at=when))
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary=f"binding-live-{i}",
        )
        await store.add_fiber(replace(fiber, created_at=when))

    yield store

    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await store.close()
    except Exception:
        pass


class TestTodayFibersCountLive:
    async def test_counts_only_todays_fibers(self, storage) -> None:  # type: ignore[no-untyped-def]
        stats = await storage.get_enhanced_stats(storage._get_brain_id())

        assert stats["fiber_count"] == len(OFFSETS), "fixture did not seed as expected"
        assert stats["today_fibers_count"] == EXPECTED_TODAY
        # The bug's signature: today == all-time. Assert the two differ on data
        # constructed so that they must.
        assert stats["today_fibers_count"] != stats["fiber_count"]

    async def test_a_string_binding_would_still_count_everything(self, storage) -> None:  # type: ignore[no-untyped-def]
        """Positive control — proves the fixture can *detect* the defect.

        Without this, a green test above would be indistinguishable from a
        fixture that seeded nothing useful. Issue the buggy query by hand: it
        must return the all-time total, while the datetime binding returns 2.
        """
        bid = storage._get_brain_id()
        sql = "SELECT count() AS c FROM fiber WHERE brain_id = $bid AND created_at >= $t GROUP ALL"

        as_string = await storage._query(sql, bid=bid, t=TODAY_MIDNIGHT.isoformat())
        as_datetime = await storage._query(sql, bid=bid, t=TODAY_MIDNIGHT)

        assert int(as_string[0]["c"]) == len(OFFSETS), (
            "expected the string binding to match every row (datetime outranks "
            "string, so the predicate is unconditionally true)"
        )
        assert int(as_datetime[0]["c"]) == EXPECTED_TODAY


class TestFindNeuronsTimeRangeLive:
    async def test_time_range_is_not_silently_empty(self, storage) -> None:  # type: ignore[no-untyped-def]
        """``created_at <= $string`` is unconditionally false -> always []."""
        neurons = await storage.find_neurons(
            time_range=(TODAY_MIDNIGHT - timedelta(days=10), NOW + timedelta(minutes=1))
        )

        assert len(neurons) == len(OFFSETS)

    async def test_time_range_actually_filters(self, storage) -> None:  # type: ignore[no-untyped-def]
        """Control for the test above: a narrow window must exclude rows."""
        neurons = await storage.find_neurons(
            time_range=(TODAY_MIDNIGHT, NOW + timedelta(minutes=1))
        )

        assert len(neurons) == EXPECTED_TODAY
