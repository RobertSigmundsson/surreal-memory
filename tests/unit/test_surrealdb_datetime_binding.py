"""Regression tests for the string-vs-datetime binding trap on SurrealDB.

SurrealDB does **not** raise when a datetime column is compared against a bound
string. It orders values across types by type rank, and datetime outranks
string, so the bound value is never actually consulted:

    RETURN d'2000-01-01T00:00:00Z' >= '2099-06-01'   -- true
    RETURN d'2000-01-01T00:00:00Z' <= 'zzzz'         -- false

Both measured on SurrealDB 3.2.3. That makes ``created_at >= $str``
unconditionally true and ``created_at <= $str`` unconditionally false, and both
directions shipped:

* ``get_enhanced_stats`` bound ``today.isoformat()``, so ``today_fibers_count``
  reported the all-time fiber total. On the production brain it read 1680 when
  30 fibers had been created that day — a metric the dashboard, the ``smem
  info`` CLI and the MCP stats tool had all been rendering wrong for weeks.
* ``find_neurons(time_range=...)`` bound both ends as strings, so the ``<=``
  half could never match and every time-ranged neuron lookup came back empty:
  0 rows against the same brain where datetime bindings return 453.

The mock in ``test_surrealdb_enhanced_stats.py`` cannot catch this — it returns
a canned row without inspecting the bindings, so it passes either way. Neither
can a test that only asserts the field is an ``int``. These tests instead model
the measured cross-type rule in ``_surreal_datetime_cmp`` and drive the real
``store.py`` code paths through it, so binding a string fails them.

``InMemoryStorage`` is exercised alongside as a parity control: it filters in
Python, where ``>=`` on a str/datetime pair raises rather than silently
misfiltering, so it was always correct — which is precisely why a
backend-agnostic test would have missed the bug entirely.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.neuron import NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

# A fixed "now" keeps the fixtures deterministic; every offset below is relative
# to it, so the tests never straddle a real midnight.
NOW = utcnow().replace(hour=14, minute=30, second=0, microsecond=0)
TODAY_MIDNIGHT = NOW.replace(hour=0, minute=0, second=0, microsecond=0)

#: created_at values spread across several days. Three land today (one right at
#: midnight, to pin the inclusive boundary), four are older.
FIBER_TIMES: list[datetime] = [
    TODAY_MIDNIGHT,  # today, exactly on the boundary -> counted
    TODAY_MIDNIGHT + timedelta(hours=9),  # today -> counted
    TODAY_MIDNIGHT + timedelta(hours=14, minutes=29),  # today -> counted
    TODAY_MIDNIGHT - timedelta(microseconds=1),  # yesterday, 1us before -> not
    TODAY_MIDNIGHT - timedelta(days=1),
    TODAY_MIDNIGHT - timedelta(days=3),
    TODAY_MIDNIGHT - timedelta(days=30),
]
EXPECTED_TODAY = 3


def _surreal_datetime_cmp(stored: datetime, bound: Any, op: str) -> bool:
    """Model SurrealDB's comparison of a datetime column against a bound value.

    Same-type comparisons behave normally. Cross-type ones fall back to type
    rank — datetime sorts after string, unconditionally — which is the whole
    defect: no error, no empty result, just a predicate that stopped depending
    on its operand.
    """
    if isinstance(bound, datetime):
        return stored >= bound if op == ">=" else stored <= bound
    if isinstance(bound, str):
        return op == ">="  # datetime always outranks string
    raise AssertionError(f"unexpected binding type for a datetime column: {type(bound)!r}")


def _make_store(fiber_times: list[datetime]) -> SurrealDBStorage:
    """A ``SurrealDBStorage`` whose ``_query`` enforces SurrealDB's type rules.

    Only the fragments the tested code paths emit are interpreted; everything
    else returns the empty/zero shape so ``get_enhanced_stats`` still completes.
    """
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "binding-test-brain"

    async def fake_query(sql: str, **params: Any) -> list[dict[str, Any]]:
        s = sql.lower()

        if "from fiber" in s and "created_at >=" in s and "group all" in s:
            bound = params["today"]
            return [{"c": sum(1 for t in fiber_times if _surreal_datetime_cmp(t, bound, ">="))}]

        if "from neuron" in s and "created_at >=" in s and "created_at <=" in s:
            start, end = params["time_start"], params["time_end"]
            return [
                {"id": f"neuron:n{i}", "type": "concept", "content": f"n{i}", "created_at": t}
                for i, t in enumerate(fiber_times)
                if _surreal_datetime_cmp(t, start, ">=") and _surreal_datetime_cmp(t, end, "<=")
            ]

        if "count()" in s and "group all" in s:
            return [{"c": 0}]
        if "group by type" in s:
            return []
        return []

    store._query = fake_query  # type: ignore[assignment,method-assign]
    return store


class TestTodayFibersCount:
    def test_counts_only_todays_fibers_not_the_all_time_total(self) -> None:
        """The headline defect: the count must not be the whole table."""
        store = _make_store(FIBER_TIMES)
        stats = asyncio.run(store.get_enhanced_stats("binding-test-brain"))

        assert stats["today_fibers_count"] == EXPECTED_TODAY
        # Guard the specific wrong answer the string binding produced, so a
        # future regression reads as "all-time total again", not just "off by n".
        assert stats["today_fibers_count"] != len(FIBER_TIMES)

    def test_binds_a_datetime_object_not_an_isoformat_string(self) -> None:
        """Pin the root cause directly: the binding's *type* is the contract."""
        captured: dict[str, Any] = {}
        store = _make_store(FIBER_TIMES)
        inner = store._query

        async def capturing_query(sql: str, **params: Any) -> list[dict[str, Any]]:
            if "from fiber" in sql.lower() and "created_at >=" in sql.lower():
                captured["today"] = params.get("today")
            return await inner(sql, **params)  # type: ignore[no-any-return]

        store._query = capturing_query  # type: ignore[assignment,method-assign]
        asyncio.run(store.get_enhanced_stats("binding-test-brain"))

        assert isinstance(captured["today"], datetime), (
            "today must be bound as a datetime; an isoformat string makes the "
            "predicate unconditionally true on SurrealDB"
        )
        assert captured["today"] == TODAY_MIDNIGHT.replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def test_counts_zero_when_nothing_was_created_today(self) -> None:
        """An empty result must be a real zero, not the string binding's total."""
        store = _make_store([TODAY_MIDNIGHT - timedelta(days=d) for d in (1, 2, 5)])
        stats = asyncio.run(store.get_enhanced_stats("binding-test-brain"))

        assert stats["today_fibers_count"] == 0


class TestFindNeuronsTimeRange:
    def test_time_range_returns_matches_instead_of_always_empty(self) -> None:
        """The ``<=`` half was unconditionally false, so this was always []."""
        store = _make_store(FIBER_TIMES)
        neurons = asyncio.run(
            store.find_neurons(
                time_range=(TODAY_MIDNIGHT - timedelta(days=4), NOW),
                type=NeuronType.CONCEPT,
            )
        )

        # Everything except the 30-day-old outlier falls inside the window.
        assert len(neurons) == len(FIBER_TIMES) - 1
        assert neurons, "time-ranged lookup must not be silently empty"

    def test_time_range_excludes_rows_outside_the_window(self) -> None:
        """Control for the test above: the filter must actually filter."""
        store = _make_store(FIBER_TIMES)
        neurons = asyncio.run(
            store.find_neurons(
                time_range=(TODAY_MIDNIGHT, NOW),
                type=NeuronType.CONCEPT,
            )
        )

        assert len(neurons) == EXPECTED_TODAY

    def test_binds_datetime_objects_for_both_bounds(self) -> None:
        captured: dict[str, Any] = {}
        store = _make_store(FIBER_TIMES)
        inner = store._query

        async def capturing_query(sql: str, **params: Any) -> list[dict[str, Any]]:
            if "time_start" in params:
                captured.update(params)
            return await inner(sql, **params)  # type: ignore[no-any-return]

        store._query = capturing_query  # type: ignore[assignment,method-assign]
        asyncio.run(store.find_neurons(time_range=(TODAY_MIDNIGHT, NOW)))

        assert isinstance(captured["time_start"], datetime)
        assert isinstance(captured["time_end"], datetime)


class TestCrossBackendParity:
    """``InMemoryStorage`` was never wrong here — document why it can't be.

    It filters with a plain Python ``>=``, which raises ``TypeError`` on a
    str/datetime pair instead of silently misfiltering. So the in-memory
    backend agreed with the *fixed* SurrealDB backend all along, and any test
    written against it would have reported green while production was wrong.
    """

    def test_in_memory_backend_agrees_with_the_fixed_surrealdb_backend(self) -> None:
        from dataclasses import replace

        from surreal_memory.core.brain import Brain
        from surreal_memory.core.fiber import Fiber
        from surreal_memory.core.neuron import Neuron
        from surreal_memory.storage.memory_store import InMemoryStorage

        async def _run() -> int:
            storage = InMemoryStorage()
            brain = Brain.create(name="binding-parity")
            await storage.save_brain(brain)
            storage.set_brain(brain.id)

            for i, when in enumerate(FIBER_TIMES):
                neuron = Neuron.create(type=NeuronType.CONCEPT, content=f"content-{i}")
                await storage.add_neuron(neuron)
                fiber = Fiber.create(
                    neuron_ids={neuron.id},
                    synapse_ids=set(),
                    anchor_neuron_id=neuron.id,
                    summary=f"fiber-{i}",
                )
                await storage.add_fiber(replace(fiber, created_at=when))

            stats = await storage.get_enhanced_stats(brain.id)
            return int(stats["today_fibers_count"])

        assert asyncio.run(_run()) == EXPECTED_TODAY

    def test_python_comparison_would_have_raised_on_the_bad_binding(self) -> None:
        """Why the two backends diverged rather than both being wrong."""
        with pytest.raises(TypeError):
            _ = FIBER_TIMES[0] >= TODAY_MIDNIGHT.isoformat()  # type: ignore[operator]
