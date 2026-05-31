"""Regression test for the dashboard <-> CLI metric divergence (no live DB).

SurrealDB ``get_enhanced_stats`` must return a ``synapse_stats`` block with
per-type counts. Without it, ``DiagnosticsEngine`` computed ``diversity = 0``
and ``recall_confidence = 0`` on the SurrealDB backend (while the SQLite
backend computed real values), so the dashboard and the ``smem`` CLI reported
different grades for the very same brain — the "Grade F vs D" divergence.
"""

from __future__ import annotations

import asyncio
from typing import Any

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _make_store_with_scripted_query() -> SurrealDBStorage:
    """A SurrealDBStorage whose ``_query`` is scripted (no real connection)."""
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._current_brain_id = "test-brain"

    async def fake_query(sql: str, **_params: Any) -> list[dict[str, Any]]:
        s = sql.lower()
        if "from neuron" in s and "count()" in s and "group all" in s:
            return [{"c": 100}]
        if "from synapse" in s and "count()" in s and "group all" in s:
            return [{"c": 300}]
        if "from fiber" in s and "count()" in s and "group all" in s:
            return [{"c": 40}]
        if "from neuron" in s and "group by type" in s:
            return [{"type": "memory", "c": 100}]
        if "from synapse" in s and "group by type" in s:
            return [
                {"type": "related_to", "cnt": 150, "avg_w": 0.5, "total_r": 0},
                {"type": "after", "cnt": 100, "avg_w": 0.3, "total_r": 2},
                {"type": "co_occurs", "cnt": 50, "avg_w": 0.9, "total_r": 1},
            ]
        return []

    store._query = fake_query  # type: ignore[assignment,method-assign]
    return store


def test_enhanced_stats_includes_synapse_stats_by_type() -> None:
    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))

    assert "synapse_stats" in stats, "synapse_stats missing -> diversity would be 0"
    by_type = stats["synapse_stats"]["by_type"]
    assert set(by_type) == {"related_to", "after", "co_occurs"}
    # DiagnosticsEngine._compute_diversity reads entry["count"].
    assert by_type["related_to"]["count"] == 150
    assert by_type["after"]["total_reinforcements"] == 2
    assert stats["synapse_stats"]["avg_weight"] > 0


def test_enhanced_stats_synapse_stats_yields_nonzero_diversity() -> None:
    """The whole point: with synapse_stats present, diversity computes > 0."""
    from surreal_memory.engine.diagnostics import DiagnosticsEngine

    store = _make_store_with_scripted_query()
    stats = asyncio.run(store.get_enhanced_stats("test-brain"))
    diversity = DiagnosticsEngine._compute_diversity(stats["synapse_stats"])
    assert diversity > 0.0
