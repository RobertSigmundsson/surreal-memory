"""Regression tests for the SurrealDB maturation mixin.

Guards two RUN-004 fixes:
1. ``save_maturation`` upserts by the EXISTING record id, not a recomputed
   ``{brain}_{fiber}`` sid. A brain rename updates the ``brain_id`` field but not
   the record id, so recomputing the id would target a non-existent record and
   the write would silently no-op (this is what broke review/maturation writes
   on the renamed ``default`` brain).
2. ``_row_to_maturation`` round-trips stage, rehearsal data, and the
   ``reinforcement_timestamps`` array faithfully.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.storage.surrealdb.maturation import (
    SurrealDBMaturationMixin,
    _row_to_maturation,
)


class _MaturationHarness(SurrealDBMaturationMixin):
    """Minimal concrete mixin wiring the three abstract hooks to test doubles."""

    def __init__(self, conn: MagicMock, brain_id: str, query_result: list) -> None:
        self._conn = conn
        self._brain = brain_id
        self._query_result = query_result

    def _ensure_conn(self) -> MagicMock:
        return self._conn

    def _get_brain_id(self) -> str:
        return self._brain

    async def _query(self, sql: str, **params: object) -> list:
        return self._query_result


@pytest.mark.asyncio
async def test_save_maturation_merges_by_existing_record_id() -> None:
    """Existing row with a stale (pre-rename) record id must still be updated."""
    conn = MagicMock()
    conn.merge = AsyncMock()
    conn.insert = AsyncMock()
    # Record id still carries the OLD brain prefix after a rename to "default".
    stale_id = "maturation:`my_brain.v2_abc_123`"
    harness = _MaturationHarness(conn, "default", query_result=[{"id": stale_id}])

    await harness.save_maturation(
        MaturationRecord(fiber_id="abc-123", brain_id="default", stage=MemoryStage.SEMANTIC)
    )

    conn.merge.assert_awaited_once()
    # Must target the EXISTING id, NOT the recomputed "maturation:default_abc_123".
    assert conn.merge.call_args[0][0] == stale_id
    conn.insert.assert_not_called()


@pytest.mark.asyncio
async def test_save_maturation_inserts_when_absent() -> None:
    """No existing row → insert with the computed deterministic id."""
    conn = MagicMock()
    conn.merge = AsyncMock()
    conn.insert = AsyncMock()
    harness = _MaturationHarness(conn, "default", query_result=[])

    await harness.save_maturation(
        MaturationRecord(fiber_id="abc-123", brain_id="default", stage=MemoryStage.WORKING)
    )

    conn.insert.assert_awaited_once()
    inserted = conn.insert.call_args[0][1]
    assert inserted["id"] == "default_abc_123"
    assert inserted["stage"] == "working"
    conn.merge.assert_not_called()


def test_row_to_maturation_roundtrip() -> None:
    row = {
        "fiber_id": "abc-123",
        "brain_id": "default",
        "stage": "semantic",
        "stage_entered_at": "2026-06-20T00:00:00Z",
        "rehearsal_count": 3,
        "reinforcement_timestamps": ["2026-06-20T00:00:00", "2026-06-21T00:00:00"],
    }

    rec = _row_to_maturation(row)

    assert rec.fiber_id == "abc-123"
    assert rec.brain_id == "default"
    assert rec.stage is MemoryStage.SEMANTIC
    assert rec.rehearsal_count == 3
    assert rec.reinforcement_timestamps == (
        "2026-06-20T00:00:00",
        "2026-06-21T00:00:00",
    )
