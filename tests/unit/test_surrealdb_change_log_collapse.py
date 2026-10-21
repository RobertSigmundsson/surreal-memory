"""Focused guards for bounded, brain-scoped pending update collapse."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class _RecordID:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


@pytest.mark.asyncio
async def test_collapse_uses_typed_record_list_and_reports_rows_actually_deleted() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    old = _RecordID("change_log:old_update")
    middle = _RecordID("change_log:middle_update")
    newest = _RecordID("change_log:newest_update")
    store._query = AsyncMock(
        return_value=[
            {"id": old, "entity_type": "synapse", "entity_id": "s-1", "sequence": 1},
            {
                "id": middle,
                "entity_type": "synapse",
                "entity_id": "s-1",
                "sequence": 2,
            },
            {
                "id": newest,
                "entity_type": "synapse",
                "entity_id": "s-1",
                "sequence": 3,
            },
        ]
    )
    # A concurrent sync made one selected candidate ineligible before DELETE.
    # RETURN BEFORE.id still gives the exact number that the server removed.
    store._query_values = AsyncMock(return_value=[old])

    assert await store.collapse_pending_updates(max_rows=37) == 1

    select = store._query.await_args
    assert "SELECT id, entity_type, entity_id, sequence FROM change_log" in select.args[0]
    assert select.kwargs["cap"] == 37
    store._query_values.assert_awaited_once()
    delete = store._query_values.await_args
    assert delete.args[0] == (
        "DELETE FROM $record_ids WHERE brain_id = $brain_id "
        "AND operation = 'update' AND synced = false RETURN BEFORE.id"
    )
    assert delete.kwargs["record_ids"] == [old, middle]
    assert delete.kwargs["record_ids"][0] is old
    assert delete.kwargs["brain_id"] == "brain-a"


@pytest.mark.asyncio
async def test_collapse_does_not_send_ids_that_fail_the_strict_record_id_check() -> None:
    store = object.__new__(SurrealDBStorage)
    store.set_brain("brain-a")
    store._query = AsyncMock(
        return_value=[
            {
                "id": _RecordID("change_log:unsafe'quoted"),
                "entity_type": "neuron",
                "entity_id": "n-1",
                "sequence": 1,
            },
            {
                "id": _RecordID("change_log:safe_newest"),
                "entity_type": "neuron",
                "entity_id": "n-1",
                "sequence": 2,
            },
        ]
    )
    store._query_values = AsyncMock()

    assert await store.collapse_pending_updates() == 0
    store._query.assert_awaited_once()
    store._query_values.assert_not_awaited()
