"""Unit coverage for preserving SDK record IDs in entity-ref mutations."""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb.keyword_entity import (
    SurrealDBKeywordEntityMixin,
)


class _FakeConnection:
    def __init__(self) -> None:
        self.merged: list[tuple[object, dict[str, bool]]] = []
        self.deleted: list[object] = []

    async def merge(self, record_id: object, data: dict[str, bool]) -> None:
        self.merged.append((record_id, data))

    async def delete(self, record_id: object) -> None:
        self.deleted.append(record_id)


class _FakeStore(SurrealDBKeywordEntityMixin):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.connection = _FakeConnection()

    def _ensure_conn(self) -> _FakeConnection:
        return self.connection

    def _get_brain_id(self) -> str:
        return "brain-test"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        return self.rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "calls_attr"),
    [
        ("mark_entity_refs_promoted", "merged"),
        ("prune_old_entity_refs", "deleted"),
    ],
)
async def test_entity_ref_mutations_preserve_typed_ids_and_plain_mock_ids(
    operation: str,
    calls_attr: str,
) -> None:
    typed_id = object()
    string_id = "entity_refs:ascii-test"
    store = _FakeStore([{"id": typed_id}, {"id": string_id}])

    method = getattr(store, operation)
    result = await method("Łódź") if operation == "mark_entity_refs_promoted" else await method()

    assert result == 2
    calls = getattr(store.connection, calls_attr)
    recorded_ids = [call[0] if isinstance(call, tuple) else call for call in calls]
    assert recorded_ids[0] is typed_id
    assert recorded_ids[1] == string_id
