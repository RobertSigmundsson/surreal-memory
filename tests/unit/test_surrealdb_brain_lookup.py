"""Unit tests for SurrealDB brain lookup (no live DB required).

Regression coverage for the duplicate-brain bug: get_brain() must match a
brain by its `name` field, not only by a record-id substring. When it only
matched by id, get_brain("my-brain.v2") always returned None (record ids are
random UUIDs), so the bootstrap re-created a fresh brain on every process
start, accumulating hundreds of orphan rows.
"""

from __future__ import annotations

from typing import Any

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class _BrainLookupStore(SurrealDBStorage):
    """Instantiate without connecting; stub the query layer with fixed rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def _ensure_conn(self) -> Any:  # type: ignore[override]
        return object()

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        return self._rows


async def test_get_brain_matches_by_name_field() -> None:
    # Record id is a random UUID; the only link to the requested brain is name.
    rows = [
        {
            "id": "brain:9f1c0e2a-0000-4000-8000-000000000000",
            "name": "my-brain.v2",
            "metadata": {},
            "created_at": "2026-05-28T00:00:00Z",
            "updated_at": "2026-05-28T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    brain = await store.get_brain("my-brain.v2")
    assert brain is not None
    assert brain.name == "my-brain.v2"


async def test_get_brain_still_matches_by_record_id() -> None:
    rows = [
        {
            "id": "brain:default",
            "name": "default",
            "metadata": {},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    brain = await store.get_brain("default")
    assert brain is not None
    assert brain.name == "default"


async def test_get_brain_returns_none_when_absent() -> None:
    rows = [
        {
            "id": "brain:default",
            "name": "default",
            "metadata": {},
            "created_at": "2026-05-27T00:00:00Z",
            "updated_at": "2026-05-27T00:00:00Z",
        }
    ]
    store = _BrainLookupStore(rows)
    assert await store.get_brain("does-not-exist") is None


async def test_list_brain_names_returns_distinct_sorted() -> None:
    # Duplicate brain rows (the orphan-row leak) must collapse to one name.
    rows = [
        {"name": "my-brain.v2"},
        {"name": "default"},
        {"name": "my-brain.v2"},
    ]
    store = _BrainLookupStore(rows)
    assert await store.list_brain_names() == ["default", "my-brain.v2"]


async def test_list_brain_names_ignores_blank_names() -> None:
    rows = [{"name": "default"}, {"name": ""}, {"name": None}]
    store = _BrainLookupStore(rows)
    assert await store.list_brain_names() == ["default"]
