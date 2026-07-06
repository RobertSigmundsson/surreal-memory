"""Issue #36: get_fibers(exclude_expired=True) drops soft-forgotten memories.

Soft-forget sets typed_memory.expires_at=now; recall must exclude such fibers
immediately (before consolidation cleanup) instead of only under hard delete.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _make_store() -> SurrealDBStorage:
    """A SurrealDBStorage with _query/_get_brain_id stubbed (no real connection)."""
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
    store._get_brain_id = lambda: "default"  # type: ignore[method-assign,assignment]
    return store


@pytest.mark.asyncio
async def test_exclude_expired_adds_typed_memory_filter() -> None:
    store = _make_store()
    await store.get_fibers(limit=5, exclude_expired=True)
    sql = store._query.call_args.args[0]  # type: ignore[attr-defined]
    assert "typed_memory" in sql
    assert "expires_at" in sql
    assert "time::now()" in sql


@pytest.mark.asyncio
async def test_default_keeps_all_fibers() -> None:
    store = _make_store()
    await store.get_fibers(limit=5)
    sql = store._query.call_args.args[0]  # type: ignore[attr-defined]
    assert "typed_memory" not in sql
    assert "expires_at" not in sql
