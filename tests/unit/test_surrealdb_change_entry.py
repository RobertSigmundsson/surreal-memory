"""change_log read methods must return ChangeEntry, the interface contract.

Regression for a live 500 (SMEM-REBASE-330 / E4b round 4): the SurrealDB backend
returned sync.protocol.SyncChange (has ``sequence``, str ``changed_at``) while
both SyncEngine consumers read ``c.id`` and ``c.changed_at.isoformat()`` —
POST /hub/sync died with AttributeError on any brain whose change_log had
pending entries, and passed only when there was nothing to sync.

Adapted for the v3.6.2 rebase: the original home of this class
(tests/unit/test_surrealdb_device_records.py) was removed upstream when the
device-record fixes landed natively; the ChangeEntry contract on the SurrealDB
backend still needs its regression cover, so the class lives on here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.sync_records import ChangeEntry
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _storage() -> SurrealDBStorage:
    storage = SurrealDBStorage()
    storage.set_brain("brain1")
    return storage


class TestChangeLogEntryShape:
    _ROW = {
        "sequence": 11,
        "brain_id": "brain1",
        "entity_type": "neuron",
        "entity_id": "n-1",
        "operation": "insert",
        "device_id": "abc123",
        "changed_at": "2026-01-01T00:00:00Z",
        "payload": {"k": "v"},
        "synced": False,
    }

    @pytest.mark.asyncio
    async def test_get_changes_since_returns_change_entries(self) -> None:
        storage = _storage()
        storage._query = AsyncMock(return_value=[dict(self._ROW)])  # type: ignore[method-assign]

        changes = await storage.get_changes_since(0)

        assert isinstance(changes[0], ChangeEntry)
        # the exact attribute chain handle_hub_sync performs:
        assert changes[0].id == 11
        assert changes[0].changed_at.isoformat()

    @pytest.mark.asyncio
    async def test_get_unsynced_changes_returns_change_entries(self) -> None:
        storage = _storage()
        storage._query = AsyncMock(return_value=[dict(self._ROW)])  # type: ignore[method-assign]

        changes = await storage.get_unsynced_changes()

        assert isinstance(changes[0], ChangeEntry)
        assert changes[0].id == 11
        assert changes[0].changed_at.isoformat()
