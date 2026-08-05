"""SurrealDB device methods must return DeviceRecord, the interface contract.

Regression for two live 500s (SMEM-REBASE-330 / E4b): the backend returned
sync.device.DeviceInfo (a 3-field identity record, no last_sync_sequence),
so the hub routes' response serialization — which reads
``device.last_sync_sequence`` outside their try blocks — raised
AttributeError as an unhandled ASGI 500. register_device additionally left
the device row persisted before dying ("partial write + 500"). The in-memory
backend and the route-level mocks always carried the field, so only the real
SurrealDB path failed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.sync_records import DeviceRecord
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _storage() -> SurrealDBStorage:
    storage = SurrealDBStorage()
    storage.set_brain("brain1")
    return storage


class TestDeviceRecordShape:
    @pytest.mark.asyncio
    async def test_register_device_returns_device_record_with_sequence(self) -> None:
        storage = _storage()
        conn = AsyncMock()
        storage._ensure_conn = lambda: conn  # type: ignore[method-assign]

        device = await storage.register_device("abc123", "laptop")

        assert isinstance(device, DeviceRecord)
        assert device.last_sync_sequence == 0
        assert device.brain_id == "brain1"
        assert device.registered_at.isoformat()  # datetime, not str

    @pytest.mark.asyncio
    async def test_list_devices_maps_rows_to_device_records(self) -> None:
        storage = _storage()
        rows = [
            {
                "device_id": "abc123",
                "brain_id": "brain1",
                "device_name": "laptop",
                "registered_at": "2026-01-01T00:00:00Z",
                "last_sync_sequence": 7,
            }
        ]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        devices = await storage.list_devices()

        assert len(devices) == 1
        assert isinstance(devices[0], DeviceRecord)
        assert devices[0].last_sync_sequence == 7
        # the exact attribute chain the hub route serialization performs:
        assert devices[0].registered_at.isoformat()

    @pytest.mark.asyncio
    async def test_list_devices_tolerates_missing_sequence_field(self) -> None:
        storage = _storage()
        rows = [{"device_id": "dd", "registered_at": "2026-01-01T00:00:00Z"}]
        storage._query = AsyncMock(return_value=rows)  # type: ignore[method-assign]

        devices = await storage.list_devices()

        assert devices[0].last_sync_sequence == 0


class TestReRegisterUpsert:
    """Re-registering an existing (brain, device) pair must upsert, not 500.

    The old fallback passed a raw "device:<brain>_<device>" string to merge;
    the SDK parses that as SurrealQL, so a hyphen in the brain id reads as
    subtraction (ValidationError) and every re-register raised. The merge must
    target a RecordID and update the name only — never reset registered_at or
    the device's last_sync_sequence.
    """

    @pytest.mark.asyncio
    async def test_reregister_merges_name_only_via_record_id(self) -> None:
        from datetime import datetime

        from surrealdb import RecordID

        storage = _storage()
        conn = AsyncMock()
        conn.insert.side_effect = Exception("Database record already exists")
        storage._ensure_conn = lambda: conn  # type: ignore[method-assign]
        original = DeviceRecord(
            device_id="abc123",
            brain_id="brain1",
            device_name="renamed",
            last_sync_at=None,
            last_sync_sequence=7,
            registered_at=datetime(2026, 1, 1),
        )
        storage.get_device = AsyncMock(return_value=original)  # type: ignore[method-assign]

        device = await storage.register_device("abc123", "renamed")

        (rid, body), _ = conn.merge.await_args
        assert isinstance(rid, RecordID)
        assert body == {"device_name": "renamed"}  # name only — no cursor reset
        assert device.last_sync_sequence == 7  # preserved from the stored row
        assert device.registered_at == datetime(2026, 1, 1)  # original kept
