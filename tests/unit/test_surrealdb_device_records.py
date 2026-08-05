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
