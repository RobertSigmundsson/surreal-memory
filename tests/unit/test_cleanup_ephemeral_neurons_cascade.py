"""Regression guard for audit finding DB-01 (re-accrual, 2026-07-26): the
2026-07-19 fix (144c39d) made delete_neuron() cascade synapse deletes
brain-agnostically, but cleanup_ephemeral_neurons() used its own raw
conn.delete() with no cascade at all — orphaning both synapses and
neuron_state rows every time smem_auto's session-end ephemeral sweep ran.
This test asserts cleanup_ephemeral_neurons() now routes through
delete_neuron() so the cascade applies here too.
"""

from __future__ import annotations

from typing import Any

from surreal_memory.storage.surrealdb.store import SurrealDBStorage


class _FakeConn:
    """Captures SDK conn.delete() record ids."""

    def __init__(self) -> None:
        self.deletes: list[str] = []

    async def delete(self, record_id: str) -> None:
        self.deletes.append(record_id)


class _FakeStore:
    """Fake `self` exposing only what cleanup_ephemeral_neurons + delete_neuron touch."""

    def __init__(self, ephemeral_rows: list[dict[str, Any]]) -> None:
        self._conn = _FakeConn()
        self.queries: list[str] = []
        self._ephemeral_rows = ephemeral_rows

    def _ensure_conn(self) -> Any:
        return self._conn

    def _get_brain_id(self) -> str:
        return "uruboros"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(sql)
        if "FROM neuron WHERE" in sql and "ephemeral" in sql:
            return self._ephemeral_rows
        return []

    async def _record_change_internal(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def delete_neuron(self, neuron_id: str) -> bool:
        return await SurrealDBStorage.delete_neuron(self, neuron_id)


async def test_cleanup_ephemeral_neurons_cascades_synapses_and_state() -> None:
    store = _FakeStore(ephemeral_rows=[{"id": "neuron:00012d0f_c36e_4cd5_b2b4_7538fa683a17"}])

    deleted = await SurrealDBStorage.cleanup_ephemeral_neurons(store, max_age_hours=24.0)

    assert deleted == 1
    synapse_deletes = [q for q in store.queries if "DELETE synapse" in q]
    state_deletes = [q for q in store.queries if "DELETE neuron_state" in q]

    assert any("in = neuron:" in q for q in synapse_deletes)
    assert any("out = neuron:" in q for q in synapse_deletes)
    assert len(state_deletes) == 1
    assert store._conn.deletes == ["neuron:00012d0f_c36e_4cd5_b2b4_7538fa683a17"]


async def test_cleanup_ephemeral_neurons_skips_when_none_expired() -> None:
    store = _FakeStore(ephemeral_rows=[])

    deleted = await SurrealDBStorage.cleanup_ephemeral_neurons(store, max_age_hours=24.0)

    assert deleted == 0
    assert store._conn.deletes == []
