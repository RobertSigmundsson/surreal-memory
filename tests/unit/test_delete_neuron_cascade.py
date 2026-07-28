"""Regression guard for audit finding DB-01: delete_neuron must remove ALL synapses
whose in/out points at the deleted neuron, regardless of the synapse's own brain_id.

Some write paths create synapses with a NULL brain_id. The previous implementation
deleted synapses with a `brain_id = '<brain>' AND in/out = neuron` filter, which
silently skipped NULL-brain synapses — orphaning them when the neuron was pruned
(the schema's cascade_delete_synapses event does not fire on the SDK conn.delete()).
This test asserts the synapse-delete predicate is brain-agnostic (endpoint only).
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
    """Fake `self` exposing only what delete_neuron touches (no live DB, no full init)."""

    def __init__(self) -> None:
        self._conn = _FakeConn()
        self.queries: list[str] = []

    def _ensure_conn(self) -> Any:
        return self._conn

    def _get_brain_id(self) -> str:
        return "uruboros"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(sql)
        return []

    async def _record_change_internal(self, *args: Any, **kwargs: Any) -> None:
        return None


async def test_delete_neuron_deletes_synapses_brain_agnostically() -> None:
    store = _FakeStore()

    ok = await SurrealDBStorage.delete_neuron(store, "00012d0f-c36e-4cd5-b2b4-7538fa683a17")

    assert ok is True
    synapse_deletes = [q for q in store.queries if "DELETE synapse" in q]

    # Both endpoints are swept (in AND out), each as its own index-friendly single-field DELETE.
    assert any("in = neuron:" in q for q in synapse_deletes)
    assert any("out = neuron:" in q for q in synapse_deletes)

    # DB-01 regression guard: NO brain_id predicate — that filter skipped NULL-brain
    # synapses and left dangling orphans.
    assert all("brain_id" not in q for q in synapse_deletes), synapse_deletes

    # The neuron record itself is deleted via the SDK (underscore-normalised id).
    sid = "00012d0f_c36e_4cd5_b2b4_7538fa683a17"
    assert f"neuron:{sid}" in store._conn.deletes


async def test_delete_neuron_returns_false_when_delete_raises() -> None:
    """Error path: a failing conn.delete() is swallowed and reported as False."""

    class _BoomConn(_FakeConn):
        async def delete(self, record_id: str) -> None:
            raise RuntimeError("delete boom")

    store = _FakeStore()
    store._conn = _BoomConn()

    ok = await SurrealDBStorage.delete_neuron(store, "abc-def")
    assert ok is False
