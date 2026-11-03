"""C2/P3 guard: ``batch_update_last_accessed`` is, and stays, a no-op.

It once wrote ``neuron.last_accessed_at`` (SQLite). On SurrealDB that table has a
changefeed whose after-image carries the embedding, and every event breaks the strict
``semantic_link`` source fence; nothing reads the column. A future "fix" that makes the
method write to ``neuron`` must fail here and be reconsidered (see the base docstring).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.base import NeuralStorage
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURREALDB_URL", "http://127.0.0.1:65535")
    monkeypatch.setenv("SURREALDB_USER", "root")
    monkeypatch.setenv("SURREALDB_PASS", "x")


def test_surrealdb_backend_issues_no_write_at_all() -> None:
    s = SurrealDBStorage()
    s._current_brain_id = "default"
    s._query = AsyncMock(return_value=[])
    s._conn = AsyncMock()

    asyncio.run(s.batch_update_last_accessed(["neuron:a", "neuron:b"]))

    assert s._query.await_count == 0
    assert s._conn.method_calls == []


def test_surrealdb_backend_does_not_override_the_noop() -> None:
    assert SurrealDBStorage.batch_update_last_accessed is NeuralStorage.batch_update_last_accessed


async def test_in_memory_neuron_rows_are_unchanged() -> None:
    storage = InMemoryStorage()
    storage.set_brain("p3")
    neuron = Neuron.create(type=NeuronType.ENTITY, content="p3")
    await storage.add_neuron(neuron)
    before = await storage.get_neuron(neuron.id)

    await storage.batch_update_last_accessed([neuron.id])

    assert await storage.get_neuron(neuron.id) == before


def test_docstring_states_the_truth() -> None:
    doc = NeuralStorage.batch_update_last_accessed.__doc__ or ""
    assert "NO-OP" in doc and "neuron_state" in doc
