"""Disposable SurrealDB tests for bounded semantic keyset ranges."""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from datetime import timedelta
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_public_id
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_keyset_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        return hostname == "localhost" or bool(
            hostname and ipaddress.ip_address(hostname).is_loopback
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires explicit loopback SURREALDB_URL for disposable SurrealDB 3.2.4",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-keyset-range-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


def _captured_queries(store: SurrealDBStorage) -> list[tuple[str, dict[str, object]]]:
    queries: list[tuple[str, dict[str, object]]] = []
    original_query = store._query

    async def capture(sql: str, **params):
        queries.append((sql, params))
        return await original_query(sql, **params)

    store._query = capture  # type: ignore[method-assign]
    return queries


def _public_record_ids(rows: list[dict[str, object]]) -> list[str]:
    return [_to_public_id(_record_id_part(str(row["id"]))) for row in rows]


async def _assert_record_scan(
    store: SurrealDBStorage, query: tuple[str, dict[str, object]]
) -> None:
    sql, params = query
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN FULL", **params)
    plan_text = json.dumps(plan, ensure_ascii=False, default=str)
    assert "RecordIdScan" in plan_text, plan_text[:800]
    assert ".." in plan_text, plan_text[:800]


@pytest.mark.asyncio
async def test_neuron_keyset_range_fills_page_and_matches_brain_scoped_reference_query(
    store: SurrealDBStorage,
) -> None:
    reference_time = utcnow() + timedelta(days=1)
    future_time = reference_time + timedelta(days=1)
    cursor = "1234567890000000"
    brain_a_id = store._get_brain_id()
    brain_a_ids = [cursor, "1234567890000001", "1234567890000002", "1234567890000003"]
    for neuron_id in brain_a_ids:
        await store.add_neuron(
            Neuron(
                id=neuron_id,
                type=NeuronType.CONCEPT,
                content=neuron_id,
                created_at=future_time if neuron_id.endswith("2") else reference_time,
            )
        )

    other_brain = Brain.create(name="semantic-keyset-other-brain")
    await store.save_brain(other_brain)
    store.set_brain(other_brain.id)
    await store.add_neuron(
        Neuron.create(type=NeuronType.CONCEPT, content="foreign", neuron_id="1234567890000004")
    )
    store.set_brain(brain_a_id)
    queries = _captured_queries(store)
    actual = await store.find_neurons_after_id(cursor, limit=2, created_before=reference_time)
    sql, params = queries[0]
    reference_rows = await SurrealDBStorage._query_response(
        store,
        "SELECT * FROM neuron WHERE brain_id = $brain_id AND id > type::record('neuron', $cursor_id) "
        "AND (created_at IS NONE OR created_at <= $created_before) AND ephemeral = false "
        "ORDER BY id ASC LIMIT 2",
        brain_id=store._get_brain_id(),
        cursor_id=cursor,
        created_before=reference_time,
    )

    assert [row.id for row in actual] == _public_record_ids(reference_rows)
    assert [row.id for row in actual] == ["1234567890000001", "1234567890000003"]
    assert "FROM neuron:`1234567890000000`..`2`" in sql
    assert params["cursor_id"] == cursor
    await _assert_record_scan(store, queries[0])

    # Underscore-prefixed record names remain quoted and cursor-exclusive.
    for neuron_id in ("_123", "_124", "_125"):
        await store.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content=neuron_id, neuron_id=neuron_id)
        )
    underscore_page = await store.find_neurons_after_id(
        "_123", limit=2, created_before=reference_time
    )
    assert [row.id for row in underscore_page] == ["-124", "-125"]

    for neuron_id in ("7123", "7999", "8001", "8002"):
        await store.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content=neuron_id, neuron_id=neuron_id)
        )
    fill_offset = len(queries)
    filled_page = await store.find_neurons_after_id("7123", limit=3, created_before=reference_time)
    fill_queries = queries[fill_offset:]
    assert [row.id for row in filled_page] == ["7999", "8001", "8002"]
    assert len(fill_queries) == 2
    assert "FROM neuron:`7123`..`8`" in fill_queries[0][0]
    assert "FROM neuron:`8`..`9`" in fill_queries[1][0]


@pytest.mark.asyncio
async def test_synapse_keyset_range_fills_page_and_keeps_other_brains_out(
    store: SurrealDBStorage,
) -> None:
    reference_time = utcnow() + timedelta(days=1)
    cursor = "1234567891000000"
    for synapse_id in (cursor, "1234567891000001", "1234567891000002", "1234567891000003"):
        await store.add_synapse(
            Synapse.create(
                source_id="source",
                target_id="target",
                type=SynapseType.RELATED_TO,
                synapse_id=synapse_id,
            )
        )
    brain_id = store._get_brain_id()
    other_brain = Brain.create(name="semantic-synapse-keyset-other")
    await store.save_brain(other_brain)
    store.set_brain(other_brain.id)
    await store.add_synapse(
        Synapse.create(
            source_id="source",
            target_id="target",
            type=SynapseType.RELATED_TO,
            synapse_id="1234567891000004",
        )
    )
    store.set_brain(brain_id)

    queries = _captured_queries(store)
    actual = await store.get_synapses_after_id(cursor, limit=2, created_before=reference_time)
    sql, params = queries[0]
    reference_rows = await SurrealDBStorage._query_response(
        store,
        "SELECT * FROM synapse WHERE brain_id = $brain_id AND id > type::record('synapse', $cursor_id) "
        "AND (created_at IS NONE OR created_at <= $created_before) ORDER BY id ASC LIMIT 2",
        brain_id=brain_id,
        cursor_id=cursor,
        created_before=reference_time,
    )

    assert [row.id for row in actual] == _public_record_ids(reference_rows)
    assert [row.id for row in actual] == ["1234567891000001", "1234567891000002"]
    assert "FROM synapse:`1234567891000000`..`2`" in sql
    assert params["cursor_id"] == cursor
    await _assert_record_scan(store, queries[0])

    for synapse_id in ("7123", "7999", "8001", "8002"):
        await store.add_synapse(
            Synapse.create(
                source_id="source",
                target_id="target",
                type=SynapseType.RELATED_TO,
                synapse_id=synapse_id,
            )
        )
    fill_offset = len(queries)
    filled_page = await store.get_synapses_after_id("7123", limit=3, created_before=reference_time)
    fill_queries = queries[fill_offset:]
    assert [row.id for row in filled_page] == ["7999", "8001", "8002"]
    assert len(fill_queries) == 2
    assert "FROM synapse:`7123`..`8`" in fill_queries[0][0]
    assert "FROM synapse:`8`..`9`" in fill_queries[1][0]
