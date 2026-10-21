"""Disposable SurrealDB regressions for record-targeted prune operations."""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.storage.surrealdb._ids import _record_id_part, _to_public_id
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_prune_record_targets_it")


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
        reason="requires an explicit loopback URL for disposable SurrealDB 3.2.x",
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
    brain = Brain.create(name="prune-record-targets-it")
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


def _public_ids(rows: list[dict[str, object]]) -> list[str]:
    return [_to_public_id(_record_id_part(str(row["id"]))) for row in rows]


async def _assert_direct_record_source(
    store: SurrealDBStorage, query: tuple[str, dict[str, object]]
) -> None:
    sql, params = query
    plan = await SurrealDBStorage._query_response(store, sql + " EXPLAIN FULL", **params)
    plan_text = json.dumps(plan, ensure_ascii=False, default=str)
    assert "SourceExpr" in plan_text, plan_text[:1200]
    assert "IndexScan" not in plan_text, plan_text[:1200]
    assert "TableScan" not in plan_text, plan_text[:1200]


def _synapse(synapse_id: str, source: str, target: str) -> Synapse:
    return Synapse.create(
        source_id=source,
        target_id=target,
        type=SynapseType.RELATED_TO,
        synapse_id=synapse_id,
    )


@pytest.mark.asyncio
async def test_get_synapses_by_ids_record_scans_without_crossing_brains(
    store: SurrealDBStorage,
) -> None:
    active_ids = ["odd-record-name", "1122334455667788"]
    for synapse_id in active_ids:
        await store.add_synapse(_synapse(synapse_id, "active-source", "active-target"))

    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="prune-record-targets-foreign")
    await store.save_brain(foreign_brain)
    store.set_brain(foreign_brain.id)
    await store.add_synapse(_synapse("foreign-selected", "foreign-source", "foreign-target"))
    store.set_brain(active_brain_id)

    queries = _captured_queries(store)
    found = await store.get_synapses_by_ids(
        [*active_ids, "foreign-selected", "missing-selected", active_ids[0]]
    )

    assert {synapse.id for synapse in found} == set(active_ids)
    assert len(found) == len(active_ids)
    assert len(queries) == 1
    await _assert_direct_record_source(store, queries[0])


@pytest.mark.asyncio
async def test_delete_synapses_batch_targets_selected_rows_and_logs_once_on_replay(
    store: SurrealDBStorage,
) -> None:
    selected_ids = ["selected-non-uuid", "2233445566778899"]
    retained_id = "active-retained"
    for synapse_id in [*selected_ids, retained_id]:
        await store.add_synapse(_synapse(synapse_id, "active-source", "active-target"))

    active_brain_id = store._get_brain_id()
    foreign_brain = Brain.create(name="prune-delete-foreign")
    await store.save_brain(foreign_brain)
    store.set_brain(foreign_brain.id)
    foreign_selected_id = "foreign-selected"
    await store.add_synapse(_synapse(foreign_selected_id, "foreign-source", "foreign-target"))
    foreign_brain_id = store._get_brain_id()
    store.set_brain(active_brain_id)

    queries = _captured_queries(store)
    first_count = await store.delete_synapses_batch(
        [*selected_ids, foreign_selected_id, "missing-selected"]
    )
    replay_count = await store.delete_synapses_batch(
        [*selected_ids, foreign_selected_id, "missing-selected"]
    )

    assert first_count == len(selected_ids)
    assert replay_count == 0
    remaining = await SurrealDBStorage._query_response(
        store,
        "SELECT id, brain_id FROM synapse WHERE id IN ["
        "type::record('synapse', $foreign_id), type::record('synapse', $retained_id), "
        "type::record('synapse', $foreign_selected_id)]",
        foreign_id="selected_non_uuid",
        retained_id="active_retained",
        foreign_selected_id="foreign_selected",
    )
    assert set(_public_ids(remaining)) == {foreign_selected_id, retained_id}
    assert {str(row["brain_id"]) for row in remaining} == {
        active_brain_id,
        foreign_brain_id,
    }

    change_rows = await SurrealDBStorage._query_response(
        store,
        "SELECT entity_id, operation FROM change_log WHERE brain_id = $brain_id "
        "AND entity_type = 'synapse' AND operation = 'delete' "
        "AND entity_id IN $entity_ids",
        brain_id=active_brain_id,
        entity_ids=selected_ids,
    )
    assert {(str(row["entity_id"]), str(row["operation"])) for row in change_rows} == {
        (synapse_id, "delete") for synapse_id in selected_ids
    }
    delete_queries = [sql for sql, _ in queries if sql.startswith("DELETE FROM [")]
    assert len(delete_queries) == 2
    assert all(
        "WHERE brain_id = $brain_id" in sql and "RETURN BEFORE" in sql for sql in delete_queries
    )
