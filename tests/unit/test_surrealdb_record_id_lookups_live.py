"""Live-DB proof for the remaining `id = $rid` string comparisons.

Same defect as the alerts mixin: a lookup binds ``rid=f"<table>:{sid}"`` — a
string — and compares it with ``id = $rid``. ``id`` holds a record id, so the
predicate is unconditionally false and the row can never match. Two mixins
still carried it:

* ``cognitive.py`` — ``get_knowledge_gap`` and ``resolve_knowledge_gap``, so a
  gap could never be read back or closed (``mcp/cognitive_handler.py`` calls
  both).
* ``versions.py`` — ``get_version`` and ``delete_version``, so restoring or
  diffing a brain snapshot could never find its version
  (``engine/brain_versioning.py`` calls ``get_version`` on four paths).

Each test seeds a decoy row as well, so a build that simply drops the id filter
fails here instead of passing: against a single-row table an unfiltered lookup
is indistinguishable from a correct one.

Skipped unless ``SURREALDB_URL`` points at a running SurrealDB.
"""

from __future__ import annotations

import json
import os

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.engine.brain_versioning import BrainVersion
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL pointing to a running SurrealDB",
)

BRAIN_NAME = "record-id-lookups-live"


@pytest.fixture
async def storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    store = SurrealDBStorage(url=SURREALDB_URL)
    await store.initialize()
    brain = Brain.create(name=BRAIN_NAME)
    await store.save_brain(brain)
    store.set_brain(brain.id)

    yield store

    try:
        await cleanup_live_brains(store, own_brain_id=brain.id)
    finally:
        await store.close()


def _make_version(version_id: str, number: int) -> BrainVersion:
    return BrainVersion(
        id=version_id,
        brain_id="",
        version_name=f"v{number}",
        version_number=number,
        description="record-id-lookups-live",
        neuron_count=1,
        synapse_count=0,
        fiber_count=0,
        snapshot_hash=f"hash-{number}",
        created_at=utcnow(),
    )


async def test_get_knowledge_gap_finds_the_gap_it_just_created(storage) -> None:  # type: ignore[no-untyped-def]
    decoy = await storage.add_knowledge_gap(topic="decoy", detection_source="test")
    gap_id = await storage.add_knowledge_gap(topic="target", detection_source="test")
    assert decoy != gap_id

    gap = await storage.get_knowledge_gap(gap_id)

    assert gap is not None, "get_knowledge_gap returned None for a gap that exists"
    assert gap["topic"] == "target", "the lookup ignored the id and returned another row"


async def test_resolve_knowledge_gap_closes_the_right_row(storage) -> None:  # type: ignore[no-untyped-def]
    decoy = await storage.add_knowledge_gap(topic="decoy", detection_source="test")
    gap_id = await storage.add_knowledge_gap(topic="target", detection_source="test")

    assert await storage.resolve_knowledge_gap(gap_id) is True

    resolved = await storage.get_knowledge_gap(gap_id)
    assert resolved is not None and resolved["resolved_at"] is not None
    # The decoy must still be open: a lookup that ignored the id would have
    # matched it instead.
    still_open = await storage.get_knowledge_gap(decoy)
    assert still_open is not None and still_open["resolved_at"] is None

    # Already resolved: the second call must report no work done.
    assert await storage.resolve_knowledge_gap(gap_id) is False


async def test_unknown_gap_id_is_reported_as_missing(storage) -> None:  # type: ignore[no-untyped-def]
    await storage.add_knowledge_gap(topic="decoy", detection_source="test")

    assert await storage.get_knowledge_gap("00000000-0000-4000-8000-000000000000") is None
    assert await storage.resolve_knowledge_gap("00000000-0000-4000-8000-000000000000") is False


async def test_get_version_returns_the_requested_snapshot(storage) -> None:  # type: ignore[no-untyped-def]
    brain_id = storage._get_brain_id()
    decoy = _make_version("11111111-1111-4111-8111-111111111111", 1)
    target = _make_version("22222222-2222-4222-8222-222222222222", 2)
    await storage.save_version(brain_id, decoy, json.dumps({"which": "decoy"}))
    await storage.save_version(brain_id, target, json.dumps({"which": "target"}))

    result = await storage.get_version(brain_id, target.id)

    assert result is not None, "get_version returned None for a version that exists"
    version, snapshot_json = result
    assert version.version_number == 2, "the lookup ignored the id and returned another row"
    assert json.loads(snapshot_json) == {"which": "target"}


async def test_delete_version_removes_only_the_named_one(storage) -> None:  # type: ignore[no-untyped-def]
    brain_id = storage._get_brain_id()
    decoy = _make_version("33333333-3333-4333-8333-333333333333", 3)
    target = _make_version("44444444-4444-4444-8444-444444444444", 4)
    await storage.save_version(brain_id, decoy, json.dumps({"which": "decoy"}))
    await storage.save_version(brain_id, target, json.dumps({"which": "target"}))

    assert await storage.delete_version(brain_id, target.id) is True

    assert await storage.get_version(brain_id, target.id) is None
    remaining = await storage.list_versions(brain_id)
    assert [v.version_number for v in remaining] == [3], "delete hit the wrong row"

    # Deleting it twice must report that there was nothing left to delete.
    assert await storage.delete_version(brain_id, target.id) is False


async def test_unknown_version_id_is_reported_as_missing(storage) -> None:  # type: ignore[no-untyped-def]
    brain_id = storage._get_brain_id()
    await storage.save_version(
        brain_id, _make_version("55555555-5555-4555-8555-555555555555", 5), json.dumps({})
    )

    assert await storage.get_version(brain_id, "99999999-9999-4999-8999-999999999999") is None
    assert await storage.delete_version(brain_id, "99999999-9999-4999-8999-999999999999") is False
