"""Regression: a pre-compression neuron snapshot must actually reach the database.

``save_neuron_snapshot`` sent ``compressed_at`` as a ``datetime`` while the
schema declares ``neuron_snapshots.compressed_at TYPE string``. On a SCHEMAFULL
table the write is refused with a coercion error, and the only caller
(``engine/compression.py``) wraps the call in a fail-soft ``except`` that logs
and compresses anyway - so tier 3-4 compression destroyed the original content
while its snapshot silently never existed, leaving ``recover_neuron_content``
nothing to restore from.

Only a live engine can catch this: the in-memory backend keeps a plain dict with
no schema, so the type mismatch is invisible there. The test is skipped when
SURREALDB_URL is unset so CI without docker still passes.
"""

from __future__ import annotations

import os
from datetime import datetime

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.utils.timeutils import utcnow
from tests.unit._surrealdb_live import cleanup_live_brains, ensure_real_surrealdb_sdk

SURREALDB_URL = os.getenv("SURREALDB_URL")

pytestmark = pytest.mark.skipif(
    not SURREALDB_URL,
    reason="requires SURREALDB_URL env var pointing to a running SurrealDB",
)


@pytest.fixture
async def surrealdb_storage():  # type: ignore[no-untyped-def]
    ensure_real_surrealdb_sdk()
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    storage = SurrealDBStorage(url=SURREALDB_URL)
    await storage.initialize()
    brain = Brain.create(name="neuron-snapshot-live")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    yield storage
    try:
        await cleanup_live_brains(storage, own_brain_id=brain.id)
    except Exception:
        pass
    try:
        await storage.close()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_save_neuron_snapshot_persists(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """The snapshot is readable after the write, and its timestamp round-trips."""
    brain_id = surrealdb_storage.current_brain_id
    before = utcnow()

    await surrealdb_storage.save_neuron_snapshot(
        neuron_id="snap-1",
        brain_id=brain_id,
        original_content="content that destructive compression is about to replace",
        compressed_at=before.isoformat(),
        tier=3,
    )

    snapshot = await surrealdb_storage.get_neuron_snapshot("snap-1")
    assert snapshot is not None, "the snapshot was refused by the schema and silently lost"
    assert snapshot["original_content"] == (
        "content that destructive compression is about to replace"
    )
    assert snapshot["tier"] == 3
    assert isinstance(snapshot["compressed_at"], datetime)
    assert snapshot["compressed_at"] == before.replace(tzinfo=None)


@pytest.mark.asyncio
async def test_save_neuron_snapshot_upserts(surrealdb_storage) -> None:  # type: ignore[no-untyped-def]
    """A second write for the same neuron updates the row (the merge branch).

    The merge path shares the record payload with the insert path, so it is
    subject to the same coercion. Note this test can only demonstrate the merge
    working: before the fix, the first write never lands, so there is no row to
    merge onto and the failure surfaces on the insert.
    """
    brain_id = surrealdb_storage.current_brain_id

    await surrealdb_storage.save_neuron_snapshot(
        neuron_id="snap-2",
        brain_id=brain_id,
        original_content="first",
        compressed_at=utcnow().isoformat(),
        tier=3,
    )
    second_at = utcnow()
    await surrealdb_storage.save_neuron_snapshot(
        neuron_id="snap-2",
        brain_id=brain_id,
        original_content="second",
        compressed_at=second_at.isoformat(),
        tier=4,
    )

    snapshot = await surrealdb_storage.get_neuron_snapshot("snap-2")
    assert snapshot is not None
    assert snapshot["original_content"] == "second"
    assert snapshot["tier"] == 4
    # The merged row carries the second write's timestamp, not the first's.
    assert snapshot["compressed_at"] == second_at.replace(tzinfo=None)
