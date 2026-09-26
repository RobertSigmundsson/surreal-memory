"""Reasoning merge-key on the paged pattern scan, against a live SurrealDB.

3.12.0 walks pattern fibers with ``get_fibers_after_id`` (keyset pages) instead of one
``find_fibers(limit=...)`` call, and replays ``pending_patterns`` after a crash. The merge
key must work on both: a twin that only a later page reaches is still found, and a
replayed pending pattern folds into its twin exactly once.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio

import surreal_memory.engine.reasoning_distiller as rd
from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.reasoning_distiller import distill_reasoning_patterns
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]

_BATCH_A = [
    "Wait, let me reconsider the traceback. Now I verify the exception is gone.",
    "Hold on, let me reconsider this traceback. I verify the exception stopped.",
    "Actually, let me reconsider that traceback. Verify the exception is handled.",
]
_BATCH_B = [
    "Hold on, rethink this bug. Then validate the crash is gone for good.",
    "Wait, rethink the bug here. Then validate the crash no longer happens.",
    "Actually, rethink that bug. Then validate the crash is fixed properly.",
]


class _StopBeforeWriteError(Exception):
    pass


@pytest.fixture
def no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "_get_embedder", lambda *_a, **_k: None)


def _ucfg(tmp_path: Path) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(
            mining_enabled=True,
            min_cluster_support=2,
            min_patterns_per_category=1,
            min_confidence=0.2,
            pattern_targets={"claude-fable-5": 100},
        ),
    )


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
    brain = Brain.create(name="merge-paged-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(storage: SurrealDBStorage, prefix: str, contents: list[str]) -> None:
    await storage.insert_reasoning_traces(
        storage._get_brain_id(),
        [
            {
                "trace_hash": f"{prefix}-{i}",
                "model": "claude-fable-5",
                "session_id": "s",
                "project": "p",
                "task_context": "",
                "content": c,
                "content_chars": len(c),
                "created_at": "2026-03-01T00:00:00",
            }
            for i, c in enumerate(contents)
        ],
    )


async def _patterns(storage: SurrealDBStorage) -> list[Fiber]:
    return await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)


async def test_merge_key_finds_a_twin_beyond_the_first_keyset_page(
    store: SurrealDBStorage, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    brain_id = store._get_brain_id()
    cfg = _ucfg(tmp_path)
    await _seed(store, "a", _BATCH_A)
    first = await distill_reasoning_patterns(store, brain_id, cfg)
    assert first.patterns_learned == 1
    # Fibers whose ids sort BEFORE the pattern fiber, and a one-row page: the twin is
    # reachable only by walking the keyset past page 1 on the real store.
    for i in range(3):
        dummy = Fiber.create(neuron_ids={"n"}, synapse_ids=set(), anchor_neuron_id="n")
        await store.add_fiber(replace(dummy, id=f"00000000-0000-4000-8000-00000000000{i}"))
    monkeypatch.setattr(rd, "_PATTERN_PAGE_SIZE", 1)

    await _seed(store, "b", _BATCH_B)
    second = await distill_reasoning_patterns(store, brain_id, cfg)

    fibers = await _patterns(store)
    assert len(fibers) == 1
    assert (second.patterns_learned, second.patterns_merged) == (0, 1)
    assert fibers[0].metadata["_reasoning_frequency"] == 6


async def test_pending_pattern_replays_through_the_merge_path_once(
    store: SurrealDBStorage, tmp_path: Path, no_embedder: None
) -> None:
    brain_id = store._get_brain_id()
    cfg = _ucfg(tmp_path)
    await _seed(store, "a", _BATCH_A)
    await distill_reasoning_patterns(store, brain_id, cfg)

    captured: list[dict[str, object]] = []

    async def crash_after_naming(patterns: object) -> None:
        batch = list(patterns)  # type: ignore[call-overload]
        if batch:
            captured.extend(batch)
            raise _StopBeforeWriteError  # the process "dies" before the database write

    await _seed(store, "b", _BATCH_B)
    with pytest.raises(_StopBeforeWriteError):
        await distill_reasoning_patterns(
            store, brain_id, cfg, pattern_checkpoint=crash_after_naming
        )
    assert len(captured) == 1

    first = await distill_reasoning_patterns(store, brain_id, cfg, pending_patterns=captured)
    again = await distill_reasoning_patterns(store, brain_id, cfg, pending_patterns=captured)

    fibers = await _patterns(store)
    assert len(fibers) == 1
    assert (first.patterns_learned, first.patterns_merged) == (0, 1)
    assert (again.patterns_learned, again.patterns_merged) == (0, 0)
    assert fibers[0].metadata["_reasoning_frequency"] == 6
    assert len(fibers[0].metadata["_reasoning_signatures"]) == 2
