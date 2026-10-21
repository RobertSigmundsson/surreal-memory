"""Resume of the unbounded semantic_link snapshot when scan order differs from priority order.

With at most MAX_NEURONS_TO_LINK eligible neurons the snapshot keeps every candidate. After the
per-type interleave the ids were re-sorted but their priorities were not, so a resume paired the
wrong priority with each neuron: edges came back with swapped orientation after a pause at the
start of the similarity stage, and a pause inside it failed validation outright (independent
review r1, P3).
"""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from surreal_memory.engine.semantic_discovery import discover_semantic_synapses
from tests.unit.test_consolidation_mid_strategies_resume import _REFERENCE_TIME, _Storage

_OLD = _REFERENCE_TIME - timedelta(days=1)


@pytest.fixture(autouse=True)
def _fixture_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )


V = {
    "n-0": [1.0, 0.0, 0.0],
    "n-1": [0.99, 0.14, 0.0],
    "n-2": [0.0, 1.0, 0.0],
    "n-3": [0.0, 0.99, 0.14],
    "n-4": [0.0, 0.0, 1.0],
    "n-5": [0.14, 0.0, 0.99],
    "n-6": [0.1, 0.05, 0.99],
    "n-7": [0.7, 0.7, 0.0],
}
TYPES = {
    "n-0": NeuronType.CONCEPT,
    "n-1": NeuronType.CONCEPT,
    "n-2": NeuronType.ENTITY,
    "n-3": NeuronType.CONCEPT,
    "n-4": NeuronType.ENTITY,
    "n-5": NeuronType.CONCEPT,
    "n-6": NeuronType.CONCEPT,
    "n-7": NeuronType.ENTITY,
}


def _world() -> _Storage:
    neurons = [
        dc_replace(
            Neuron.create(
                TYPES[nid], f"content {nid}", metadata={"_embedding": vec}, neuron_id=nid
            ),
            created_at=_OLD,
        )
        for nid, vec in V.items()
    ]
    storage = _Storage(neurons=neurons, synapses=[])
    storage.brain.config = _config()
    return storage


async def _run(storage: _Storage, resume: dict[str, Any] | None, pause_at: int | None):
    seen: list[dict[str, Any]] = []
    calls = 0

    async def checkpoint(stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        seen.append(json.loads(json.dumps(details)))

    async def budget() -> None:
        nonlocal calls
        calls += 1
        if pause_at is not None and calls == pause_at:
            raise ConsolidationPausedError("pause")

    try:
        result = await discover_semantic_synapses(
            storage,
            _config(),
            checkpoint=checkpoint,
            resume_state=resume,
            budget_check=budget,
            reference_time=_REFERENCE_TIME,
        )
    except ConsolidationPausedError:
        return None, seen
    return result, seen


@pytest.mark.asyncio
async def test_unbounded_snapshot_resume_keeps_edge_orientation() -> None:
    silent, _ = await _run(_world(), None, None)
    for pause_at in range(2, 12):
        storage = _world()
        paused, seen = await _run(storage, None, pause_at)
        if paused is None and seen and seen[-1].get("stage") in {"synapses", "similarity"}:
            break
    assert paused is None
    manifest = seen[-1]
    resumed, _ = await _run(storage, manifest, None)
    assert resumed is not None
    a = sorted((s.id, s.source_id, s.target_id) for s in silent.synapses)
    b = sorted((s.id, s.source_id, s.target_id) for s in resumed.synapses)
    assert a == b


@pytest.mark.asyncio
async def test_unbounded_snapshot_resume_mid_similarity_validates() -> None:
    silent, _ = await _run(_world(), None, None)
    chosen = None
    for pause_at in range(2, 30):
        storage = _world()
        paused, seen = await _run(storage, None, pause_at)
        if (
            paused is None
            and seen
            and seen[-1].get("stage") == "similarity"
            and seen[-1].get("cursor") is not None
        ):
            chosen = pause_at
            break
    assert chosen is not None, "no mid-similarity pause found"
    manifest = seen[-1]
    resumed, _ = await _run(storage, manifest, None)
    a = sorted((s.id, s.source_id, s.target_id) for s in silent.synapses)
    b = sorted((s.id, s.source_id, s.target_id) for s in resumed.synapses)
    assert a == b
