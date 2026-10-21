"""What a discovery under vector-worker traffic may return (independent review r1, B1).

A neuron the scan has already read and dropped for want of a vector, or one it has not reached yet,
can get its vector from the worker while the run is in flight. That change is tolerated (the
alternative is a run that fails whenever it overlaps the worker), so the result is not one of the
two silent runs but lies between them: nothing the silent run at the end would not also hold, and
everything the silent run at the token holds. Pairs of neurons that changed during the run that are
missing are found by the next run.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine import semantic_discovery as sd
from surreal_memory.engine.semantic_discovery import discover_semantic_synapses
from tests.unit.test_semantic_discovery_working_set import (
    _OLD,
    _REFERENCE_TIME,
    _config,
    _FeedStorage,
    _pairs,
)

VECTORS = {
    "n-0": [1.0, 0.0, 0.0],
    "n-1": [0.99, 0.14, 0.0],  # near n-0
    "n-2": [0.0, 1.0, 0.0],
    "n-3": [0.0, 0.99, 0.14],  # near n-2
    "n-4": [0.0, 0.0, 1.0],
    "n-5": [0.14, 0.0, 0.99],  # near n-4
    "n-6": [0.1, 0.05, 0.99],  # near n-4 and n-5
    "n-7": [0.7, 0.7, 0.0],
}
LATE = {"n-1", "n-6"}  # created before the cutoff, vector not filled yet


@pytest.fixture(autouse=True)
def _fixture_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr(sd, "_EMBEDDING_PAGE_SIZE", 4)


def _neuron(nid: str, vector: list[float] | None) -> Neuron:
    metadata: dict[str, Any] = {} if vector is None else {"_embedding": vector}
    return dc_replace(
        Neuron.create(NeuronType.CONCEPT, f"content of {nid}", metadata=metadata, neuron_id=nid),
        created_at=_OLD,
    )


def _storage(*, filled: set[str], cls: type[_FeedStorage] = _FeedStorage) -> _FeedStorage:
    neurons = [
        _neuron(nid, vector if nid not in LATE or nid in filled else None)
        for nid, vector in VECTORS.items()
    ]
    storage = cls(neurons=neurons, synapses=[])
    storage.brain.config = _config()
    return storage


class _WorkerBetweenPages(_FeedStorage):
    """The vector worker fills n-1 (already read, dropped) and n-6 (not read yet) between pages."""

    calls = 0

    async def find_neurons_after_id(self, cursor_id: str | None, **kwargs: Any) -> list[Neuron]:  # type: ignore[override]
        type(self).calls += 1
        if type(self).calls == 2:
            for nid in sorted(LATE):
                self.set_vector(nid, VECTORS[nid])
        return await super().find_neurons_after_id(cursor_id, **kwargs)


async def _pair_set(storage: _FeedStorage) -> set[tuple[str, str]]:
    async def checkpoint(_stage: str, _cursor: str | None, _details: dict[str, Any]) -> None:
        return None

    result = await discover_semantic_synapses(
        storage,
        _config(),
        checkpoint=checkpoint,
        resume_state=None,
        budget_check=None,
        reference_time=_REFERENCE_TIME,
    )
    return {(a, b) for a, b, _cosine in _pairs(result)}


@pytest.mark.asyncio
async def test_a_vector_filled_during_the_run_gives_a_result_between_the_two_silent_runs() -> None:
    silent_at_token = await _pair_set(_storage(filled=set()))
    silent_at_end = await _pair_set(_storage(filled=LATE))
    _WorkerBetweenPages.calls = 0

    under_traffic = await _pair_set(_storage(filled=set(), cls=_WorkerBetweenPages))

    assert silent_at_token < silent_at_end  # the scenario really separates the two silent runs
    assert silent_at_token <= under_traffic <= silent_at_end
