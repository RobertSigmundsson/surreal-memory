"""What a discovery under vector-worker traffic may return (independent reviews r1 B1, r2 P-r2-1).

A neuron the scan has already read and dropped for want of a vector, or one it has not reached yet,
can get its vector from the worker while the run is in flight. That change is tolerated (the
alternative is a run that fails whenever it overlaps the worker), so the SET of pairs the run
returns is not guaranteed to equal either silent run: it can differ from both, in both directions
(top-K per neuron, the candidate cap, ``max_pairs``, the type ranks that fix an edge's direction).

What does hold, and is what these tests pin, is a guarantee per edge: every edge written is valid
against the neurons as they are at the end of the run (cosine at or above the threshold on their
final vectors, both ends candidates whose digest still matches). A pair that was missed is found by
the next run, because by then both neurons have their vectors.
"""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine import semantic_discovery as sd
from surreal_memory.engine.semantic_discovery import (
    SemanticDiscoveryResult,
    discover_semantic_synapses,
)
from tests.unit.test_semantic_discovery_working_set import (
    _OLD,
    _REFERENCE_TIME,
    _config,
    _FeedStorage,
)

THRESHOLD = 0.7  # semantic_discovery_similarity_threshold of _config()
Spec = dict[str, tuple[NeuronType, list[float]]]


@pytest.fixture(autouse=True)
def _fixture_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def _neuron(nid: str, kind: NeuronType, vector: list[float] | None) -> Neuron:
    metadata: dict[str, Any] = {} if vector is None else {"_embedding": vector}
    return dc_replace(
        Neuron.create(kind, f"content of {nid}", metadata=metadata, neuron_id=nid),
        created_at=_OLD,
    )


def _storage(spec: Spec, lazy: set[str], *, filled: set[str], worker: bool) -> _FeedStorage:
    """``lazy`` neurons exist without a vector; ``filled`` ones already have it; ``worker`` fills the
    rest between the first and the second page of the neuron scan."""

    class _Worker(_FeedStorage):
        page_calls = 0  # not "calls": _FeedStorage has an attribute of that name

        async def find_neurons_after_id(  # type: ignore[override]
            self, cursor_id: str | None, **kwargs: Any
        ) -> list[Neuron]:
            self.page_calls += 1
            if worker and self.page_calls == 2:
                for nid in sorted(lazy - filled):
                    self.set_vector(nid, spec[nid][1])
            return await super().find_neurons_after_id(cursor_id, **kwargs)

    neurons = [
        _neuron(nid, kind, vector if nid not in lazy or nid in filled else None)
        for nid, (kind, vector) in spec.items()
    ]
    storage = _Worker(neurons=neurons, synapses=[])
    storage.brain.config = _config()
    return storage


async def _discover(storage: _FeedStorage) -> SemanticDiscoveryResult:
    async def checkpoint(_stage: str, _cursor: str | None, _details: dict[str, Any]) -> None:
        return None

    return await discover_semantic_synapses(
        storage,
        _config(),
        checkpoint=checkpoint,
        resume_state=None,
        budget_check=None,
        reference_time=_REFERENCE_TIME,
    )


def _undirected(result: SemanticDiscoveryResult) -> set[tuple[str, str]]:
    return {tuple(sorted((s.source_id, s.target_id))) for s in result.synapses}  # type: ignore[misc]


async def _three(spec: Spec, lazy: set[str]) -> tuple[set[Any], set[Any], SemanticDiscoveryResult]:
    """(silent run at the token, silent run at the end, run with the worker in flight)."""
    at_token = _undirected(await _discover(_storage(spec, lazy, filled=set(), worker=False)))
    at_end = _undirected(await _discover(_storage(spec, lazy, filled=lazy, worker=False)))
    traffic = await _discover(_storage(spec, lazy, filled=set(), worker=True))
    return at_token, at_end, traffic


def _assert_every_edge_is_valid_at_the_end(spec: Spec, result: SemanticDiscoveryResult) -> None:
    for synapse in result.synapses:
        a, b = spec[synapse.source_id][1], spec[synapse.target_id][1]
        assert _cosine(a, b) >= THRESHOLD, (synapse.source_id, synapse.target_id)


def _two_clusters(top_k_crowd: int) -> Spec:
    """n-0..: ``top_k_crowd`` neurons that are all very close to n-A, then A and B with cos(A,B)=0.8."""
    u = _unit([1.8, 0.6, 0.0])
    spec: Spec = {
        f"n-{i}": (NeuronType.CONCEPT, _unit([u[0], u[1], 0.05 * i])) for i in range(top_k_crowd)
    }
    spec[f"n-{top_k_crowd}"] = (NeuronType.CONCEPT, [1.0, 0.0, 0.0])  # A
    spec[f"n-{top_k_crowd + 1}"] = (NeuronType.CONCEPT, [0.8, 0.6, 0.0])  # B
    return spec


@pytest.mark.asyncio
async def test_every_edge_written_under_worker_traffic_is_valid_at_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sd, "_EMBEDDING_PAGE_SIZE", 4)
    spec: Spec = {
        "n-0": (NeuronType.CONCEPT, [1.0, 0.0, 0.0]),
        "n-1": (NeuronType.CONCEPT, [0.99, 0.14, 0.0]),  # near n-0, vector filled during the run
        "n-2": (NeuronType.CONCEPT, [0.0, 1.0, 0.0]),
        "n-3": (NeuronType.CONCEPT, [0.0, 0.99, 0.14]),
        "n-4": (NeuronType.CONCEPT, [0.0, 0.0, 1.0]),
        "n-5": (NeuronType.CONCEPT, [0.14, 0.0, 0.99]),
        "n-6": (NeuronType.CONCEPT, [0.1, 0.05, 0.99]),  # filled during the run
        "n-7": (NeuronType.CONCEPT, [0.7, 0.7, 0.0]),
    }

    at_token, at_end, traffic = await _three(spec, {"n-1", "n-6"})

    assert at_token < at_end  # the scenario really separates the two silent runs
    assert _undirected(traffic) <= at_end  # holds here; not a guarantee (see the next test)
    _assert_every_edge_is_valid_at_the_end(spec, traffic)


@pytest.mark.asyncio
async def test_the_pair_set_under_traffic_can_differ_from_both_silent_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Characterization, not a wish: the top-K cut per neuron makes the set differ in both directions."""
    monkeypatch.setattr(sd, "_EMBEDDING_PAGE_SIZE", 5)
    crowd = sd.SEMANTIC_TOP_K  # neurons that crowd A and B's top-K once their vectors exist
    spec = _two_clusters(crowd)
    lazy = {f"n-{i}" for i in range(crowd)}
    a_b = (f"n-{crowd}", f"n-{crowd + 1}")

    at_token, at_end, traffic = await _three(spec, lazy)

    under_traffic = _undirected(traffic)
    assert a_b in at_token and a_b not in at_end  # the crowd pushes A-B out of both top-K lists
    assert a_b in under_traffic  # the crowd was dropped for want of a vector when A and B were read
    assert (
        not under_traffic <= at_end
    )  # so the result is NOT contained in the silent run at the end
    _assert_every_edge_is_valid_at_the_end(spec, traffic)
