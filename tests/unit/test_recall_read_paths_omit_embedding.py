"""Recall's read paths must not drag the stored vector over the wire.

Why these tests: ``_row_to_neuron`` surfaces ``embedding_vec`` as
``metadata["_embedding"]``, so every point read and every inlined neighbour ships
1024 floats per row whether or not the caller looks at them. Measured on fresh
pristine copies with variants interleaved ABBA (the run-to-run drift of a single
recall is larger than the effect, so a one-shot comparison cannot see it): the
point-read projection alone is 1.025x of the recall wall and the neighbour one
alone 0.989x — noise — while both together are 0.8295x and 0.8229x on two copies,
faster in 10/10 and 10/10 samples. The gain therefore exists only while BOTH read
paths stay projected, which is exactly what a later refactor would silently undo.

The default stays ``True`` on purpose: the PUT route and ``content_refresh`` decide
whether to re-embed by reading ``metadata["_embedding"]`` back, so a projected
default would leave a vector describing text that no longer exists.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.activation import SpreadingActivation
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage


def _store_capturing(captured: list[str]) -> SurrealDBStorage:
    storage = SurrealDBStorage()
    storage._current_brain_id = "testbrain"

    async def fake_query(sql: str, **params: object) -> list[dict[str, object]]:
        captured.append(sql)
        return []

    storage._query = fake_query  # type: ignore[method-assign]
    return storage


@pytest.mark.asyncio
async def test_get_neuron_projects_the_vector_away_only_when_asked() -> None:
    captured: list[str] = []
    storage = _store_capturing(captured)

    await storage.get_neuron("n-1")
    await storage.get_neuron("n-1", include_embedding=False)

    assert len(captured) == 2, captured
    assert captured[0].startswith("SELECT * FROM neuron:"), captured[0]
    assert "OMIT" not in captured[0], captured[0]
    assert captured[1].startswith("SELECT * OMIT embedding_vec FROM neuron:"), captured[1]


@pytest.mark.asyncio
async def test_get_neurons_batch_projects_the_vector_away_only_when_asked() -> None:
    captured: list[str] = []
    storage = _store_capturing(captured)

    await storage.get_neurons_batch(["n-1"])
    await storage.get_neurons_batch(["n-1"], include_embedding=False)

    assert len(captured) == 2, captured
    assert "OMIT" not in captured[0], captured[0]
    assert captured[1].startswith("SELECT * OMIT embedding_vec FROM neuron:"), captured[1]


@pytest.mark.asyncio
async def test_get_neighbors_omits_the_inlined_neighbour_vectors() -> None:
    captured: list[str] = []
    storage = _store_capturing(captured)

    await storage.get_neighbors("n-1", direction="out")
    await storage.get_neighbors("n-1", direction="out", include_embedding=False)

    assert len(captured) == 2, captured
    assert "OMIT" not in captured[0], captured[0]
    # OMIT must name the ALIASES: omitting ``in.embedding_vec`` was measured to
    # leave the alias untouched, i.e. the vector still came back.
    assert "OMIT in_neuron.embedding_vec, out_neuron.embedding_vec" in captured[1], captured[1]
    assert "in.* AS in_neuron" in captured[1], captured[1]


class _RecordingStorage(InMemoryStorage):
    """In-memory backend that records how the recall path asked for neurons."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_calls: list[bool] = []
        self.neighbor_calls: list[bool] = []

    async def get_neurons_batch(  # type: ignore[override]
        self, neuron_ids: list[str], include_embedding: bool = True
    ) -> dict[str, Neuron]:
        self.batch_calls.append(include_embedding)
        return await super().get_neurons_batch(neuron_ids, include_embedding=include_embedding)

    async def get_neighbors(  # type: ignore[override]
        self,
        neuron_id: str,
        direction: str = "both",
        synapse_types: list[SynapseType] | None = None,
        min_weight: float | None = None,
        include_embedding: bool = True,
    ) -> list[tuple[Neuron, Synapse]]:
        self.neighbor_calls.append(include_embedding)
        return await super().get_neighbors(  # type: ignore[misc]
            neuron_id,
            direction=direction,  # type: ignore[arg-type]
            synapse_types=synapse_types,
            min_weight=min_weight,
            include_embedding=include_embedding,
        )


@pytest.mark.asyncio
async def test_spreading_activation_asks_for_neurons_without_the_vector() -> None:
    """The wiring, not just the store: activation is the dominant reader.

    Three golden queries issue 252 of their 273 point reads through
    ``get_neurons_batch`` and 68 of 71 traversals through ``get_neighbors`` from
    this one call path, so a store that *can* project but a caller that never asks
    would leave the whole measured gain on the table.
    """
    storage = _RecordingStorage()
    brain = Brain.create(name="test", config=BrainConfig())
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    a = Neuron(id="n-a", type=NeuronType.CONCEPT, content="anchor")
    b = Neuron(id="n-b", type=NeuronType.CONCEPT, content="neighbour")
    await storage.add_neuron(a)
    await storage.add_neuron(b)
    await storage.add_synapse(
        Synapse(id="s-1", source_id="n-a", target_id="n-b", type=SynapseType.RELATED_TO, weight=0.9)
    )

    activation = SpreadingActivation(storage, BrainConfig())
    await activation.activate(["n-a"], max_hops=1)

    assert storage.batch_calls, "activation did not batch-fetch its anchors"
    assert all(flag is False for flag in storage.batch_calls), storage.batch_calls
    assert storage.neighbor_calls, "activation did not walk any neighbours"
    assert all(flag is False for flag in storage.neighbor_calls), storage.neighbor_calls
