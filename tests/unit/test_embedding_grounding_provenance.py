"""Tests for semantic vector anchor provenance in retrieval confidence."""

from unittest.mock import AsyncMock

import pytest

from surreal_memory.engine.activation import ActivationResult
from surreal_memory.engine.reconstruction import reconstruct_answer
from surreal_memory.engine.retrieval import _has_embedding_anchor
from surreal_memory.engine.score_fusion import RankedAnchor


def test_fiber_vector_anchor_counts_as_embedding_grounding() -> None:
    ranked_lists = [[RankedAnchor(neuron_id="n1", rank=1, retriever="fiber_vector")]]

    assert _has_embedding_anchor(ranked_lists)


def test_ungrounded_anchor_does_not_count_as_embedding_grounding() -> None:
    ranked_lists = [[RankedAnchor(neuron_id="n1", rank=1, retriever="keyword")]]

    assert not _has_embedding_anchor(ranked_lists)


@pytest.mark.parametrize(
    ("retriever", "expected_grounded", "expected_confidence"),
    [("fiber_vector", True, 0.9), ("keyword", False, 0.54)],
)
@pytest.mark.asyncio
async def test_vector_anchor_grounding_reaches_confidence_reconstruction(
    retriever: str, expected_grounded: bool, expected_confidence: float
) -> None:
    storage = AsyncMock()
    storage.get_neuron_state.return_value = None
    storage.get_synapses.return_value = []
    storage.get_neuron.return_value = None
    storage.get_neurons_batch.return_value = {}
    ranked_lists = [[RankedAnchor(neuron_id="n1", rank=1, retriever=retriever)]]
    activations = {
        "n1": ActivationResult(
            neuron_id="n1",
            activation_level=0.9,
            hop_distance=0,
            path=["n1"],
            source_anchor="n1",
        )
    }

    result = await reconstruct_answer(
        storage,
        activations,
        intersections=[],
        fibers=[],
        has_embedding_anchor=_has_embedding_anchor(ranked_lists),
    )

    assert result.score_breakdown is not None
    assert result.score_breakdown.embedding_grounded is expected_grounded
    assert result.score_breakdown.raw_total == pytest.approx(expected_confidence)
