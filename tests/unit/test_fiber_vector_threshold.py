"""The fiber-vector retriever must apply the same similarity threshold the neuron-vector
track already applies in ``_rank_knn_rows`` (``similarity >= threshold``) — before this fix,
``_sim`` was thrown away entirely (``[f.anchor_neuron_id for f, _sim in fiber_hits if
f.anchor_neuron_id]``), so a fiber with near-zero similarity to the query got the same fixed
top-N slot as a near-exact match. ``_sim`` here is ``1 - vector::distance::knn()``
(``store.py``), the same scale as the neuron track's ``embedding_similarity_threshold`` — so
these tests pin REUSE of that existing config value, not a new key.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.extraction.parser import Perspective, QueryIntent, Stimulus

_QUERY_VECTOR = [0.11, 0.22, 0.33]
_THRESHOLD = 0.52


def _make_config() -> MagicMock:
    """Everything off except the fiber-vector step — the retriever under test in isolation."""
    config = MagicMock()
    config.max_context_tokens = 500
    config.max_spread_hops = 3
    config.activation_threshold = 0.1
    config.embedding_enabled = False
    config.embedding_similarity_threshold = _THRESHOLD
    config.embedding_anchor_mode = "scan"
    config.idf_anchor_enabled = False
    config.fuzzy_search_enabled = False
    config.graph_expansion_enabled = False
    config.query_expansion_synonyms = {}
    config.query_expansion_abbreviations = {}
    config.query_expansion_max_per_term = 0
    config.fiber_vector_enabled = True
    config.fiber_vector_top_n = 10
    return config


def _make_stimulus(query: str = "what did we decide about the vector index") -> Stimulus:
    return Stimulus(
        time_hints=[],
        keywords=["vector", "index"],
        entities=[],
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query=query,
    )


def _fiber(anchor_id: str) -> Fiber:
    return Fiber.create(neuron_ids={anchor_id}, synapse_ids=set(), anchor_neuron_id=anchor_id)


def _make_storage(hits: list[tuple[Fiber, float]]) -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers_batch = AsyncMock(return_value=[])
    storage.get_neurons_batch = AsyncMock(return_value={})
    storage.get_fibers = AsyncMock(return_value=[])
    storage.get_synapses_for_neurons = AsyncMock(return_value={})
    storage.find_fibers_by_embedding = AsyncMock(return_value=hits)
    return storage


@pytest.fixture
def provider() -> AsyncMock:
    p = AsyncMock()
    p.embed = AsyncMock(return_value=list(_QUERY_VECTOR))
    p.similarity = AsyncMock(return_value=0.9)
    return p


def _pipeline(storage: AsyncMock, provider: AsyncMock) -> ReflexPipeline:
    return ReflexPipeline(
        storage=storage, config=_make_config(), use_reflex=True, embedding_provider=provider
    )


def _fiber_retriever_anchors(ranked_lists: list[list]) -> list[str]:
    return [a.neuron_id for lst in ranked_lists for a in lst if a.retriever == "fiber_vector"]


class TestFiberVectorThreshold:
    async def test_below_threshold_anchor_is_rejected(self, provider: AsyncMock) -> None:
        storage = _make_storage([(_fiber("below"), _THRESHOLD - 0.01)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == []

    async def test_above_threshold_anchor_is_kept(self, provider: AsyncMock) -> None:
        storage = _make_storage([(_fiber("above"), _THRESHOLD + 0.1)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == ["above"]

    async def test_similarity_equal_to_threshold_passes(self, provider: AsyncMock) -> None:
        """``sim >= threshold``, not ``>`` — the exact boundary value must still pass,
        mirroring ``_rank_knn_rows``'s own ``similarity >= threshold`` on the neuron-vector
        track."""
        storage = _make_storage([(_fiber("boundary"), _THRESHOLD)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == ["boundary"]

    async def test_all_anchors_filtered_out_adds_no_empty_list(self, provider: AsyncMock) -> None:
        """Gate ``no_anchors`` (``sufficiency.py``) only fires when ``anchor_sets`` truly has
        no anchors — an empty list appended by the fiber track after filtering would silently
        defeat it. Pins that the existing ``if fiber_anchor_ids:`` guard still holds after the
        threshold filter removes every candidate, not just when the storage call itself
        returns nothing (RUNBOOK smem-recall-tor-fibrowy §U2/A)."""
        storage = _make_storage(
            [(_fiber("below-a"), 0.1), (_fiber("below-b"), 0.2)],
        )
        pipeline = _pipeline(storage, provider)

        anchor_sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert anchor_sets == []
        assert ranked_lists == []
