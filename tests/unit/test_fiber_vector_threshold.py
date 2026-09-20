"""The fiber-vector retriever applies its OWN similarity floor,
``fiber_vector_similarity_threshold`` (``core/brain.py``) — NOT the neuron-vector track's
``embedding_similarity_threshold`` (``_rank_knn_rows``, ``retrieval.py``).

Before U2 (smem-recall-tor-fibrowy), ``_sim`` was thrown away entirely (``[f.anchor_neuron_id
for f, _sim in fiber_hits if f.anchor_neuron_id]``), so a fiber with near-zero similarity to
the query got the same fixed top-N slot as a near-exact match. U2 first reused
``embedding_similarity_threshold`` for both tracks, but a measurement on the production golden
set (ABBA, `qa/anatomia-kotwic-*.json` in the program ledger) showed the shared knob cannot
serve both: the production value 0.52 drops two golden fiber anchors (``_sim`` 0.4879 and
0.5027), but lowering the SHARED threshold to recover them also loosens the neuron track's KNN
filter and measurably lets a negative-control query accumulate anchors it should not have had.
U2-REVISIT (B5) split the knob: ``fiber_vector_similarity_threshold`` (default 0.45, comfortably
below both measured similarities) gates only the fiber track; ``embedding_similarity_threshold``
(production 0.52) keeps gating the neuron track, untouched. ``_sim`` here is
``1 - vector::distance::knn()`` (``store.py``) — same scale as the neuron track, just a
different threshold value and a different config key.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.extraction.parser import Perspective, QueryIntent, Stimulus

_QUERY_VECTOR = [0.11, 0.22, 0.33]
# Deliberately DIFFERENT from each other: this is what makes the test suite fail if the
# fiber track ever reverts to reading the neuron track's threshold (or vice versa).
_FIBER_THRESHOLD = 0.45
_NEURON_THRESHOLD = 0.52


def _make_config() -> MagicMock:
    """Everything off except the fiber-vector step — the retriever under test in isolation."""
    config = MagicMock()
    config.max_context_tokens = 500
    config.max_spread_hops = 3
    config.activation_threshold = 0.1
    config.embedding_enabled = False
    config.embedding_similarity_threshold = _NEURON_THRESHOLD
    config.embedding_anchor_mode = "scan"
    config.idf_anchor_enabled = False
    config.fuzzy_search_enabled = False
    config.graph_expansion_enabled = False
    config.query_expansion_synonyms = {}
    config.query_expansion_abbreviations = {}
    config.query_expansion_max_per_term = 0
    config.fiber_vector_enabled = True
    config.fiber_vector_top_n = 10
    config.fiber_vector_similarity_threshold = _FIBER_THRESHOLD
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
    async def test_below_fiber_threshold_anchor_is_rejected(self, provider: AsyncMock) -> None:
        storage = _make_storage([(_fiber("below"), _FIBER_THRESHOLD - 0.01)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == []

    async def test_above_fiber_threshold_anchor_is_kept(self, provider: AsyncMock) -> None:
        storage = _make_storage([(_fiber("above"), _FIBER_THRESHOLD + 0.1)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == ["above"]

    async def test_similarity_equal_to_fiber_threshold_passes(self, provider: AsyncMock) -> None:
        """``sim >= threshold``, not ``>`` — the exact boundary value must still pass,
        mirroring ``_rank_knn_rows``'s own ``similarity >= threshold`` on the neuron-vector
        track."""
        storage = _make_storage([(_fiber("boundary"), _FIBER_THRESHOLD)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == ["boundary"]

    async def test_uses_fiber_threshold_not_neuron_threshold(self, provider: AsyncMock) -> None:
        """Regression guard for the whole point of U2-REVISIT/B5: the fiber track must read
        ``fiber_vector_similarity_threshold`` (0.45 here), NOT ``embedding_similarity_threshold``
        (0.52 here) — the two are deliberately configured to DIFFERENT values in
        ``_make_config``. A similarity of 0.48 sits strictly between them: it must pass the
        fiber filter (>= 0.45) even though it would be REJECTED by the neuron-track threshold
        (< 0.52). If the fiber track ever goes back to reading the shared
        ``embedding_similarity_threshold``, this anchor is dropped and the test fails."""
        storage = _make_storage([(_fiber("between"), 0.48)])
        pipeline = _pipeline(storage, provider)

        _sets, ranked_lists, _outcome = await pipeline._find_anchors_ranked(_make_stimulus())

        assert _fiber_retriever_anchors(ranked_lists) == ["between"]

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


class TestFiberVectorThresholdDefault:
    def test_default_is_045_and_lower_than_neuron_default(self) -> None:
        """0.45 is the measured floor (U2-REVISIT/B5, `core/brain.py` docstring: two golden
        fiber anchors at _sim 0.4879/0.5027 must clear it) and must stay strictly BELOW the
        neuron track's default `embedding_similarity_threshold` — this is a track-specific
        floor, not a global relaxation of the neuron track's default."""
        config = BrainConfig()

        assert config.fiber_vector_similarity_threshold == 0.45
        assert config.fiber_vector_similarity_threshold < config.embedding_similarity_threshold
