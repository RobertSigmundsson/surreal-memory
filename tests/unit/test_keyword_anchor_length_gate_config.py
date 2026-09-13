"""The keyword-anchor length gate is a BRAIN SETTING, not a literal at the call site.

`3852d4f1` moved the N1 gate (`min_content_len=25`) out of `ReflexPipeline` and onto
`BrainConfig.keyword_anchor_min_content_len`. 25 is an empirical number measured on one
brain's 49-pair golden — it tied with the un-gated variant and lost nothing — so another
brain must be able to move it without a release, and `0` must disable the gate entirely.

Written during the 3.10.0 replay because that commit shipped without a test of its own:
its evidence (`qa/dowod-u24a.json`) was a one-off probe in a ledger, and nothing in the
suite would have noticed the call site drifting back to a literal. Both assertions fail
on 04fb0a1e, where the field does not exist.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.extraction.parser import Perspective, QueryIntent, Stimulus


def _config(prog: int) -> BrainConfig:
    """Default brain with everything that would issue extra queries turned off."""
    return replace(
        BrainConfig(),
        keyword_anchor_min_content_len=prog,
        embedding_enabled=False,
        idf_anchor_enabled=False,
        fuzzy_search_enabled=False,
        graph_expansion_enabled=False,
        fiber_vector_enabled=False,
    )


def _stimulus() -> Stimulus:
    return Stimulus(
        time_hints=[],
        keywords=["deployment"],
        entities=[],
        intent=QueryIntent.RECALL,
        perspective=Perspective.RECALL,
        raw_query="what did we decide about the deployment",
    )


def _storage() -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons_ranked = AsyncMock(return_value=[])
    storage.find_neurons = AsyncMock(return_value=[])
    storage.find_fibers = AsyncMock(return_value=[])
    storage.find_fibers_batch = AsyncMock(return_value=[])
    storage.get_neurons_batch = AsyncMock(return_value={})
    storage.get_synapses_batch = AsyncMock(return_value={})
    storage.get_neuron_states_batch = AsyncMock(return_value={})
    return storage


def test_the_gate_is_a_brain_config_field_with_the_measured_default() -> None:
    assert hasattr(BrainConfig(), "keyword_anchor_min_content_len"), (
        "the gate is hard-coded at the call site again — a brain cannot move it"
    )
    assert BrainConfig().keyword_anchor_min_content_len == 25


@pytest.mark.parametrize("prog", [25, 0, 999])
async def test_the_configured_value_is_what_reaches_find_neurons_ranked(prog: int) -> None:
    """Whatever the brain says is what the query gets — including 0, which disables it."""
    storage = _storage()
    pipeline = ReflexPipeline(storage=storage, config=_config(prog), use_reflex=True)

    await pipeline._find_anchors_ranked(_stimulus())

    assert storage.find_neurons_ranked.await_count >= 1, "no keyword anchor query was issued"
    przekazane = {
        call.kwargs.get("min_content_len") for call in storage.find_neurons_ranked.await_args_list
    }
    assert przekazane == {prog}, (
        f"the pipeline passed {przekazane} instead of the configured {prog} — the gate is not "
        "read from BrainConfig"
    )
