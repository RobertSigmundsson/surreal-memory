"""M4 refusal gate (``BrainConfig.reranker_refusal_floor``, smem-recall-brama-odmowy,
U2 DIAGNOZA.md §7/§8) — the second decision point AFTER step 4.9 (post-rerank) in
``engine/retrieval.py``.

Recall's sufficiency gates (``engine/sufficiency.py``) score the SHAPE of the activation
landscape, not whether it actually answers the query — measured on a 27-phrase
out-of-base set plus the 49-pair golden, every existing gate accepted unconditionally
(0/98 refusals possible before this gate existed). The reranker's raw cross-encoder
score is the only signal in the pipeline that reads the (query, content) pair itself:
at the measured log-margin threshold it refuses 17/27 out-of-base phrases while
refusing ZERO golden queries, AUC 0.9728 (best of ten signals measured, DIAGNOZA.md §2).

Drives the REAL pipeline through ``InMemoryStorage`` (pattern: test_valid_at_pipeline.py)
rather than mocking retrieval internals, and replaces only the two seams that would
otherwise need a live reranker HTTP endpoint: ``unified_config.get_config`` (reranking
is deployment/runtime config, read fresh per call — KONTEKST fact 9) and
``engine.reranker.rerank_activations`` itself (both imported locally inside
``ReflexPipeline.query()``, so patching the module attribute before the call is picked
up — same technique the reranker module's own ``on_degraded``/``on_raw_top1`` callbacks
are designed for).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import RerankerConfig, get_config

_QUERY = "where does Emma live in Oslo Norway"


def _config(**overrides: Any) -> BrainConfig:
    return dataclasses.replace(
        BrainConfig(),
        embedding_enabled=False,
        idf_anchor_enabled=False,
        fuzzy_search_enabled=False,
        graph_expansion_enabled=False,
        fiber_vector_enabled=False,
        **overrides,
    )


@pytest.fixture
async def storage() -> InMemoryStorage:
    s = InMemoryStorage()
    brain = Brain.create(name="brama_odmowy_test")
    await s.save_brain(brain)
    s.set_brain(brain.id)

    # Two neurons sharing keyword anchors ("oslo"/"norway") with the query so
    # spreading activation ends up with `len(activations) > 1` — the reranker
    # block in retrieval.py only runs above that count.
    n1 = Neuron.create(type=NeuronType.CONCEPT, content="Emma lives in Oslo Norway")
    n2 = Neuron.create(type=NeuronType.CONCEPT, content="Oslo Norway has many fjords nearby")
    await s.add_neuron(n1)
    await s.add_neuron(n2)
    await s.add_fiber(
        Fiber.create(
            neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id, summary=n1.content
        )
    )
    await s.add_fiber(
        Fiber.create(
            neuron_ids={n2.id}, synapse_ids=set(), anchor_neuron_id=n2.id, summary=n2.content
        )
    )
    yield s
    await s.close()


def _reranker_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reranking is read from app config, not BrainConfig (KONTEKST fact 9) — patch the
    seam ``retrieval.py`` actually imports from (``from surreal_memory.unified_config
    import get_config as _get_app_config``, a fresh local import per call)."""
    real = get_config()
    patched = dataclasses.replace(
        real, reranker=RerankerConfig(enabled=True, endpoint="http://fake-reranker.invalid/v1")
    )
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda reload=False: patched)


def _fake_rerank_activations(raw_top1: float | None, degraded_reason: str | None) -> Any:
    """Stand-in for ``engine.reranker.rerank_activations`` — fires the SAME two
    callbacks the real function fires (``on_degraded`` xor ``on_raw_top1``), leaves
    ``activations`` bit-for-bit unchanged, and never touches the network."""

    def _fn(query: str, activations: dict, neuron_contents: dict, **kwargs: Any) -> dict:
        if degraded_reason is not None:
            kwargs["on_degraded"](degraded_reason)
        elif raw_top1 is not None:
            kwargs["on_raw_top1"](raw_top1)
        return activations

    return _fn


class TestRerankerFloorRefuses:
    async def test_raw_top1_below_floor_refuses_with_reranker_floor_gate(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _fake_rerank_activations(raw_top1=0.001, degraded_reason=None),
        )
        pipeline = ReflexPipeline(storage, _config(reranker_refusal_floor=0.002146))

        result = await pipeline.query(_QUERY)

        assert result.synthesis_method == "insufficient_signal"
        assert result.metadata["sufficiency_gate"] == "reranker_floor"
        assert result.metadata["sufficiency_confidence"] == pytest.approx(0.001)
        assert result.answer is None


class TestRerankerFloorBoundary:
    @pytest.mark.parametrize("raw_top1", [0.002146, 0.5])
    async def test_raw_top1_at_or_above_floor_does_not_refuse(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch, raw_top1: float
    ) -> None:
        """Boundary control in both directions: exactly AT the floor (`<`, not `<=`,
        so equal must pass) and comfortably above it."""
        _reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _fake_rerank_activations(raw_top1=raw_top1, degraded_reason=None),
        )
        pipeline = ReflexPipeline(storage, _config(reranker_refusal_floor=0.002146))

        result = await pipeline.query(_QUERY)

        assert result.metadata.get("sufficiency_gate") != "reranker_floor"


class TestRerankerFloorDegraded:
    async def test_rerank_degraded_skips_floor_and_flags_metadata(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cisza nie jest sukcesem: a degraded rerank must not refuse on an
        untrustworthy raw score, AND must say so in the metadata, not stay silent."""
        _reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _fake_rerank_activations(raw_top1=None, degraded_reason="forced-for-test"),
        )
        pipeline = ReflexPipeline(storage, _config(reranker_refusal_floor=0.002146))

        result = await pipeline.query(_QUERY)

        assert result.metadata.get("sufficiency_gate") != "reranker_floor"
        assert result.metadata.get("reranker_floor_skipped") == "forced-for-test"


class TestRerankerFloorDefaultOff:
    async def test_disabled_by_default_no_refusal_even_on_a_very_low_raw_score(
        self, storage: InMemoryStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An old brain (`reranker_refusal_floor` absent -> BrainConfig default None)
        must see today's behaviour: no refusal from this gate, no matter how low the
        raw score would have been."""
        _reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _fake_rerank_activations(raw_top1=0.0000001, degraded_reason=None),
        )
        pipeline = ReflexPipeline(storage, _config())  # reranker_refusal_floor left at default

        result = await pipeline.query(_QUERY)

        assert result.metadata.get("sufficiency_gate") != "reranker_floor"
        assert "reranker_floor_skipped" not in result.metadata
