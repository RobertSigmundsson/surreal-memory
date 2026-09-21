"""U1 — program smem-recall-trzy-warstwy: `refusal_mode` ("off" | "observe" |
"enforce") on the weak_landscape_floor gate (`engine/sufficiency.py`), the
`signals` field on `RetrievalTrace` (`core/retrieval_trace.py`), and the
metadata -> trace hand-off in `engine/trace_builder.py`.

Source of truth: qa/user-stories/S1-observe-nigdy-nie-odmawia.feature,
qa/user-stories/S2-trzy-warstwy-w-jednym-sladzie.feature, qa/kryteria.md
(K3, K4, K10, K12).
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.engine.sufficiency import check_sufficiency
from surreal_memory.engine.trace_builder import build_retrieval_trace

# Thresholds from qa/kryteria.md (W3, "smem-recall-brama-odmowy" D1 §2a),
# repeated verbatim in the S1 feature's Założenia.
_MIN_NEURON_COUNT = 15
_MIN_ANCHOR_SIM = 0.524835


class _FakeActivation:
    """Minimal stand-in for ActivationResult (mirrors tests/unit/test_sufficiency.py)."""

    def __init__(
        self, activation_level: float, hop_distance: int = 2, source_anchor: str = "a-0"
    ) -> None:
        self.activation_level = activation_level
        self.hop_distance = hop_distance
        self.source_anchor = source_anchor


def _landscape(n: int) -> dict[str, _FakeActivation]:
    """The same moderate-activation, gates-1-4-inert landscape used by
    TestWeakLandscapeFloor in tests/unit/test_sufficiency.py — falls through
    gates 1-4 untouched so only the weak_landscape_floor condition is under
    test."""
    return {f"n-{i}": _FakeActivation(0.3 + 0.02 * i) for i in range(n)}


def _check_enforced(refusal_mode: str, *, neuron_count: int, anchor_sim_top1: float) -> Any:
    """ "off"/"enforce" only: ALWAYS passes the enforcement knobs
    (`min_neuron_count`/`min_anchor_sim`) -- the only params either mode
    reads (round-2 correction: they are deliberate synonyms)."""
    return check_sufficiency(
        activations=_landscape(neuron_count),
        anchor_sets=[["a-0"]],
        intersections=[],
        stab_converged=True,
        stab_neurons_removed=0,
        anchor_sim_top1=anchor_sim_top1,
        min_neuron_count=_MIN_NEURON_COUNT,
        min_anchor_sim=_MIN_ANCHOR_SIM,
        refusal_mode=refusal_mode,
    )


def _check_unconfigured(refusal_mode: str, *, neuron_count: int, anchor_sim_top1: float) -> Any:
    """The enforcement knobs are intentionally NOT passed (BrainConfig
    defaults 0/None, inert) -- used for "off" baseline and "observe"
    scenarios, where they must play no role."""
    return check_sufficiency(
        activations=_landscape(neuron_count),
        anchor_sets=[["a-0"]],
        intersections=[],
        stab_converged=True,
        stab_neurons_removed=0,
        anchor_sim_top1=anchor_sim_top1,
        refusal_mode=refusal_mode,
    )


class TestSufficiencyRefusalModes:
    """Round-2 correction (runner). Two mandates now under test:

    1. "off" and "enforce" are DELIBERATE SYNONYMS: both read ONLY the
       enforcement knobs (`min_neuron_count`/`min_anchor_sim`), exactly the
       gate's ORIGINAL unconditional logic -- "off" (the default) must
       never silently disable a knob an operator explicitly configured.
    2. "observe" is the ONLY mode that changes flow, and it evaluates its
       OWN, separate thresholds (`observe_min_neuron_count`/
       `observe_min_anchor_sim`) -- it must work, and must be measurable,
       with the enforcement knobs left untouched at their inert defaults.
    """

    def test_enforce_refuses_weak_landscape(self) -> None:
        result = _check_enforced("enforce", neuron_count=10, anchor_sim_top1=0.4)
        assert result.sufficient is False
        assert result.gate == "weak_landscape_floor"
        # Observability (would_refuse/signals) only ever runs in "observe".
        assert result.would_refuse is False
        assert result.signals == {}

    def test_off_still_refuses_when_enforcement_knob_is_set(self) -> None:
        """Pins round-2 correction 1: "off" (the default) must NOT
        silently disable a knob the operator explicitly configured --
        the exact failure mode that broke
        tests/unit/test_reranker_refusal_floor.py in round 1."""
        result = check_sufficiency(
            activations=_landscape(10),
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            min_neuron_count=15,
            refusal_mode="off",
        )
        assert result.sufficient is False
        assert result.gate == "weak_landscape_floor"

    def test_observe_never_refuses_the_same_landscape(self) -> None:
        """Differencing control: the SAME landscape as
        test_enforce_refuses_weak_landscape, only refusal_mode differs --
        and this time the enforcement knobs are NOT passed at all (they
        default to 0/None, inert), proving "observe" needs none of them."""
        result = check_sufficiency(
            activations=_landscape(10),
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            anchor_sim_top1=0.4,
            refusal_mode="observe",
        )
        assert result.sufficient is True
        assert result.would_refuse is True
        assert result.would_refuse_gate == "weak_landscape_floor"
        assert "neuron_count" in result.signals

    def test_observe_uses_its_own_thresholds_not_enforcement_knobs(self) -> None:
        """Required by round-2 correction 2, literally: a landscape below
        the OBSERVE thresholds (defaults 15 / 0.524835, D1 §2a), with the
        enforcement knobs left at their inert defaults ("cannot fire") --
        "observe" must still set would_refuse. Proves turning on
        observation needs zero enforcement-knob configuration."""
        result = check_sufficiency(
            activations=_landscape(10),
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            anchor_sim_top1=0.4,
            refusal_mode="observe",
            # min_neuron_count/min_anchor_sim intentionally NOT passed.
        )
        assert result.sufficient is True
        assert result.would_refuse is True

    def test_observe_ignores_configured_enforcement_knobs(self) -> None:
        """Stronger proof than the two tests above: the enforcement knobs
        are explicitly SET here, to values that would NOT themselves trip
        (`min_neuron_count=5 <= 10`, `min_anchor_sim=0.1 <= 0.4`) -- yet
        "observe"'s OWN thresholds (defaults 15/0.524835) still fire for
        this landscape. If "observe" were reading the enforcement knobs at
        all, this landscape would NOT trip anything; it does, so the
        knobs were never consulted."""
        result = check_sufficiency(
            activations=_landscape(10),
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            anchor_sim_top1=0.4,
            min_neuron_count=5,
            min_anchor_sim=0.1,
            refusal_mode="observe",
        )
        assert result.sufficient is True
        assert result.would_refuse is True
        assert result.would_refuse_gate == "weak_landscape_floor"

    def test_off_has_no_signals_and_no_refusal(self) -> None:
        """ "off" with the enforcement knobs left at their inert defaults
        (0 / None, "cannot fire") -- today's baseline-safe behaviour."""
        result = _check_unconfigured("off", neuron_count=10, anchor_sim_top1=0.4)
        assert result.sufficient is True
        assert result.would_refuse is False
        assert result.would_refuse_gate == ""
        assert result.signals == {}

    def test_observe_marks_not_would_refuse_for_healthy_landscape(self) -> None:
        """K4: signals must exist even when the gate would NOT have refused."""
        result = _check_unconfigured("observe", neuron_count=20, anchor_sim_top1=0.9)
        assert result.sufficient is True
        assert result.would_refuse is False
        assert result.would_refuse_gate == ""
        assert result.signals != {}
        assert result.signals["neuron_count"] == 20

    def test_default_refusal_mode_preserves_today(self) -> None:
        """No refusal_mode passed at all -- proves the default is "enforce",
        i.e. every pre-existing caller (incl.
        TestWeakLandscapeFloor::test_neuron_count_below_floor_refuses) keeps
        today's behaviour unmodified."""
        result = check_sufficiency(
            activations=_landscape(10),
            anchor_sets=[["a-0"]],
            intersections=[],
            stab_converged=True,
            stab_neurons_removed=0,
            anchor_sim_top1=0.4,
            min_neuron_count=_MIN_NEURON_COUNT,
            min_anchor_sim=_MIN_ANCHOR_SIM,
        )
        assert result.sufficient is False
        assert result.gate == "weak_landscape_floor"


class TestRetrievalTraceSignals:
    """S2: signals round-trip through RetrievalTrace, bounded by contract."""

    def test_retrieval_trace_carries_signals(self) -> None:
        raw_signals = {
            "w3_would_refuse": True,
            "w3_gate": "weak_landscape_floor",
            "neuron_count": 10,
        }
        trace = RetrievalTrace(brain_id="b1", signals=raw_signals)
        assert trace.trace_version == 2

        as_dict = trace.to_dict()
        assert as_dict["signals"] == raw_signals

        rebuilt = RetrievalTrace.from_dict(as_dict)
        assert rebuilt.signals == raw_signals
        assert rebuilt.trace_version == 2

        # An old row with no trace_version/signals key at all (written before
        # this field existed) must read back as version 1, not silently
        # upgraded.
        old_row = RetrievalTrace.from_dict({})
        assert old_row.trace_version == 1
        assert old_row.signals == {}

    def test_trace_signals_are_bounded(self) -> None:
        # Non-scalars first so they are exercised well inside the 24-key cap
        # rather than possibly never reached by the loop.
        signals: dict[str, Any] = {"listy": [1, 2, 3], "dlugi": "x" * 500}
        for i in range(50):
            signals[f"k{i}"] = i

        trace = RetrievalTrace(brain_id="b1", signals=signals)
        assert len(trace.signals) <= 24
        assert "listy" not in trace.signals
        assert "dlugi" in trace.signals
        assert len(trace.signals["dlugi"]) == 120


class TestBuildRetrievalTraceSignals:
    """S2: build_retrieval_trace moves engine/retrieval.py's
    metadata["odmowa_sygnaly"] into RetrievalTrace.signals."""

    def _result(self, metadata: dict[str, Any]) -> RetrievalResult:
        return RetrievalResult(
            answer="a",
            confidence=0.5,
            depth_used=DepthLevel.CONTEXT,
            neurons_activated=3,
            fibers_matched=[],
            subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
            context="ctx",
            latency_ms=12.0,
            synthesis_method="default_pass",
            metadata=metadata,
        )

    def test_build_retrieval_trace_moves_signals_from_metadata(self) -> None:
        odmowa_sygnaly = {
            "w3_would_refuse": True,
            "w3_gate": "weak_landscape_floor",
            "neuron_count": 10,
            "anchor_sim_top1": 0.4,
            "rerank_raw_top1": None,
            "refusal_mode": "observe",
        }
        result = self._result({"odmowa_sygnaly": odmowa_sygnaly})
        trace = build_retrieval_trace(result, query="q", brain_id="b1", mode="fast")
        assert trace.signals == odmowa_sygnaly

    def test_build_retrieval_trace_defaults_to_empty_signals(self) -> None:
        result = self._result({})
        trace = build_retrieval_trace(result, query="q", brain_id="b1", mode="fast")
        assert trace.signals == {}


# ---------------------------------------------------------------------------
# Program smem-recall-trzy-warstwy, RUNNER REVIEW of U1: "M4 could not be
# measured" must never be recorded as "M4 would not have refused".
#
# The enforcement path already names its skip (`reranker_floor_skipped`). In
# observation the raw cross-encoder score can be missing for the very same
# reasons (reranker degraded, no candidate contents, early exit at 4.8) — and
# a signal that quietly reports False there would inflate the "this layer
# would have let the query through" count in the weekly report. Harness is the
# pattern from tests/unit/test_reranker_refusal_floor.py: drive the REAL
# pipeline over InMemoryStorage and replace only the two seams that would
# otherwise need a live reranker endpoint.
# ---------------------------------------------------------------------------

_OBS_QUERY = "where does Emma live in Oslo Norway"


def _obs_config(**overrides: Any) -> BrainConfig:
    import dataclasses

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
async def obs_storage():
    from surreal_memory.core.fiber import Fiber
    from surreal_memory.core.neuron import Neuron, NeuronType
    from surreal_memory.storage.memory_store import InMemoryStorage

    s = InMemoryStorage()
    brain = Brain.create(name="trzy_warstwy_observe_test")
    await s.save_brain(brain)
    s.set_brain(brain.id)
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


def _obs_reranker_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from surreal_memory.unified_config import RerankerConfig, get_config

    patched = dataclasses.replace(
        get_config(),
        reranker=RerankerConfig(enabled=True, endpoint="http://fake-reranker.invalid/v1"),
    )
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda reload=False: patched)


def _obs_fake_rerank(raw_top1: float | None, degraded_reason: str | None) -> Any:
    def _fn(query: str, activations: dict, neuron_contents: dict, **kwargs: Any) -> dict:
        if degraded_reason is not None:
            kwargs["on_degraded"](degraded_reason)
        elif raw_top1 is not None:
            kwargs["on_raw_top1"](raw_top1)
        return activations

    return _fn


class TestObserveNeverReportsUnmeasuredAsNegative:
    """`m4_measured` separates "would not refuse" from "could not tell"."""

    async def test_measured_score_above_floor_is_marked_measured(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["m4_measured"] is True
        assert sygnaly["m4_unmeasured_reason"] is None
        assert sygnaly["rerank_raw_top1"] == pytest.approx(0.9)

    async def test_degraded_reranker_is_unmeasured_not_a_negative(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: degraded => `m4_measured` False WITH a named reason,
        never a quiet `w3_would_refuse=False` that the weekly report would count
        as "this layer would have let it through"."""
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=None, degraded_reason="reranker endpoint unreachable"),
        )
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["m4_measured"] is False
        assert sygnaly["m4_unmeasured_reason"] == "reranker endpoint unreachable"
        # answer still returned normally — observation never refuses
        assert result.synthesis_method != "insufficient_signal"

    async def test_off_mode_has_no_signals_key_at_all(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )
        pipeline = ReflexPipeline(obs_storage, _obs_config())  # refusal_mode default "off"
        result = await pipeline.query(_OBS_QUERY)
        assert "odmowa_sygnaly" not in result.metadata
