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


# ---------------------------------------------------------------------------
# Program smem-recall-trzy-warstwy, unit U2: W3 leksyka/gibberish layer.
#
# `engine/leksyka.py` emulates the `smem_content` analyzer's tokenization;
# `storage/surrealdb/store.py::SurrealDBStorage.any_neuron_matches_any_token`
# is the storage primitive (one `content @@ 't1' OR content @@ 't2' ...`
# query); `engine/retrieval.py::ReflexPipeline._leksyka_sygnal` wires the two
# together, ONLY in `refusal_mode="observe"`, and follows the same "unmeasured
# is never a negative" discipline as the M4 signal above.
# ---------------------------------------------------------------------------


class TestTokenizuj:
    """U2: `tokenizuj`/`tokeny_do_sprawdzenia` emulate the `smem_content`
    analyzer contract (`blank`+`class` tokenizers, `lowercase`+`ascii`
    filters) without touching a database."""

    def test_tokenizuj_matches_analyzer_contract(self) -> None:
        from surreal_memory.engine.leksyka import tokenizuj

        tokens = tokenizuj("Nautilus 2026 termopastą uruboros_kafka")
        assert tokens == ["nautilus", "2026", "termopasta", "uruboros", "kafka"]

    def test_tokeny_do_sprawdzenia_filters_and_caps(self) -> None:
        from surreal_memory.engine.leksyka import tokeny_do_sprawdzenia

        # Below min_len is dropped ("ab", "abc", "six" has length 3);
        # duplicates ("four" twice) keep only the first occurrence, in
        # original order.
        result = tokeny_do_sprawdzenia("ab abc four five four six seven", min_len=4)
        assert result == ["four", "five", "seven"]

        # Cap at 32, first-32-in-order, no duplicates counted twice against
        # the cap. Suffixes are two ASCII letters (not digits) so the
        # `class` tokenizer's letter/digit boundary split never fragments
        # them -- each "wordXX" is one token, all distinct, all >= min_len.
        words = [f"word{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(50)]
        capped = tokeny_do_sprawdzenia(" ".join(words), min_len=4)
        assert len(capped) == 32
        assert capped == words[:32]


class TestAnyNeuronMatchesAnyTokenValidation:
    """U2: storage-layer fail-closed validation — never trust that the caller
    already filtered tokens, since they originate in user query text."""

    async def test_storage_method_rejects_unsafe_token(self) -> None:
        from surreal_memory.storage.surrealdb.store import SurrealDBStorage

        storage = SurrealDBStorage()
        with pytest.raises(ValueError):
            await storage.any_neuron_matches_any_token(["a' OR 1=1"])

        queried = False

        async def _fake_query(sql: str, **params: Any) -> list[dict[str, Any]]:
            nonlocal queried
            queried = True
            return []

        storage._query = _fake_query  # type: ignore[method-assign]
        assert await storage.any_neuron_matches_any_token([]) is False
        assert queried is False

    async def test_both_backends_agree_on_lexical_lookup(self) -> None:
        """`SurrealDBStorage` and `InMemoryStorage` must give the SAME verdict for
        the same tokens over the same content (tests/unit/test_storage_parity.py
        requires the method to exist on both; this pins that it also BEHAVES the
        same, not just that the name resolves). Matching is per analyzer TOKEN,
        not raw substring: `address` contains the substring `ddre`, but no token
        of `address` (tokenized: `["address"]`) equals `ddre` — the same way
        SurrealDB's BM25 `content @@ 'ddre'` would not match it either.
        """
        from surreal_memory.core.brain import Brain
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()
        brain = Brain.create(name="lexical_parity_test")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)
        await storage.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content="Emma lives in Oslo Norway")
        )
        await storage.add_neuron(
            Neuron.create(type=NeuronType.CONCEPT, content="please note the address below")
        )

        assert await storage.any_neuron_matches_any_token(["oslo"]) is True
        assert await storage.any_neuron_matches_any_token(["qwzlmnprt", "vxbdfghj"]) is False
        assert await storage.any_neuron_matches_any_token(["ddre"]) is False

        await storage.close()


class TestLeksykaObserveSignal:
    """U2: the pipeline wiring, driven end-to-end over `obs_storage`
    (InMemoryStorage) the same way as `TestObserveNeverReportsUnmeasuredAsNegative`
    above. `InMemoryStorage` now implements `any_neuron_matches_any_token` for
    real (parity with `SurrealDBStorage`, see `TestAnyNeuronMatchesAnyTokenValidation
    .test_both_backends_agree_on_lexical_lookup`); these tests still override it per
    test via `monkeypatch.setattr` on the fixture INSTANCE so the pipeline-wiring
    verdict is deterministic and independent of `obs_storage`'s actual two neurons."""

    async def test_observe_flags_gibberish_when_no_token_found(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )

        async def _no_match(tokens: list[str]) -> bool:
            return False

        monkeypatch.setattr(obs_storage, "any_neuron_matches_any_token", _no_match, raising=False)
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["leksyka_would_refuse"] is True
        assert sygnaly["leksyka_zbadana"] is True
        assert sygnaly["leksyka_niezbadana_powod"] is None
        assert sygnaly["leksyka_tokenow"] > 0

    async def test_observe_does_not_flag_when_a_token_is_found(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )

        async def _match(tokens: list[str]) -> bool:
            return True

        monkeypatch.setattr(obs_storage, "any_neuron_matches_any_token", _match, raising=False)
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["leksyka_would_refuse"] is False
        assert sygnaly["leksyka_zbadana"] is True

    async def test_off_mode_makes_no_lexical_query(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )
        calls = 0

        async def _counting(tokens: list[str]) -> bool:
            nonlocal calls
            calls += 1
            return False

        monkeypatch.setattr(obs_storage, "any_neuron_matches_any_token", _counting, raising=False)
        pipeline = ReflexPipeline(obs_storage, _obs_config())  # refusal_mode default "off"
        result = await pipeline.query(_OBS_QUERY)
        assert calls == 0
        assert "odmowa_sygnaly" not in result.metadata

    async def test_lexical_failure_is_named_not_silent(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The most important test in this unit: a raising storage must
        neither crash the recall nor be reported as `leksyka_would_refuse=
        False` (a silent, wrong "this layer let it through") -- it must be
        named `None`/unmeasured, with the exception text in the reason."""
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_reranker_enabled(monkeypatch)
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )

        async def _boom(tokens: list[str]) -> bool:
            raise RuntimeError("db down")

        monkeypatch.setattr(obs_storage, "any_neuron_matches_any_token", _boom, raising=False)
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["leksyka_would_refuse"] is None
        assert sygnaly["leksyka_zbadana"] is False
        assert sygnaly["leksyka_niezbadana_powod"] is not None
        assert "db down" in sygnaly["leksyka_niezbadana_powod"]
        # recall must still answer normally -- observation never refuses.
        assert result.synthesis_method != "insufficient_signal"


# ---------------------------------------------------------------------------
# Program smem-recall-trzy-warstwy, unit U3: Jev (TypeSafe System One)
# refusal-observability signal.
#
# `engine/jev_pytania.py` holds the (versioned) questions + starting
# threshold; `engine/jev_gate.py` is the stdlib-`urllib` HTTP client
# (`zapytaj_jev`/`OdpowiedzJev`/`redaguj`); `engine/retrieval.py` wires it in
# right after step 4.9, gated on BOTH `refusal_mode == "observe"` and
# `jev.mode == "observe"`. Direct tests below drive `zapytaj_jev` itself,
# monkeypatching the ONE seam `jev_gate._blocking_post` (never a real
# connection, per the program's hard network ban). The two pipeline tests
# reuse the `obs_storage`/`_obs_config`/`_obs_fake_rerank` harness above,
# patching `[jev]` the same way `_obs_reranker_enabled` patches `[reranker]`.
# ---------------------------------------------------------------------------


def _obs_jev_enabled(
    monkeypatch: pytest.MonkeyPatch,
    *,
    jev_mode: str = "observe",
    api_key_env: str = "SMEM_TEST_JEV_KEY",
) -> None:
    """Patches BOTH `[reranker]` (so `neuron_contents` gets populated -- Jev's
    candidates are the reranker's fetched content, no second storage read)
    and `[jev]` on the effective app config, the same seam
    `_obs_reranker_enabled` uses alone."""
    import dataclasses

    from surreal_memory.unified_config import JevConfig, RerankerConfig, get_config

    patched = dataclasses.replace(
        get_config(),
        reranker=RerankerConfig(enabled=True, endpoint="http://fake-reranker.invalid/v1"),
        jev=JevConfig(mode=jev_mode, api_key_env=api_key_env),
    )
    monkeypatch.setattr("surreal_memory.unified_config.get_config", lambda reload=False: patched)


def _jev_ok_body(**answers_overrides: Any) -> bytes:
    import json as _json

    answers = {
        "odpowiada": {"type": "noul", "noul": 0.97},
        "sensowne": {"type": "noul", "noul": 0.98},
        "ta_domena": {"type": "noul", "noul": 0.9},
        "jakosc": {"type": "score", "score": 3, "confidence": 0.8},
    }
    answers.update(answers_overrides)
    return _json.dumps(
        {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 401}}
    ).encode("utf-8")


class TestJevGateDirect:
    """Direct tests of `engine/jev_gate.zapytaj_jev` -- no pipeline, no
    storage, the atrapa is `jev_gate._blocking_post` (the ONE HTTP-response
    mock this program allows, per the runner's hard ban on any other mock in
    production code paths)."""

    async def test_jev_body_shape_has_pin_and_no_extra_top_level(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json as _json

        from surreal_memory.engine import jev_gate

        captured: dict[str, Any] = {}

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            captured["url"] = url
            captured["body"] = _json.loads(body)
            captured["headers"] = headers
            return 200, _jev_ok_body()

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        result = await jev_gate.zapytaj_jev(
            query="gdzie mieszka Emma",
            memories="Emma mieszka w Oslo",
            gateway_url="http://127.0.0.1:4001/typesafe/v1/systemone",
            api_key="test-key",
            model="jev-1.13.0",
            timeout_ms=2000,
            sekrety=[],
        )
        assert result.status == "OK"
        body = captured["body"]
        assert set(body.keys()) == {"state", "model", "questions"}
        assert body["model"] == "jev-1.13.0"
        assert set(body["questions"].keys()) == {
            "odpowiada",
            "sensowne",
            "ta_domena",
            "jakosc",
        }
        assert captured["headers"]["Authorization"] == "Bearer test-key"

    async def test_jev_timeout_is_niedostepny_not_a_low_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine import jev_gate

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            raise TimeoutError("timed out")

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        result = await jev_gate.zapytaj_jev(
            query="q",
            memories="m",
            gateway_url="http://127.0.0.1:4001/typesafe/v1/systemone",
            api_key="test-key",
            model="jev-1.13.0",
            timeout_ms=50,
            sekrety=[],
        )
        assert result.status == "JEV_NIEDOSTEPNY"
        assert result.odpowiada is None
        assert result.sensowne is None
        assert result.ta_domena is None
        assert result.jakosc is None
        assert result.jakosc_conf is None
        assert result.tok is None
        assert result.powod is not None

    async def test_jev_403_is_odrzucil_with_named_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine import jev_gate

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            return 403, b"forbidden: prefixed model rejected"

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        result = await jev_gate.zapytaj_jev(
            query="q",
            memories="m",
            gateway_url="http://127.0.0.1:4001/typesafe/v1/systemone",
            api_key="test-key",
            model="typesafe/jev-1.13.0",
            timeout_ms=2000,
            sekrety=[],
        )
        assert result.status == "JEV_ODRZUCIL"
        assert result.powod is not None
        assert "403" in result.powod
        assert result.odpowiada is None

    async def test_jev_missing_key_never_calls_network(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine import jev_gate

        wywolania = 0

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            nonlocal wywolania
            wywolania += 1
            return 200, _jev_ok_body()

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        result = await jev_gate.zapytaj_jev(
            query="q",
            memories="m",
            gateway_url="http://127.0.0.1:4001/typesafe/v1/systemone",
            api_key="",
            model="jev-1.13.0",
            timeout_ms=2000,
            sekrety=[],
        )
        assert result.status == "JEV_NIEDOSTEPNY"
        assert result.powod == "brak klucza"
        assert wywolania == 0

    async def test_jev_redacts_injected_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from surreal_memory.engine import jev_gate

        sekret = "sk-TESTOWY-NIE-JEST-PRAWDZIWY-0123456789"
        captured: dict[str, Any] = {}

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            captured["body_text"] = body.decode("utf-8")
            return 200, _jev_ok_body()

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        result = await jev_gate.zapytaj_jev(
            query="q",
            memories=f"leaked secret is {sekret} inside memories",
            gateway_url="http://127.0.0.1:4001/typesafe/v1/systemone",
            api_key="test-key",
            model="jev-1.13.0",
            timeout_ms=2000,
            sekrety=[sekret],
        )
        assert sekret not in captured["body_text"]
        assert result.zredagowano == 1

    def test_sha256_pytan_is_stable(self) -> None:
        from surreal_memory.engine.jev_pytania import PYTANIA, sha256_pytan

        a = sha256_pytan()
        b = sha256_pytan()
        assert a == b
        assert len(a) == 64

        original = PYTANIA["odpowiada"]["instructions"]
        try:
            PYTANIA["odpowiada"]["instructions"] = original + " (zmienione dla testu)"
            assert sha256_pytan() != a
        finally:
            PYTANIA["odpowiada"]["instructions"] = original


class TestJevPipelineWiring:
    """End-to-end over `obs_storage`, the same harness as
    `TestObserveNeverReportsUnmeasuredAsNegative`/`TestLeksykaObserveSignal`
    above -- drives the REAL `ReflexPipeline`, replacing only
    `jev_gate._blocking_post` (never a real connection)."""

    async def test_jev_off_makes_no_call(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine import jev_gate
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_jev_enabled(monkeypatch, jev_mode="off")
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )
        wywolania = 0

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            nonlocal wywolania
            wywolania += 1
            return 200, _jev_ok_body()

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        assert wywolania == 0
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert not any(k.startswith("jev_") for k in sygnaly)

    async def test_jev_would_refuse_is_none_when_status_not_ok(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine import jev_gate
        from surreal_memory.engine.retrieval import ReflexPipeline

        _obs_jev_enabled(monkeypatch, jev_mode="observe")
        monkeypatch.setenv("SMEM_TEST_JEV_KEY", "test-key-value")
        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )

        def _fake_post(
            url: str, body: bytes, headers: dict[str, str], timeout_s: float
        ) -> tuple[int, bytes]:
            return 500, b"internal error"

        monkeypatch.setattr(jev_gate, "_blocking_post", _fake_post)
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["jev_status"] == "JEV_ODRZUCIL", sygnaly
        assert sygnaly["jev_would_refuse"] is None
        assert sygnaly["jev_odpowiada"] is None
        # recall must still answer normally -- observation never refuses.
        assert result.synthesis_method != "insufficient_signal"


class TestJevUnmeasuredIsNamed:
    """Runner review of U3: "Jev was not measured" must carry the same key set
    plus a NAMED `jev_powod` — never the absence of `jev_*` keys (which means
    observation is OFF) and never a quiet `None` without a reason."""

    async def test_early_exit_at_48_names_jev_as_never_called(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        from surreal_memory.core.fiber import Fiber
        from surreal_memory.core.neuron import Neuron, NeuronType
        from surreal_memory.engine.retrieval import ReflexPipeline
        from surreal_memory.storage.memory_store import InMemoryStorage
        from surreal_memory.unified_config import JevConfig, get_config

        patched = dataclasses.replace(get_config(), jev=JevConfig(mode="observe"))
        monkeypatch.setattr(
            "surreal_memory.unified_config.get_config", lambda reload=False: patched
        )
        s = InMemoryStorage()
        brain = Brain.create(name="jev_early_exit")
        await s.save_brain(brain)
        s.set_brain(brain.id)
        n1 = Neuron.create(type=NeuronType.CONCEPT, content="Emma lives in Oslo Norway")
        await s.add_neuron(n1)
        await s.add_fiber(
            Fiber.create(
                neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id, summary=n1.content
            )
        )
        pipeline = ReflexPipeline(s, _obs_config(refusal_mode="observe"))
        # a query with no anchors at all -> gate 1 `no_anchors` short-circuits at 4.8
        result = await pipeline.query("qwzlmnprt vxbdfghj")
        assert result.synthesis_method == "insufficient_signal"
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["jev_status"] == "JEV_NIEDOSTEPNY"
        assert sygnaly["jev_would_refuse"] is None
        assert "4.8" in sygnaly["jev_powod"]
        await s.close()

    async def test_measured_jev_carries_powod_none_and_failure_carries_reason(
        self, obs_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dataclasses

        from surreal_memory.engine import jev_gate
        from surreal_memory.engine.retrieval import ReflexPipeline
        from surreal_memory.unified_config import JevConfig, RerankerConfig, get_config

        monkeypatch.setattr(
            "surreal_memory.engine.reranker.rerank_activations",
            _obs_fake_rerank(raw_top1=0.9, degraded_reason=None),
        )
        # ONE patched config carrying BOTH seams: the reranker (so 4.9 runs and
        # candidate contents exist) and jev observe. Building the jev patch from
        # the ORIGINAL config would silently drop the reranker and turn this into
        # a "no candidate contents" case — measured while writing this test.
        patched = dataclasses.replace(
            get_config(),
            reranker=RerankerConfig(enabled=True, endpoint="http://fake-reranker.invalid/v1"),
            jev=JevConfig(mode="observe", api_key_file=""),
        )
        monkeypatch.setattr(
            "surreal_memory.unified_config.get_config", lambda reload=False: patched
        )
        monkeypatch.setenv("LITELLM_KEY_ROJ_JEV", "klucz-testowy-nieprawdziwy")
        monkeypatch.setattr(jev_gate, "_blocking_post", lambda *a, **k: (403, b'{"error":"scope"}'))
        pipeline = ReflexPipeline(obs_storage, _obs_config(refusal_mode="observe"))
        result = await pipeline.query(_OBS_QUERY)
        sygnaly = result.metadata["odmowa_sygnaly"]
        assert sygnaly["jev_status"] == "JEV_ODRZUCIL", sygnaly
        assert "403" in sygnaly["jev_powod"]
        assert sygnaly["jev_would_refuse"] is None
