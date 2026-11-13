"""Tests for quality scorer engine."""

from __future__ import annotations

import pytest

from surreal_memory.engine.quality_scorer import (
    QualityResult,
    _is_machine_noise,
    check_write_gate,
    score_memory,
)
from surreal_memory.mcp.auto_capture import build_capture_context
from surreal_memory.unified_config import WriteGateConfig


class TestIsMachineNoise:
    """Pins the machine-noise denylist added in #94.

    A denylist rots silently: widening ``_NOISE_TAG_RE`` can start swallowing real
    memories with nothing failing, and narrowing it lets the junk back in. Both
    directions are pinned, per signal.
    """

    @pytest.mark.parametrize(
        "content",
        [
            "Session activity: Zrobione. Zweryfikowane...",  # 180 real false-accepts
            "session activity: lowercased variant",  # prefix match is case-insensitive
            "<task-notification>build finished</task-notification>",
            "ps -o pid,cmd -p 1234",
            "echo hello",
            "curl -s http://127.0.0.1:8001/health",
        ],
    )
    def test_rejects_machine_output(self, content: str) -> None:
        assert _is_machine_noise(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "<summary>tool output</summary>",
            "<status>running</status>",
            "<system-reminder>ambient context</system-reminder>",
            'Background command "pytest" completed',
        ],
    )
    def test_rejects_structural_tags(self, content: str) -> None:
        assert _is_machine_noise(content) is True

    @pytest.mark.parametrize(
        "content",
        [
            "Chose SurrealDB over Postgres because the graph traversal is native.",
            "Root cause: the reranker was enabled but unreachable, so recall degraded.",
            # Mentions a shell command mid-sentence — the prefix rule must not fire.
            "We debugged it by running curl against the health endpoint.",
            # Contains the word 'summary' but no structural tag.
            "The summary of the incident is in the postmortem doc.",
            "Toni prefers direct answers with no filler.",
        ],
    )
    def test_keeps_real_knowledge(self, content: str) -> None:
        assert _is_machine_noise(content) is False


class TestScoreMemory:
    """Tests for score_memory()."""

    def test_minimal_content_low_quality(self) -> None:
        """Very short content with no metadata scores low."""
        result = score_memory("hello")
        assert result.quality == "low"
        assert result.score < 4
        assert len(result.hints) > 0

    def test_short_content_has_length_hint(self) -> None:
        """Content under 10 chars gets a length hint."""
        result = score_memory("hi")
        assert any("short" in h.lower() for h in result.hints)

    def test_long_content_gets_length_points(self) -> None:
        """Content >= 50 chars gets 2 length points."""
        content = "a" * 60
        result = score_memory(content)
        # At least 2 points from length
        assert result.score >= 2

    def test_context_dict_adds_3_points(self) -> None:
        """Providing context dict adds 3 points."""
        base = score_memory("some content here")
        with_ctx = score_memory(
            "some content here",
            context={"reason": "testing"},
        )
        assert with_ctx.score >= base.score + 3

    def test_empty_context_no_bonus(self) -> None:
        """Empty context dict does not add points."""
        base = score_memory("some content here")
        with_empty = score_memory("some content here", context={})
        assert with_empty.score == base.score

    def test_tags_add_1_point(self) -> None:
        """Providing at least 1 tag adds 1 point."""
        base = score_memory("some content here")
        with_tags = score_memory("some content here", tags=["test"])
        assert with_tags.score == base.score + 1

    def test_empty_tags_no_bonus(self) -> None:
        """Empty tag list does not add points."""
        base = score_memory("some content here")
        with_empty = score_memory("some content here", tags=[])
        assert with_empty.score == base.score

    def test_non_fact_type_adds_1_point(self) -> None:
        """Non-default type adds 1 point."""
        base = score_memory("some content here", memory_type="fact")
        decision = score_memory("some content here", memory_type="decision")
        assert decision.score == base.score + 1

    def test_no_type_gives_hint(self) -> None:
        """No type provided gives a hint."""
        result = score_memory("some content here")
        assert any("type" in h.lower() for h in result.hints)

    def test_causal_word_adds_point(self) -> None:
        """Causal language adds 1 cognitive richness point."""
        without = score_memory("chose PostgreSQL for payments")
        with_causal = score_memory("chose PostgreSQL because ACID needed for payments")
        assert with_causal.score > without.score

    def test_temporal_word_adds_point(self) -> None:
        """Temporal language adds 1 cognitive richness point."""
        without = score_memory("upgraded the database schema")
        with_temporal = score_memory("after upgrading the database schema, tests broke")
        assert with_temporal.score > without.score

    def test_comparative_word_adds_point(self) -> None:
        """Comparative language adds 1 cognitive richness point."""
        without = score_memory("selected PostgreSQL for the project")
        with_comp = score_memory("selected PostgreSQL over MongoDB for the project")
        assert with_comp.score > without.score

    def test_no_cognitive_words_gives_hint(self) -> None:
        """No causal/temporal/comparative words gives a reasoning hint."""
        result = score_memory("PostgreSQL database")
        assert any("reasoning" in h.lower() or "why" in h.lower() for h in result.hints)

    def test_high_quality_no_hints(self) -> None:
        """High quality memories get no hints."""
        result = score_memory(
            "Chose PostgreSQL over MongoDB because ACID needed for payment processing",
            memory_type="decision",
            tags=["database", "payments"],
            context={"reason": "ACID compliance", "alternatives": ["MongoDB"]},
        )
        assert result.quality == "high"
        assert result.score >= 7
        assert len(result.hints) == 0

    def test_medium_quality(self) -> None:
        """Medium quality has score 4-6."""
        result = score_memory(
            "PostgreSQL is our database because it supports ACID",
            memory_type="fact",
            tags=["db"],
        )
        assert result.quality == "medium"
        assert 4 <= result.score <= 6

    def test_score_capped_at_10(self) -> None:
        """Score never exceeds 10."""
        result = score_memory(
            "After testing, chose PostgreSQL over MongoDB because ACID "
            "needed and it was faster than alternatives for our use case",
            memory_type="decision",
            tags=["database", "payments", "architecture"],
            context={
                "reason": "ACID for payments",
                "alternatives": ["MongoDB", "CockroachDB"],
                "impact": "critical",
            },
        )
        assert result.score <= 10

    def test_immutability(self) -> None:
        """score_memory does not mutate inputs."""
        tags = ["original"]
        context = {"reason": "test"}
        tags_copy = list(tags)
        context_copy = dict(context)
        score_memory("test content", tags=tags, context=context)
        assert tags == tags_copy
        assert context == context_copy


class TestProvenanceContextEarnsNoPoints:
    """W-4: kontekst z samej proweniencji haka (``build_capture_context``) nie daje +3.

    Regresja: hak zawsze podaje ``captured_by``/``detected_type``/``confidence``, więc +3 za „kontekst niepusty”
    było stałym prezentem — 345/345 wpisów z haków przechodziło bramkę (PROJEKT Q8).
    """

    CONTENT = "some content here"

    def test_provenance_keys_are_exactly_what_the_hook_context_builds(self) -> None:
        from surreal_memory.engine.quality_scorer import _PROVENANCE_KEYS

        ctx = build_capture_context({"type": "decision", "confidence": 0.8}, source="stop_hook")
        assert set(ctx) == set(_PROVENANCE_KEYS)

    def test_provenance_only_context_adds_nothing(self) -> None:
        base = score_memory(self.CONTENT)
        ctx = build_capture_context({"type": "decision", "confidence": 0.8}, source="stop_hook")
        assert score_memory(self.CONTENT, context=ctx).score == base.score

    def test_each_single_provenance_key_adds_nothing(self) -> None:
        from surreal_memory.engine.quality_scorer import _PROVENANCE_KEYS

        base = score_memory(self.CONTENT)
        for key in _PROVENANCE_KEYS:
            assert score_memory(self.CONTENT, context={key: "x"}).score == base.score, key

    def test_reason_key_adds_3(self) -> None:
        base = score_memory(self.CONTENT)
        assert score_memory(self.CONTENT, context={"reason": "bo tak"}).score == base.score + 3

    def test_provenance_plus_reason_adds_3_once(self) -> None:
        base = score_memory(self.CONTENT)
        ctx = {
            **build_capture_context({"type": "decision", "confidence": 0.8}, source="stop_hook"),
            "reason": "bo tak",
        }
        assert score_memory(self.CONTENT, context=ctx).score == base.score + 3

    def test_auto_path_ceiling_reaches_default_threshold(self) -> None:
        """Sufit ścieżki auto haka policzony FUNKCJĄ (nie stałą) ≥ ``auto_capture_min_score``.

        Pułapka „próg nieosiągalny” (wiki ``smem-write-gate-auto-capture``): maksimum fragmentu z haka — kontekst
        z SAMEJ proweniencji (+0), tagi 1, typ ≠ fact 1, długość ≥ 50 ⇒ 2, trzy markery kognitywne 3 — musi być
        ≥ progu ścieżki auto z domyślnej konfiguracji (PROJEKT §2.4: 2 + 1 + 1 + 3 = 7).
        """
        content = "After testing, chose PostgreSQL over MongoDB because ACID is needed for payments"
        ceiling = score_memory(
            content,
            memory_type="decision",
            tags=["stop_hook", "session_end"],
            context=build_capture_context({"type": "decision"}, source="stop_hook"),
        ).score
        assert ceiling == 7
        assert ceiling >= WriteGateConfig().auto_capture_min_score

    def test_gate_without_provenance_points_for_short_four_point_fragment(self) -> None:
        """Fragment 4-punktowy (długość 2 + tagi 1 + typ 1) przechodzi próg 4 bez kontekstu i z kontekstem proweniencji."""
        cfg = WriteGateConfig(auto_capture_min_score=4)
        content = "x" * 60  # długość ≥ 50 ⇒ 2 pkt; bez markerów kognitywnych
        ctx = build_capture_context({"type": "preference"}, source="stop_hook")
        for context in (None, ctx):
            res = check_write_gate(
                content,
                gate_config=cfg,
                is_auto_capture=True,
                memory_type="preference",
                tags=["stop_hook"],
                context=context,
            )
            assert res.score == 4 and not res.rejected

    def test_default_threshold_is_five_and_below_the_ceiling(self) -> None:
        """W-4b: domyślny próg ścieżki auto == 5, a sufit (7) go osiąga — próg nie jest nieosiągalny."""
        assert WriteGateConfig().auto_capture_min_score == 5

    def test_provenance_only_context_scores_four_and_is_rejected_by_default_gate(self) -> None:
        """Kontekst WYŁĄCZNIE z proweniencji = 4 pkt (długość 2 + tagi 1 + typ 1) < 5 ⇒ odrzucony domyślną bramką.

        Dodanie klucza spoza proweniencji (+3) daje 7 ≥ 5 ⇒ przyjęty (kontrola pozytywna: próg nie odrzuca wszystkiego).
        """
        cfg = WriteGateConfig(enabled=True, mode="enforce", auto_capture_mode="enforce")
        content = "x" * 60  # długość ≥ 50 ⇒ 2 pkt; bez markerów kognitywnych
        ctx = build_capture_context({"type": "preference"}, source="stop_hook")
        res = check_write_gate(
            content,
            gate_config=cfg,
            is_auto_capture=True,
            memory_type="preference",
            tags=["stop_hook"],
            context=ctx,
        )
        assert res.score == 4 and res.rejected
        res_reason = check_write_gate(
            content,
            gate_config=cfg,
            is_auto_capture=True,
            memory_type="preference",
            tags=["stop_hook"],
            context={**ctx, "reason": "bo tak"},
        )
        assert res_reason.score == 7 and not res_reason.rejected


class TestQualityResult:
    """Tests for QualityResult dataclass."""

    def test_to_dict_with_hints(self) -> None:
        result = QualityResult(score=3, quality="low", hints=("Add tags",))
        d = result.to_dict()
        assert d == {"quality": "low", "score": 3, "hints": ["Add tags"]}

    def test_to_dict_no_hints(self) -> None:
        result = QualityResult(score=8, quality="high", hints=())
        d = result.to_dict()
        assert d == {"quality": "high", "score": 8}
        assert "hints" not in d

    def test_frozen(self) -> None:
        result = QualityResult(score=5, quality="medium")
        with pytest.raises(AttributeError):
            result.score = 10  # type: ignore[misc]
