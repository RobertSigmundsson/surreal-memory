"""Tests for LLM dedup judge."""

from __future__ import annotations

from surreal_memory.engine.dedup.llm_judge import (
    DedupVerdict,
    _parse_verdict,
    create_judge,
)


class TestParseVerdict:
    def test_duplicate_response(self) -> None:
        response = "DUPLICATE\nThese convey the same information about database choice."
        result = _parse_verdict(response)
        assert result.verdict == DedupVerdict.DUPLICATE
        assert "same information" in result.reason

    def test_distinct_response(self) -> None:
        response = "DISTINCT\nDifferent topics despite similar keywords."
        result = _parse_verdict(response)
        assert result.verdict == DedupVerdict.DISTINCT

    def test_uncertain_response(self) -> None:
        response = "UNCERTAIN\nNot enough context to determine."
        result = _parse_verdict(response)
        assert result.verdict == DedupVerdict.UNCERTAIN

    def test_partial_match_duplicate(self) -> None:
        response = "These are DUPLICATE entries."
        result = _parse_verdict(response)
        assert result.verdict == DedupVerdict.DUPLICATE

    def test_unknown_response_is_uncertain(self) -> None:
        response = "I'm not sure about this."
        result = _parse_verdict(response)
        assert result.verdict == DedupVerdict.UNCERTAIN

    def test_empty_response(self) -> None:
        result = _parse_verdict("")
        assert result.verdict == DedupVerdict.UNCERTAIN

    def test_no_reason_line(self) -> None:
        result = _parse_verdict("DUPLICATE")
        assert result.verdict == DedupVerdict.DUPLICATE
        assert result.reason == ""


class TestCreateJudge:
    def test_openai_provider(self) -> None:
        judge = create_judge("openai", "gpt-4o-mini")
        assert judge is not None

    def test_anthropic_provider(self) -> None:
        judge = create_judge("anthropic", "claude-sonnet-4-5-20250929")
        assert judge is not None

    def test_none_provider(self) -> None:
        judge = create_judge("none", "")
        assert judge is None

    def test_unknown_provider(self) -> None:
        judge = create_judge("unknown", "model")
        assert judge is None
