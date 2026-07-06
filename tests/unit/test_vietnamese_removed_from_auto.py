"""Tests for the STOP_WORDS_VI-removed-from-"auto" fix.

STOP_WORDS_VI has 9 ASCII-only entries (no Vietnamese diacritics): ai, anh, bao,
cho, em, khi, ra, sao, trong. Being ASCII, each collides with the character set
of ordinary English/Polish text and was silently deleted from every "auto"-mode
extraction. Empirically, only "ai" (-> "AI") and "em" (-> "an em dash") are
confirmed real collisions in this project's actual vocabulary (live-reproduced
during the concept-fragment-fix run this same day). The other 7 were checked
here for plausibility (dictionary lookup + grep across this repo's own
EN/PL-language surfaces) and found to have no evidence of real EN/PL usage in
this project — every occurrence found is inside a Vietnamese-specific code path
(parser.py/relations.py/entities.py/auto_capture.py), confirming they are
genuine VI function words with no EN/PL homonym here, not a collision.
"""

from __future__ import annotations

import pytest

from surreal_memory.extraction.keywords import (
    STOP_WORDS_VI,
    _get_stop_words,
    extract_weighted_keywords,
)


def _unigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " not in r.text}


def _bigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " in r.text}


ASCII_ONLY_VI_ENTRIES = frozenset({"ai", "anh", "bao", "cho", "em", "khi", "ra", "sao", "trong"})


class TestAsciiOnlyViEntriesIdentified:
    def test_exactly_nine_ascii_only_entries(self) -> None:
        ascii_entries = {w for w in STOP_WORDS_VI if w.isascii()}
        assert ascii_entries == ASCII_ONLY_VI_ENTRIES


class TestConfirmedCollisions:
    """Real, live-reproduced collisions: EN/PL content silently losing words."""

    def test_ai_survives_as_unigram_in_auto_mode(self) -> None:
        text = "Zbudowalismy system agentow AI ktory uczy sie sam"
        unigrams = _unigrams(text)
        assert "ai" in unigrams

    def test_ai_forms_bigrams_in_auto_mode(self) -> None:
        text = "Zbudowalismy system agentow AI ktory uczy sie sam"
        bigrams = _bigrams(text)
        assert "agentow ai" in bigrams
        assert "ai uczy" in bigrams

    def test_em_survives_as_unigram_in_auto_mode(self) -> None:
        text = "This is an em dash in a sentence"
        unigrams = _unigrams(text)
        assert "em" in unigrams

    def test_em_dash_bigram_forms_in_auto_mode(self) -> None:
        text = "This is an em dash in a sentence"
        bigrams = _bigrams(text)
        assert "em dash" in bigrams


class TestOtherAsciiEntriesSurvive:
    """Mechanical survival check for the remaining 7 entries: no plausible
    real-world EN/PL collision was found (see module docstring), but each
    still must no longer be silently filtered if it DOES occur in auto-mode
    text — this is the fix's actual, testable guarantee, independent of how
    likely real-world occurrence is."""

    @pytest.mark.parametrize(
        "word,sentence",
        [
            ("anh", "The report was reviewed by anh from the team"),
            ("bao", "We ordered a pork bao for lunch today"),
            ("cho", "The engineer named cho signed off on the release"),
            ("khi", "The variable khi tracks the rotation angle"),
            ("ra", "The RA team approved the compliance report"),
            ("sao", "The satellite sao completed its orbit"),
            ("trong", "The word trong only exists in Vietnamese"),
        ],
    )
    def test_word_survives_as_unigram_in_auto_mode(self, word: str, sentence: str) -> None:
        unigrams = _unigrams(sentence)
        assert word in unigrams, f"{word!r} was still filtered in auto mode: {sentence!r}"


class TestExplicitViBranchUnchanged:
    def test_get_stop_words_vi_still_returns_full_stop_words_vi(self) -> None:
        assert _get_stop_words("vi", "") == STOP_WORDS_VI

    def test_vi_explicit_language_still_filters_ai_and_em(self) -> None:
        # The explicit "vi" branch is out of scope for this fix — "ai"/"em" must
        # still be filtered when language="vi" is passed explicitly.
        unigrams_ai = _unigrams("AI la mot he thong", language="vi")
        assert "ai" not in unigrams_ai
        unigrams_em = _unigrams("em la mot nguoi ban", language="vi")
        assert "em" not in unigrams_em
