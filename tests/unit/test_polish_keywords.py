"""Tests for Polish stop-word filtering and clause-boundary-aware bigram formation.

Repro corpus reuses the Phase 1 live-derived sentences so these tests encode the
actual observed poisoning failure (cross-clause junk bigrams + bare PL function
words surviving as concepts), not synthetic hopes.
"""

from __future__ import annotations

from surreal_memory.extraction.keywords import (
    STOP_WORDS_EN,
    STOP_WORDS_PL,
    _get_stop_words,
    extract_weighted_keywords,
)

PHASE1_REPRO_SENTENCE = (
    "Sprawdzilem przez reguly czy twojej decyzji dotyczy ten branch, "
    "i potwierdzone zostaly dwie zmiany w sesji."
)


def _unigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " not in r.text}


def _bigrams(text: str, language: str = "auto") -> set[str]:
    return {r.text for r in extract_weighted_keywords(text, language=language) if " " in r.text}


class TestPolishStopWordFiltering:
    """Bare Polish function words must never survive as unigram keywords."""

    def test_polish_function_words_filtered(self) -> None:
        unigrams = _unigrams(PHASE1_REPRO_SENTENCE)
        for word in ("przez", "czy", "ten", "zostaly", "twojej", "dotyczy"):
            assert word not in unigrams, f"'{word}' survived filtering as a unigram"

    def test_polish_stopwords_diacritic_and_ascii_forms(self) -> None:
        pairs = [("się", "sie"), ("że", "ze"), ("już", "juz"), ("która", "ktora")]
        for diacritic, ascii_form in pairs:
            for form in (diacritic, ascii_form):
                text = f"Projekt {form} rozwija dynamicznie kazdego dnia"
                unigrams = _unigrams(text)
                assert form not in unigrams, f"'{form}' survived filtering in: {text!r}"


class TestClauseBoundaryBigrams:
    """Bigrams must never bridge a clause/sentence boundary."""

    def test_no_bigram_across_sentence_boundary(self) -> None:
        bigrams = _bigrams("Redis failed. Query timeout occurred.")
        assert "failed query" not in bigrams
        assert "redis failed" in bigrams
        assert "query timeout" in bigrams

    def test_no_bigram_across_comma(self) -> None:
        bigrams = _bigrams("ten branch, i potwierdzone zostaly")
        assert "branch potwierdzone" not in bigrams

    def test_no_bigram_across_em_dash(self) -> None:
        # U3 fresh-process proof (2026-07-06) live-reproduced this: a filename
        # fragment ("py" from "keywords.py") glued onto the next clause's first
        # word across an em-dash, since em-dash wasn't originally a clause boundary.
        bigrams = _bigrams("blad w keywords.py — bigramy nie powinny przecinac granic")
        assert "py bigramy" not in bigrams
        # Sanity: the words still survive as unigrams / pair within their own clause.
        assert "bigramy nie" not in bigrams  # "nie" is a stopword, excluded anyway
        unigrams = _unigrams("blad w keywords.py — bigramy nie powinny przecinac granic")
        assert "py" in unigrams
        assert "bigramy" in unigrams

    def test_no_bigram_across_en_dash(self) -> None:
        # The en-dash (U+2013) is a clause boundary too, not only the em-dash
        # (—, U+2014): editors and OSes emit it for ranges and asides, so without
        # it the same cross-clause junk bigram (a filename fragment glued onto the
        # next clause) slips through.
        text = "blad w keywords.py – bigramy nie powinny przecinac granic"  # noqa: RUF001
        bigrams = _bigrams(text)
        assert "py bigramy" not in bigrams
        unigrams = _unigrams(text)
        assert "py" in unigrams
        assert "bigramy" in unigrams

    def test_hyphen_is_not_a_clause_boundary(self) -> None:
        # A hyphen inside a compound term must NOT split a clause: the two halves
        # are adjacent content words and should still pair as a bigram (contrast
        # with the en-/em-dash, which do split).
        bigrams = _bigrams("modul cache-aside dziala szybko")
        assert "cache aside" in bigrams

    def test_phase1_junk_bigrams_eliminated(self) -> None:
        junk = {
            "reguly czy",
            "czy twojej",
            "decyzji dotyczy",
            "dotyczy ten",
            "branch potwierdzone",
            "zostaly dwie",
            "sprawdzilem przez",
            "przez reguly",
        }
        bigrams = _bigrams(PHASE1_REPRO_SENTENCE)
        assert junk & bigrams == set()

    def test_coherent_bigram_within_clause_survives(self) -> None:
        bigrams = _bigrams("Spreading activation w database migration zakonczona sukcesem.")
        assert "database migration" in bigrams


class TestStopwordBridgedBigrams:
    """Stopword-bridged pairs (gap<=2, same clause) must still form — that's the point
    of gap<=2 rather than strict adjacency."""

    def test_stopword_bridged_bigram_within_gap(self) -> None:
        assert "query database" in _bigrams("query the database")
        assert "sprawdzilem reguly" in _bigrams("Sprawdzilem przez reguly")

    def test_bigram_gap_three_no_longer_pairs(self) -> None:
        # "redis" / "is" / "the" / "backbone" -> content words at original gap 3
        assert "redis backbone" not in _bigrams("redis is the backbone")


class TestPolishLanguageHintBranch:
    def test_polish_language_hint_branch(self) -> None:
        pl_stop_words = _get_stop_words("pl", "")
        assert pl_stop_words >= STOP_WORDS_PL
        assert pl_stop_words >= STOP_WORDS_EN


class TestAsciiCollisionExclusions:
    """ASCII short-forms colliding with domain acronyms must not be filtered."""

    def test_ci_survives_as_keyword_in_auto_mode(self) -> None:
        # "ci" collides with "CI" (continuous integration), ubiquitous in this
        # project's dev content — the same ASCII-collision class this PR removes
        # for Vietnamese ("ai"/"em"). Guard against reintroducing it.
        assert "ci" not in STOP_WORDS_PL
        assert "ci" in _unigrams("Fixed CI lint failure, CI now passes cleanly")
