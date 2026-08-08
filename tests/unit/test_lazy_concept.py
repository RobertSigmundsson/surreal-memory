"""Lazy concept promotion: a keyword earns a neuron only once it recurs.

Concepts used to become permanent on their FIRST appearance while entities had to be
mentioned twice (``ExtractEntityNeuronsStep``). Since the keyword extractor emits mostly
adjacent-word bi-grams, that asymmetry is what silts a brain up: replaying 1199 real
memories from a production brain showed 82 % of concept creations were for a keyword that
never appeared again -- debris like "normy przedmiarowej" or "architektura silnika".

Recurrence is read from ``keyword_document_frequency``, which the encoder already
maintains, and which is incremented LATER in the pipeline than concept extraction -- so
the count seen here excludes the memory being encoded. These tests pin that ordering,
because if the increment ever moved earlier every keyword would look like a repeat and
the promotion gate would quietly stop gating.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import NeuronType
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.storage.memory_store import InMemoryStorage

pytestmark = pytest.mark.asyncio

# The recurring terms lead in BOTH texts on purpose. concept_limit is
# min(20, max(3, len//100)), so a short memory only ever offers its top 3 keywords, and
# keyword weight decays with position -- put the repeats at the end and they never reach
# the promotion check at all, which would make these tests measure ranking, not promotion.
_TEXT = "Przedmiar robot budowlanych oraz jednostki miary pozycji wedlug rozporzadzenia."
_TEXT_AGAIN = "Przedmiar robot budowlanych oraz jednostki miary pozycji w drugim zrodle."
_TEXT_THIRD = "Przedmiar robot budowlanych oraz jednostki miary pozycji w trzeciej notatce."


async def _encoder(**cfg: object) -> tuple[MemoryEncoder, InMemoryStorage]:
    storage = InMemoryStorage()
    brain = Brain.create(name="lazy-concept", config=replace(BrainConfig(), **cfg))  # type: ignore[arg-type]
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return MemoryEncoder(storage, brain.config), storage


def _keyword_concepts(result: object) -> set[str]:
    """Only the concepts ExtractConceptNeuronsStep made.

    Three different steps emit ``NeuronType.CONCEPT`` and only one of them is under test:
    the anchor (the memory text itself, tagged ``is_anchor``), promoted named entities
    (tagged ``entity_type`` -- ``_entity_type_to_neuron_type`` maps several entity kinds
    onto CONCEPT), and keyword extraction (no metadata at all). Counting all three is how
    a test ends up asserting about a neuron the lazy gate never had a say over.
    """
    out = set()
    for n in result.neurons_created:  # type: ignore[attr-defined]
        if n.type != NeuronType.CONCEPT:
            continue
        meta = n.metadata or {}
        if "is_anchor" in meta or "entity_type" in meta:
            continue
        out.add(n.content.lower())
    return out


def _anchor_present(result: object, text: str) -> bool:
    return any(
        n.content.lower() == text.lower() and (n.metadata or {}).get("is_anchor")
        for n in result.neurons_created  # type: ignore[attr-defined]
    )


class TestLazyConceptPromotion:
    async def test_first_sighting_creates_no_keyword_concept(self) -> None:
        enc, _ = await _encoder()
        result = await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))

        # The anchor neuron (the memory text itself) is also a CONCEPT and is created by
        # a different step, so it must survive -- the memory is never withheld.
        assert _anchor_present(result, _TEXT), "the memory's own anchor must still be stored"
        assert _keyword_concepts(result) == set(), (
            f"first sighting must defer every keyword, got {_keyword_concepts(result)}"
        )

    async def test_second_sighting_promotes(self) -> None:
        """Positive control: without this, the assertion above would also pass if
        concept extraction were broken outright."""
        enc, _ = await _encoder()
        await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))
        result = await enc.encode(_TEXT_AGAIN, timestamp=datetime(2024, 2, 5, 15, 0))

        promoted = _keyword_concepts(result)
        assert promoted, "a keyword seen in a previous memory must be promoted on the next one"
        assert any("przedmiar" in c or "jednostki" in c or "miary" in c for c in promoted), (
            f"expected the recurring terms among the promoted concepts, got {promoted}"
        )

    async def test_deferred_keywords_are_recorded_not_silently_dropped(self) -> None:
        """'Nothing created' must stay distinguishable from 'nothing found'."""
        from surreal_memory.engine.pipeline import PipelineContext
        from surreal_memory.engine.pipeline_steps import ExtractConceptNeuronsStep

        storage = InMemoryStorage()
        brain = Brain.create(name="lazy-concept-ctx", config=BrainConfig())
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        ctx = PipelineContext(
            content=_TEXT,
            timestamp=datetime(2024, 2, 4, 15, 0),
            metadata={},
            tags=set(),
            language="pl",
        )
        ctx = await ExtractConceptNeuronsStep().execute(ctx, storage, brain.config)

        assert ctx.concept_neurons == []
        assert ctx.deferred_concepts, "deferred keywords must be recorded for observability"

    async def test_disabled_restores_eager_creation(self) -> None:
        enc, _ = await _encoder(lazy_concept_enabled=False)
        result = await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))

        assert _keyword_concepts(result), (
            "with lazy promotion off, the first sighting must create concepts as before"
        )

    async def test_higher_threshold_needs_more_sightings(self) -> None:
        enc, _ = await _encoder(lazy_concept_promotion_threshold=3)
        await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))
        second = await enc.encode(_TEXT_AGAIN, timestamp=datetime(2024, 2, 5, 15, 0))
        assert _keyword_concepts(second) == set(), (
            "at threshold 3 a second sighting is still not enough"
        )

        third = await enc.encode(_TEXT_THIRD, timestamp=datetime(2024, 2, 6, 15, 0))
        assert _keyword_concepts(third), "the third sighting must promote"


class TestLazyConceptRobustness:
    """The gate must never be the reason encoding dies, and never silently invert."""

    async def test_broken_df_counter_fails_open(self) -> None:
        """A storage whose DF lookup raises must fall back to creating concepts.
        A mute brain is worse than a noisy one, and the entity path behaves the same."""
        enc, storage = await _encoder()

        async def _boom(_keywords: list[str]) -> dict[str, int]:
            raise RuntimeError("DF table unavailable")

        storage.get_keyword_df_batch = _boom  # type: ignore[method-assign]
        result = await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))

        assert _keyword_concepts(result), (
            "a broken DF counter must not silently stop concept extraction"
        )

    async def test_non_dict_df_result_does_not_crash(self) -> None:
        """Storage doubles hand back all sorts of things; the shape guard belongs at
        the lookup, not in the comparison three lines down."""
        enc, storage = await _encoder()

        async def _garbage(_keywords: list[str]) -> dict[str, int]:
            return None  # type: ignore[return-value]

        storage.get_keyword_df_batch = _garbage  # type: ignore[method-assign]
        result = await enc.encode(_TEXT, timestamp=datetime(2024, 2, 4, 15, 0))
        assert _anchor_present(result, _TEXT)
