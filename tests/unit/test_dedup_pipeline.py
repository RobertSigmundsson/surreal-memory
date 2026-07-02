"""Tests for dedup pipeline (3-tier cascade)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.dedup.config import DedupConfig
from surreal_memory.engine.dedup.llm_judge import DedupJudgment, DedupVerdict
from surreal_memory.engine.dedup.pipeline import DedupPipeline
from surreal_memory.utils.simhash import simhash


def _make_anchor(content: str, neuron_id: str = "anchor-1") -> Neuron:
    return Neuron(
        id=neuron_id,
        type=NeuronType.CONCEPT,
        content=content,
        metadata={"is_anchor": True},
        content_hash=simhash(content),
    )


def _make_storage(candidates: list[Neuron] | None = None) -> AsyncMock:
    storage = AsyncMock()
    storage.find_neurons = AsyncMock(return_value=candidates or [])
    return storage


class TestDedupDisabled:
    @pytest.mark.asyncio
    async def test_disabled_returns_not_duplicate(self) -> None:
        cfg = DedupConfig(enabled=False)
        storage = _make_storage()
        pipeline = DedupPipeline(config=cfg, storage=storage)

        result = await pipeline.check_duplicate("hello world")
        assert result.is_duplicate is False
        assert "disabled" in result.reason


class TestTier1SimHash:
    @pytest.mark.asyncio
    async def test_exact_duplicate_detected(self) -> None:
        content = "We decided to use PostgreSQL for the database"
        anchor = _make_anchor(content)
        cfg = DedupConfig(enabled=True, simhash_threshold=10)
        storage = _make_storage([anchor])

        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate(content)

        assert result.is_duplicate is True
        assert result.existing_neuron_id == "anchor-1"
        assert result.tier == 1
        assert result.similarity_score > 0.8

    @pytest.mark.asyncio
    async def test_near_duplicate_detected(self) -> None:
        original = "We decided to use PostgreSQL for the database"
        similar = "We decided to use PostgreSQL for the databases"
        anchor = _make_anchor(original)
        cfg = DedupConfig(enabled=True, simhash_threshold=10)
        storage = _make_storage([anchor])

        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate(similar)

        assert result.is_duplicate is True
        assert result.tier == 1

    @pytest.mark.asyncio
    async def test_different_content_not_duplicate(self) -> None:
        anchor = _make_anchor("We use PostgreSQL for the database")
        cfg = DedupConfig(enabled=True, simhash_threshold=5)
        storage = _make_storage([anchor])

        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate(
            "The weather is nice today and we should go outside"
        )

        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_no_candidates_returns_not_duplicate(self) -> None:
        cfg = DedupConfig(enabled=True)
        storage = _make_storage([])

        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate("some content")

        assert result.is_duplicate is False
        assert "no candidates" in result.reason

    @pytest.mark.asyncio
    async def test_anchor_without_hash_skipped(self) -> None:
        anchor = Neuron(
            id="anchor-no-hash",
            type=NeuronType.CONCEPT,
            content="some content",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(enabled=True)
        storage = _make_storage([anchor])

        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate("some content")

        assert result.is_duplicate is False

    @pytest.mark.asyncio
    async def test_exact_duplicate_outside_word_search_window(self) -> None:
        """Exact dup must be caught even when the first-significant-word search
        misses it (common leading token -> true dup outside the LIMIT window)."""
        content = "Error: agent registry creates a new record per init instead of upsert"
        true_dup = _make_anchor(content, neuron_id="anchor-true-dup")
        unrelated = [
            _make_anchor(f"Error: some unrelated failure number {i}", neuron_id=f"anchor-{i}")
            for i in range(30)
        ]

        async def find_neurons(**kwargs: object) -> list[Neuron]:
            if kwargs.get("content_exact") is not None:
                return [true_dup]
            return unrelated

        storage = AsyncMock()
        storage.find_neurons = AsyncMock(side_effect=find_neurons)

        cfg = DedupConfig(enabled=True)
        pipeline = DedupPipeline(config=cfg, storage=storage)
        result = await pipeline.check_duplicate(content)

        assert result.is_duplicate is True
        assert result.existing_neuron_id == "anchor-true-dup"
        assert result.tier == 1


class TestTier2Embedding:
    @pytest.mark.asyncio
    async def test_high_similarity_is_duplicate(self) -> None:
        anchor = _make_anchor("PostgreSQL database setup", neuron_id="emb-1")
        # Set hash to 0 so tier 1 doesn't match
        anchor = Neuron(
            id="emb-1",
            type=NeuronType.CONCEPT,
            content="PostgreSQL database setup",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(
            enabled=True,
            embedding_threshold=0.85,
            embedding_ambiguous_low=0.75,
        )
        storage = _make_storage([anchor])

        # Mock embedding provider with high similarity
        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        embedding_provider.similarity = AsyncMock(return_value=0.92)

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        result = await pipeline.check_duplicate("PostgreSQL database configuration")

        assert result.is_duplicate is True
        assert result.tier == 2
        assert result.similarity_score >= 0.85

    @pytest.mark.asyncio
    async def test_low_similarity_not_duplicate(self) -> None:
        anchor = Neuron(
            id="emb-2",
            type=NeuronType.CONCEPT,
            content="PostgreSQL database setup",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(
            enabled=True,
            embedding_threshold=0.85,
            embedding_ambiguous_low=0.75,
        )
        storage = _make_storage([anchor])

        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        embedding_provider.similarity = AsyncMock(return_value=0.5)

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        result = await pipeline.check_duplicate("The weather is sunny")

        assert result.is_duplicate is False
        assert result.tier == 2

    @pytest.mark.asyncio
    async def test_borderline_defers_to_tier3(self) -> None:
        """Ambiguous zone (0.75-0.85) should NOT return a result from tier 2."""
        anchor = Neuron(
            id="emb-3",
            type=NeuronType.CONCEPT,
            content="PostgreSQL database setup",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(
            enabled=True,
            embedding_threshold=0.85,
            embedding_ambiguous_low=0.75,
        )
        storage = _make_storage([anchor])

        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        embedding_provider.similarity = AsyncMock(return_value=0.80)

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
        )
        # Without LLM judge, this falls through to "no tier found a match"
        result = await pipeline.check_duplicate("PostgreSQL db setup guide")

        assert result.is_duplicate is False
        assert "no tier found" in result.reason


class TestTier3LLM:
    @pytest.mark.asyncio
    async def test_llm_duplicate_verdict(self) -> None:
        anchor = Neuron(
            id="llm-1",
            type=NeuronType.CONCEPT,
            content="We use PostgreSQL for storage",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(
            enabled=True,
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        storage = _make_storage([anchor])

        # Mock embedding provider to return borderline similarity
        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0, 0.0])
        embedding_provider.similarity = AsyncMock(return_value=0.80)

        # Mock LLM judge
        llm_judge = AsyncMock()
        llm_judge.judge = AsyncMock(
            return_value=DedupJudgment(
                verdict=DedupVerdict.DUPLICATE,
                reason="Same core fact about PostgreSQL",
                confidence=0.85,
            )
        )

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
            llm_judge=llm_judge,
        )
        result = await pipeline.check_duplicate("PostgreSQL is our database")

        assert result.is_duplicate is True
        assert result.tier == 3

    @pytest.mark.asyncio
    async def test_llm_distinct_verdict(self) -> None:
        anchor = Neuron(
            id="llm-2",
            type=NeuronType.CONCEPT,
            content="We use PostgreSQL for storage",
            metadata={"is_anchor": True},
            content_hash=0,
        )
        cfg = DedupConfig(
            enabled=True,
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
        )
        storage = _make_storage([anchor])

        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0, 0.0])
        embedding_provider.similarity = AsyncMock(return_value=0.80)

        llm_judge = AsyncMock()
        llm_judge.judge = AsyncMock(
            return_value=DedupJudgment(
                verdict=DedupVerdict.DISTINCT,
                reason="Different topics despite similar words",
                confidence=0.9,
            )
        )

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
            llm_judge=llm_judge,
        )
        result = await pipeline.check_duplicate("PostgreSQL migration strategy")

        assert result.is_duplicate is False
        # DISTINCT no longer short-circuits Tier 3 — it continues checking
        # remaining candidates. With only 1 candidate, loop ends and falls
        # through to "no tier found a match" (tier=0).
        assert result.tier == 0

    @pytest.mark.asyncio
    async def test_llm_max_pairs_respected(self) -> None:
        """LLM judge should be called at most max_pairs_per_encode times."""
        anchors = [
            Neuron(
                id=f"llm-{i}",
                type=NeuronType.CONCEPT,
                content=f"Content {i}",
                metadata={"is_anchor": True},
                content_hash=0,
            )
            for i in range(5)
        ]
        cfg = DedupConfig(
            enabled=True,
            llm_enabled=True,
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_max_pairs_per_encode=2,
        )
        storage = _make_storage(anchors)

        embedding_provider = AsyncMock()
        embedding_provider.embed = AsyncMock(return_value=[1.0])
        embedding_provider.similarity = AsyncMock(return_value=0.80)

        llm_judge = AsyncMock()
        llm_judge.judge = AsyncMock(
            return_value=DedupJudgment(
                verdict=DedupVerdict.UNCERTAIN,
                reason="Cannot determine",
                confidence=0.3,
            )
        )

        pipeline = DedupPipeline(
            config=cfg,
            storage=storage,
            embedding_provider=embedding_provider,
            llm_judge=llm_judge,
        )
        await pipeline.check_duplicate("Some content")

        assert llm_judge.judge.call_count <= 2
