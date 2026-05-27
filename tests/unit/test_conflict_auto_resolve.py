"""Tests for auto-conflict resolution (Phase E)."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from surreal_memory.engine.conflict_auto_resolve import (
    _count_superseded,
    try_auto_resolve,
)
from surreal_memory.engine.conflict_detection import Conflict, ConflictType
from surreal_memory.utils.timeutils import utcnow


def _make_conflict(
    existing_content: str = "We use PostgreSQL",
    new_content: str = "We use MySQL",
    confidence: float = 0.8,
) -> Conflict:
    return Conflict(
        type=ConflictType.FACTUAL_CONTRADICTION,
        existing_neuron_id="existing-id-1234",
        existing_content=existing_content,
        new_content=new_content,
        confidence=confidence,
    )


def _make_neuron(
    created_at: object | None = None,
    metadata: dict[str, object] | None = None,
) -> MagicMock:
    neuron = MagicMock()
    neuron.id = "existing-id-1234"
    neuron.created_at = created_at or utcnow()
    neuron.content = "We use PostgreSQL"
    neuron.metadata = metadata or {}
    return neuron


def _make_storage(
    neuron: object | None = None,
    state: object | None = None,
) -> AsyncMock:
    storage = AsyncMock()
    storage.get_neuron = AsyncMock(return_value=neuron)
    if state is None:
        mock_state = MagicMock()
        mock_state.activation_level = 0.5
        storage.get_neuron_state = AsyncMock(return_value=mock_state)
    else:
        storage.get_neuron_state = AsyncMock(return_value=state)
    return storage


class TestAutoResolveRules:
    @pytest.mark.asyncio
    async def test_missing_neuron_auto_resolves_keep_new(self) -> None:
        storage = _make_storage()
        storage.get_neuron = AsyncMock(return_value=None)
        conflict = _make_conflict()

        result = await try_auto_resolve(conflict, storage)
        assert result.auto_resolved is True
        assert result.resolution == "keep_new"

    @pytest.mark.asyncio
    async def test_rule1_stale_existing_high_confidence_new(self) -> None:
        """New confidence >= 0.8 AND existing is STALE -> keep_new."""
        old_time = utcnow() - timedelta(days=200)
        neuron = _make_neuron(created_at=old_time)
        storage = _make_storage(neuron=neuron)

        conflict = _make_conflict()
        result = await try_auto_resolve(conflict, storage, new_confidence=0.85)

        assert result.auto_resolved is True
        assert result.resolution == "keep_new"
        assert "stale" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rule1_ancient_existing(self) -> None:
        """New confidence >= 0.8 AND existing is ANCIENT -> keep_new."""
        old_time = utcnow() - timedelta(days=500)
        neuron = _make_neuron(created_at=old_time)
        storage = _make_storage(neuron=neuron)

        conflict = _make_conflict()
        result = await try_auto_resolve(conflict, storage, new_confidence=0.9)

        assert result.auto_resolved is True
        assert result.resolution == "keep_new"

    @pytest.mark.asyncio
    async def test_rule2_same_session_more_specific(self) -> None:
        """Same session (< 1h) AND new content is longer -> keep_new."""
        recent_time = utcnow() - timedelta(minutes=30)
        neuron = _make_neuron(created_at=recent_time)
        storage = _make_storage(neuron=neuron)

        conflict = _make_conflict(
            existing_content="Use PostgreSQL",
            new_content="We decided to use PostgreSQL with read replicas for scalability",
        )
        result = await try_auto_resolve(conflict, storage, new_confidence=0.5)

        assert result.auto_resolved is True
        assert result.resolution == "keep_new"
        assert "same session" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_rule3_repeatedly_superseded(self) -> None:
        """Existing neuron superseded 2+ times -> keep_new."""
        neuron = _make_neuron(
            metadata={"_disputed": True, "_superseded": True, "_dispute_count": 3}
        )
        storage = _make_storage(neuron=neuron)

        conflict = _make_conflict()
        result = await try_auto_resolve(conflict, storage, new_confidence=0.5)

        assert result.auto_resolved is True
        assert result.resolution == "keep_new"
        assert "superseded" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_safety_guard_both_fresh_requires_manual(self) -> None:
        """Both FRESH and high-confidence -> require manual."""
        recent_time = utcnow() - timedelta(days=2)
        neuron = _make_neuron(created_at=recent_time)
        state = MagicMock()
        state.activation_level = 0.7
        storage = _make_storage(neuron=neuron, state=state)

        conflict = _make_conflict()
        result = await try_auto_resolve(conflict, storage, new_confidence=0.8)

        assert result.auto_resolved is False
        assert result.resolution == ""
        assert "fresh" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_no_rule_matches_requires_manual(self) -> None:
        """When no rule matches, require manual resolution."""
        # Recent but not same-session, not stale, not superseded
        recent_time = utcnow() - timedelta(days=15)
        neuron = _make_neuron(created_at=recent_time)
        storage = _make_storage(neuron=neuron)

        # Low activation so safety guard doesn't fire
        state = MagicMock()
        state.activation_level = 0.3
        storage.get_neuron_state = AsyncMock(return_value=state)

        conflict = _make_conflict()
        result = await try_auto_resolve(conflict, storage, new_confidence=0.5)

        assert result.auto_resolved is False


class TestCountSuperseded:
    def test_empty_metadata(self) -> None:
        assert _count_superseded({}) == 0

    def test_disputed_only(self) -> None:
        assert _count_superseded({"_disputed": True}) == 1

    def test_disputed_and_superseded(self) -> None:
        assert _count_superseded({"_disputed": True, "_superseded": True}) == 2

    def test_dispute_count_overrides(self) -> None:
        assert _count_superseded({"_dispute_count": 5}) == 5
