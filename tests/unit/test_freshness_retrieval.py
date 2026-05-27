"""Tests for freshness-weighted retrieval (Phase D)."""

from __future__ import annotations

from datetime import timedelta

from surreal_memory.core.brain import BrainConfig
from surreal_memory.safety.freshness import FreshnessLevel, evaluate_freshness
from surreal_memory.utils.timeutils import utcnow


class TestFreshnessWeightConfig:
    def test_default_freshness_weight_zero(self) -> None:
        cfg = BrainConfig()
        assert cfg.freshness_weight == 0.0

    def test_with_updates_freshness_weight(self) -> None:
        cfg = BrainConfig()
        updated = cfg.with_updates(freshness_weight=0.5)
        assert updated.freshness_weight == 0.5
        # Original unchanged
        assert cfg.freshness_weight == 0.0


class TestFreshnessScoreMultipliers:
    """Verify that the freshness weight formula produces expected multipliers."""

    def test_fw_zero_no_effect(self) -> None:
        """fw=0.0 should produce multiplier of 1.0 regardless of age."""
        fw = 0.0
        for level in FreshnessLevel:
            now = utcnow()
            age_map = {
                FreshnessLevel.FRESH: timedelta(days=1),
                FreshnessLevel.RECENT: timedelta(days=15),
                FreshnessLevel.AGING: timedelta(days=60),
                FreshnessLevel.STALE: timedelta(days=200),
                FreshnessLevel.ANCIENT: timedelta(days=500),
            }
            created = now - age_map[level]
            result = evaluate_freshness(created, now)
            multiplier = (1.0 - fw) + fw * result.score
            assert multiplier == 1.0, f"fw=0 should give 1.0 for {level}"

    def test_fw_half_fresh_no_penalty(self) -> None:
        """fw=0.5, FRESH memory should have score=1.0 -> multiplier=1.0."""
        fw = 0.5
        now = utcnow()
        created = now - timedelta(days=1)
        result = evaluate_freshness(created, now)
        assert result.level == FreshnessLevel.FRESH
        multiplier = (1.0 - fw) + fw * result.score
        assert multiplier == 1.0

    def test_fw_half_stale_penalized(self) -> None:
        """fw=0.5, STALE memory (score=0.3) -> multiplier=0.65."""
        fw = 0.5
        now = utcnow()
        created = now - timedelta(days=200)
        result = evaluate_freshness(created, now)
        assert result.level == FreshnessLevel.STALE
        assert result.score == 0.3
        multiplier = (1.0 - fw) + fw * result.score
        assert abs(multiplier - 0.65) < 0.01

    def test_fw_half_ancient_heavily_penalized(self) -> None:
        """fw=0.5, ANCIENT memory (score=0.1) -> multiplier=0.55."""
        fw = 0.5
        now = utcnow()
        created = now - timedelta(days=500)
        result = evaluate_freshness(created, now)
        assert result.level == FreshnessLevel.ANCIENT
        assert result.score == 0.1
        multiplier = (1.0 - fw) + fw * result.score
        assert abs(multiplier - 0.55) < 0.01

    def test_fw_one_full_penalty(self) -> None:
        """fw=1.0, ANCIENT memory (score=0.1) -> multiplier=0.1."""
        fw = 1.0
        now = utcnow()
        created = now - timedelta(days=500)
        result = evaluate_freshness(created, now)
        multiplier = (1.0 - fw) + fw * result.score
        assert abs(multiplier - 0.1) < 0.01
