"""Tests for smart auto-consolidation (Phase A), adaptive intervals (Phase B), and auto-dream (Phase C)."""

from __future__ import annotations

from surreal_memory.mcp.maintenance_handler import (
    HealthHint,
    HealthPulse,
    HintSeverity,
    _compute_adaptive_interval,
    _compute_effective_cooldown,
    _evaluate_thresholds,
    _select_strategies,
)
from surreal_memory.unified_config import MaintenanceConfig

# ========== Phase A: _evaluate_thresholds returns HealthHint ==========


class TestEvaluateThresholdsReturnsHealthHints:
    def test_empty_when_healthy(self) -> None:
        cfg = MaintenanceConfig()
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=50,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        assert hints == []

    def test_high_neuron_count_returns_prune_hint(self) -> None:
        cfg = MaintenanceConfig(neuron_warn_threshold=100)
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=150,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        assert len(hints) == 1
        assert isinstance(hints[0], HealthHint)
        assert hints[0].severity == HintSeverity.MEDIUM
        assert hints[0].recommended_strategy == "prune"

    def test_high_fiber_count_returns_merge_hint(self) -> None:
        cfg = MaintenanceConfig(fiber_warn_threshold=50)
        hints = _evaluate_thresholds(
            fiber_count=100,
            neuron_count=50,
            synapse_count=200,
            connectivity=4.0,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        assert any(h.recommended_strategy == "merge" for h in hints)

    def test_low_connectivity_returns_enrich_hint(self) -> None:
        cfg = MaintenanceConfig()
        hints = _evaluate_thresholds(
            fiber_count=10,
            neuron_count=50,
            synapse_count=40,
            connectivity=0.8,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        assert any(h.recommended_strategy == "enrich" for h in hints)
        assert any(h.severity == HintSeverity.LOW for h in hints)

    def test_high_orphan_ratio_critical_severity(self) -> None:
        cfg = MaintenanceConfig(orphan_ratio_threshold=0.25)
        hints = _evaluate_thresholds(
            fiber_count=5,
            neuron_count=100,
            synapse_count=200,
            connectivity=2.0,
            orphan_ratio=0.5,
            cfg=cfg,
        )
        orphan_hints = [h for h in hints if "orphan" in h.message.lower()]
        assert len(orphan_hints) == 1
        assert orphan_hints[0].severity == HintSeverity.CRITICAL

    def test_moderate_orphan_ratio_medium_severity(self) -> None:
        cfg = MaintenanceConfig(orphan_ratio_threshold=0.25)
        hints = _evaluate_thresholds(
            fiber_count=5,
            neuron_count=100,
            synapse_count=200,
            connectivity=2.0,
            orphan_ratio=0.3,
            cfg=cfg,
        )
        orphan_hints = [h for h in hints if "orphan" in h.message.lower()]
        assert len(orphan_hints) == 1
        assert orphan_hints[0].severity == HintSeverity.MEDIUM

    def test_backward_compat_hint_messages(self) -> None:
        cfg = MaintenanceConfig(neuron_warn_threshold=10)
        hints = _evaluate_thresholds(
            fiber_count=5,
            neuron_count=50,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.1,
            cfg=cfg,
        )
        pulse = HealthPulse(
            fiber_count=5,
            neuron_count=50,
            synapse_count=100,
            connectivity=2.0,
            orphan_ratio=0.1,
            expired_memory_count=0,
            stale_fiber_ratio=0.0,
            hints=tuple(hints),
            should_consolidate=True,
        )
        # hint_messages property should return strings
        assert len(pulse.hint_messages) == len(hints)
        assert all(isinstance(m, str) for m in pulse.hint_messages)


# ========== Phase A: _compute_effective_cooldown ==========


class TestComputeEffectiveCooldown:
    def test_no_hints_returns_base(self) -> None:
        assert _compute_effective_cooldown((), 60) == 60

    def test_one_low_hint_returns_base(self) -> None:
        hints = (HealthHint("test", HintSeverity.LOW, "prune"),)
        assert _compute_effective_cooldown(hints, 60) == 60

    def test_two_hints_halves_cooldown(self) -> None:
        hints = (
            HealthHint("a", HintSeverity.MEDIUM, "prune"),
            HealthHint("b", HintSeverity.LOW, "merge"),
        )
        assert _compute_effective_cooldown(hints, 60) == 30

    def test_four_hints_immediate(self) -> None:
        hints = tuple(HealthHint(f"h{i}", HintSeverity.MEDIUM, "prune") for i in range(4))
        assert _compute_effective_cooldown(hints, 60) == 0

    def test_critical_hint_immediate(self) -> None:
        hints = (HealthHint("critical", HintSeverity.CRITICAL, "prune"),)
        assert _compute_effective_cooldown(hints, 60) == 0


# ========== Phase A: _select_strategies ==========


class TestSelectStrategies:
    def test_no_hints_returns_fallback(self) -> None:
        assert _select_strategies(()) == ("prune", "merge")

    def test_single_hint_returns_its_strategy(self) -> None:
        hints = (HealthHint("test", HintSeverity.MEDIUM, "enrich"),)
        assert _select_strategies(hints) == ("enrich",)

    def test_deduplicates_strategies(self) -> None:
        hints = (
            HealthHint("a", HintSeverity.MEDIUM, "prune"),
            HealthHint("b", HintSeverity.LOW, "prune"),
        )
        assert _select_strategies(hints) == ("prune",)

    def test_orders_by_severity(self) -> None:
        hints = (
            HealthHint("a", HintSeverity.LOW, "enrich"),
            HealthHint("b", HintSeverity.CRITICAL, "prune"),
            HealthHint("c", HintSeverity.MEDIUM, "merge"),
        )
        result = _select_strategies(hints)
        assert result[0] == "prune"  # CRITICAL first
        assert "merge" in result
        assert "enrich" in result


# ========== Phase B: _compute_adaptive_interval ==========


class TestComputeAdaptiveInterval:
    def test_healthy_brain_doubles_interval(self) -> None:
        assert _compute_adaptive_interval(0, 25) == 50

    def test_healthy_brain_caps_at_50(self) -> None:
        assert _compute_adaptive_interval(0, 30) == 50

    def test_degrading_brain_keeps_base(self) -> None:
        assert _compute_adaptive_interval(1, 25) == 25
        assert _compute_adaptive_interval(2, 25) == 25

    def test_unhealthy_brain_reduces_interval(self) -> None:
        result = _compute_adaptive_interval(3, 25)
        assert result == 10  # 25 / 2.5 = 10

    def test_unhealthy_brain_minimum_5(self) -> None:
        result = _compute_adaptive_interval(5, 10)
        assert result == 5  # min(4, 5) = 5


# ========== Phase A: MaintenanceConfig defaults ==========


class TestMaintenanceConfigDefaults:
    def test_auto_consolidate_default_true(self) -> None:
        cfg = MaintenanceConfig()
        assert cfg.auto_consolidate is True

    def test_dream_cooldown_hours_default(self) -> None:
        cfg = MaintenanceConfig()
        assert cfg.dream_cooldown_hours == 24

    def test_from_dict_auto_consolidate_default_true(self) -> None:
        cfg = MaintenanceConfig.from_dict({})
        assert cfg.auto_consolidate is True

    def test_from_dict_explicit_false_respected(self) -> None:
        cfg = MaintenanceConfig.from_dict({"auto_consolidate": False})
        assert cfg.auto_consolidate is False

    def test_dream_cooldown_in_to_dict(self) -> None:
        cfg = MaintenanceConfig(dream_cooldown_hours=12)
        d = cfg.to_dict()
        assert d["dream_cooldown_hours"] == 12
