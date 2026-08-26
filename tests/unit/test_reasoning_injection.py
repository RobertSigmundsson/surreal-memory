"""Tests for engine/reasoning_injection.py — model resolution + prompt block.

Uses synthetic transcripts / settings.json (HOME redirected to tmp) and
InMemoryStorage seeded with pattern fibers. Covers the resolve_active_model
fallback chain, injection_map glob/default matching, per-category + max-patterns
+ char-budget selection, markdown format, and the session idempotency markers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.reasoning_injection import (
    already_injected,
    build_injection_context,
    get_reasoning_context,
    mark_injected,
    resolve_active_model,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN = "b1"
_ENV_MODEL_VARS = ("SMEM_REASONING_TARGET_MODEL", "ANTHROPIC_MODEL")


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect HOME to tmp (no settings.json) and clear model env vars."""
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in _ENV_MODEL_VARS:
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _ucfg(tmp_path: Path, **rt_kw: object) -> UnifiedConfig:
    base: dict[str, object] = {
        "injection_enabled": True,
        "injection_max_patterns": 5,
        "injection_max_chars": 4000,
    }
    base.update(rt_kw)
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(**base),  # type: ignore[arg-type]
    )


async def _add_pattern(
    storage: InMemoryStorage,
    model: str,
    category: str,
    title: str,
    strategy: str,
    confidence: float,
    frequency: int,
) -> None:
    neuron = Neuron.create(type=NeuronType.CONCEPT, content=title)
    await storage.add_neuron(neuron)
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary=title,
        tags=set(),
        metadata={
            "_reasoning_pattern": True,
            "_source_model": model,
            "_reasoning_category": category,
            "_reasoning_title": title,
            "_reasoning_strategy": strategy,
            "_reasoning_confidence": confidence,
            "_reasoning_frequency": frequency,
            "_reasoning_signature": title,
        },
    )
    await storage.add_fiber(fiber)


# ── resolve_active_model chain ───────────────────────────────────────────────


def test_resolve_from_payload_model(clean_env: Path) -> None:
    assert resolve_active_model({"model": "claude-sonnet-5-20250101"}) == "claude-sonnet-5"


def test_resolve_from_transcript_tail(clean_env: Path) -> None:
    # Transcripts are only read from under ~/.claude (spoof guard). HOME is
    # redirected to clean_env, so write the transcript there.
    tp = clean_env / ".claude" / "projects" / "x" / "t.jsonl"
    tp.parent.mkdir(parents=True, exist_ok=True)
    tp.write_text(
        json.dumps({"type": "user", "message": {"content": "hi"}})
        + "\n"
        + json.dumps({"type": "assistant", "message": {"model": "claude-haiku-4-5-20251001"}})
        + "\n",
        encoding="utf-8",
    )
    assert resolve_active_model({"transcript_path": str(tp)}) == "claude-haiku-4-5"


def test_resolve_transcript_outside_claude_rejected(clean_env: Path) -> None:
    # A transcript_path outside ~/.claude is untrusted (hook stdin is attacker-
    # controllable) and ignored; resolution falls through to "default".
    outside = clean_env / "outside.jsonl"
    outside.write_text(
        json.dumps({"type": "assistant", "message": {"model": "claude-opus-4-8"}}) + "\n",
        encoding="utf-8",
    )
    assert resolve_active_model({"transcript_path": str(outside)}) == "default"


def test_resolve_transcript_symlink_escape_rejected(clean_env: Path) -> None:
    # A symlink UNDER ~/.claude that points outside it must also be rejected:
    # resolve() canonicalizes through the symlink before the containment check,
    # so this can't be bypassed with a symlink (only a literal path prefix check
    # would be fooled). Guards against a future refactor to os.path.abspath.
    outside = clean_env / "real.jsonl"
    outside.write_text(
        json.dumps({"type": "assistant", "message": {"model": "claude-opus-4-8"}}) + "\n",
        encoding="utf-8",
    )
    claude_dir = clean_env / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    link = claude_dir / "sneaky.jsonl"
    link.symlink_to(outside)
    assert resolve_active_model({"transcript_path": str(link)}) == "default"


def test_resolve_from_env(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMEM_REASONING_TARGET_MODEL", "claude-opus-4-8")
    assert resolve_active_model({}) == "claude-opus-4-8"


def test_resolve_from_env_alias(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "sonnet")
    assert resolve_active_model({}) == "claude-sonnet-5"


def test_resolve_from_settings_alias(clean_env: Path) -> None:
    claude_dir = clean_env / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.json").write_text(json.dumps({"model": "opusplan"}), encoding="utf-8")
    assert resolve_active_model({}) == "claude-opus-5"


def test_model_aliases_track_the_current_lineup() -> None:
    """Regression: the alias table shipped a stale 4.8-era id for months.

    Injection is keyed on the model string in ``injection_map``, so a stale
    alias here silently routes a user's short "opus" setting to the wrong
    source model's patterns.
    """
    from surreal_memory.engine.reasoning_injection import _MODEL_ALIASES

    assert _MODEL_ALIASES["opus"] == "claude-opus-5"
    assert _MODEL_ALIASES["opusplan"] == "claude-opus-5"


def test_resolve_default_when_nothing_available(clean_env: Path) -> None:
    assert resolve_active_model({}) == "default"


def test_resolve_precedence_payload_over_env(
    clean_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMEM_REASONING_TARGET_MODEL", "claude-haiku-4-5")
    assert resolve_active_model({"model": "claude-opus-4-8"}) == "claude-opus-4-8"


# ── build_injection_context ──────────────────────────────────────────────────


async def test_build_block_renders_ranked(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage, "claude-fable-5", "planning", "planning: plan", "plan -> steps", 0.8, 2
    )
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: verify", "verify -> check", 1.0, 3
    )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert block.startswith("## Reasoning strategies (learned from claude-fable-5)")
    assert "debugging: verify" in block
    assert "planning: plan" in block
    # Higher confidence*frequency ranks first.
    assert block.index("debugging: verify") < block.index("planning: plan")


async def test_build_block_multi_source(tmp_path: Path) -> None:
    """A comma-separated map value yields one block PER source, in order.

    Learn-from-the-stronger doctrine (Robert, 2026-07-27): models below the
    apex get both apex banks. A source with no patterns contributes nothing
    (no empty header), and each block keeps its own budget.
    """
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: fable-way", "verify", 1.0, 3
    )
    await _add_pattern(storage, "claude-opus-5", "planning", "planning: opus-way", "plan", 0.9, 2)
    cfg = _ucfg(
        tmp_path,
        injection_map=(("claude-sonnet-*", "claude-fable-5,claude-opus-5"),),
    )

    block = await build_injection_context(storage, "claude-sonnet-5", cfg)
    assert "## Reasoning strategies (learned from claude-fable-5)" in block
    assert "## Reasoning strategies (learned from claude-opus-5)" in block
    assert "debugging: fable-way" in block
    assert "planning: opus-way" in block
    # Source order from the map value is preserved.
    assert block.index("claude-fable-5") < block.index("claude-opus-5")

    # Unknown third source is silently skipped — no empty header.
    cfg2 = _ucfg(
        tmp_path,
        injection_map=(("default", "claude-fable-5,claude-ghost-9"),),
    )
    block2 = await build_injection_context(storage, "anything", cfg2)
    assert "claude-fable-5" in block2
    assert "claude-ghost-9" not in block2


async def test_injection_disabled_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(
        tmp_path, injection_enabled=False, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_no_map_match_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-haiku-*", "claude-fable-5"),))
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_injection_map_default_key(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: x", "s", 1.0, 3)
    cfg = _ucfg(
        tmp_path,
        injection_map=(("claude-haiku-*", "nope"), ("default", "claude-fable-5")),
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert "learned from claude-fable-5" in block


async def test_source_without_patterns_returns_empty(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "t", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-sonnet-5"),))
    assert await build_injection_context(storage, "claude-opus-4-8", cfg) == ""


async def test_max_two_per_category(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    for i in range(4):
        await _add_pattern(
            storage, "claude-fable-5", "debugging", f"debugging: {i}", "s", 1.0, 3 - i
        )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    # Only 2 of the 4 debugging patterns are injected.
    assert block.count("debugging:") == 2


async def test_injection_max_patterns_cap(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: a", "s", 1.0, 3)
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", "s", 0.9, 3)
    cfg = _ucfg(
        tmp_path, injection_max_patterns=1, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    assert "1. **" in block
    assert "2. **" not in block


async def test_char_budget_limits_entries(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    long_strategy = "x" * 500
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: a", long_strategy, 1.0, 3
    )
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", long_strategy, 0.9, 3)
    cfg = _ucfg(
        tmp_path, injection_max_chars=120, injection_map=(("claude-opus-*", "claude-fable-5"),)
    )
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)
    # First entry always included; the second exceeds the tiny budget.
    assert "1. **" in block
    assert "2. **" not in block


async def test_build_block_warns_on_fetch_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # When the pattern-fiber fetch hits its ceiling, a model's patterns could be
    # truncated out; that must be a visible warning, not a silent empty injection.
    from surreal_memory.engine import reasoning_injection as ri

    monkeypatch.setattr(ri, "_PATTERN_FETCH_LIMIT", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: a", "s", 1.0, 3)
    await _add_pattern(storage, "claude-fable-5", "planning", "planning: b", "s", 0.9, 2)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    with caplog.at_level(logging.WARNING, logger="surreal_memory.engine.reasoning_injection"):
        block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert "ceiling" in caplog.text
    assert block  # still renders from whatever was fetched


# ── get_reasoning_context (shared hook orchestrator) ─────────────────────────


async def test_get_reasoning_context_disabled_returns_empty(
    clean_env: Path, tmp_path: Path
) -> None:
    cfg = _ucfg(tmp_path, injection_enabled=False)
    with patch("surreal_memory.unified_config.get_config", return_value=cfg):
        assert await get_reasoning_context({"session_id": "s-disabled"}) == ""


async def test_get_reasoning_context_builds_and_marks_session(
    clean_env: Path, tmp_path: Path
) -> None:
    storage = InMemoryStorage()
    storage.set_brain("default")
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: verify", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            new=AsyncMock(return_value=storage),
        ),
    ):
        block = await get_reasoning_context({"session_id": "s-happy", "model": "claude-opus-4-8"})

    assert "learned from claude-fable-5" in block
    # Session marker set → the sibling UserPromptSubmit hook won't re-inject.
    assert already_injected("s-happy") is True


async def test_get_reasoning_context_skips_when_already_injected(
    clean_env: Path, tmp_path: Path
) -> None:
    mark_injected("s-dup")  # e.g. SessionStart already injected this session
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))

    async def _must_not_open(_name: str) -> object:
        raise AssertionError("storage must not open when the session is already injected")

    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch("surreal_memory.unified_config.get_shared_storage", new=_must_not_open),
    ):
        result = await get_reasoning_context({"session_id": "s-dup", "model": "claude-opus-4-8"})

    assert result == ""


async def test_subagent_start_injects_every_time_despite_session_marker(
    clean_env: Path, tmp_path: Path
) -> None:
    # SubagentStart fires once per spawned subagent; the payload carries the
    # PARENT session_id, so honoring the per-session marker would starve every
    # subagent after the first injection. It must inject regardless of the
    # marker — and must NOT set it (the marker belongs to main-session hooks).
    mark_injected("s-parent")  # parent session already injected via SessionStart
    storage = InMemoryStorage()
    storage.set_brain("default")
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: verify", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            new=AsyncMock(return_value=storage),
        ),
    ):
        block = await get_reasoning_context(
            {
                "session_id": "s-parent",
                "model": "claude-opus-4-8",
                "hook_event_name": "SubagentStart",
            }
        )
        # A fresh session id via SubagentStart must not consume the session marker.
        fresh = await get_reasoning_context(
            {
                "session_id": "s-fresh",
                "model": "claude-opus-4-8",
                "hook_event_name": "SubagentStart",
            }
        )

    assert "learned from claude-fable-5" in block
    assert "learned from claude-fable-5" in fresh
    assert already_injected("s-fresh") is False


async def test_session_start_compact_reinjects_despite_session_marker(
    clean_env: Path, tmp_path: Path
) -> None:
    # Compaction flattens the conversation, destroying the previously injected
    # block. SessionStart(source=compact) must re-inject despite the session
    # marker; all other sources (resume/startup/clear) keep honoring it — their
    # history, including the original block, is still in the context.
    mark_injected("s-compact")
    storage = InMemoryStorage()
    storage.set_brain("default")
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: verify", "s", 1.0, 3)
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            new=AsyncMock(return_value=storage),
        ),
    ):
        compacted = await get_reasoning_context(
            {
                "session_id": "s-compact",
                "model": "claude-opus-4-8",
                "hook_event_name": "SessionStart",
                "source": "compact",
            }
        )
        resumed = await get_reasoning_context(
            {
                "session_id": "s-compact",
                "model": "claude-opus-4-8",
                "hook_event_name": "SessionStart",
                "source": "resume",
            }
        )

    assert "learned from claude-fable-5" in compacted
    assert resumed == ""


# ── session idempotency markers ──────────────────────────────────────────────


def test_marker_roundtrip(clean_env: Path) -> None:
    assert already_injected("sess-1") is False
    mark_injected("sess-1")
    assert already_injected("sess-1") is True
    # Empty session id is a no-op and never "already injected".
    assert already_injected("") is False
    mark_injected("")


def test_marker_sanitizes_session_id(clean_env: Path) -> None:
    mark_injected("../evil/../id")
    marker_root = clean_env / ".surrealmemory" / "reasoning_injected"
    assert marker_root.is_dir()
    # No traversal escape — every marker stays directly under the marker root.
    assert all(p.parent == marker_root for p in marker_root.iterdir())
    assert already_injected("../evil/../id") is True


def test_marker_dir_honors_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # SURREAL_MEMORY_DIR relocates the whole data dir; markers must follow it
    # (matches hooks/post_tool_use._get_data_dir) rather than pinning ~/.surrealmemory.
    data_dir = tmp_path / "custom-data"
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(data_dir))
    mark_injected("sess-env")
    assert (data_dir / "reasoning_injected" / "sess-env").exists()
    assert already_injected("sess-env") is True


# ── Round-robin across categories (audit defect 4) ───────────────────────────
#
# confidence is a SHARE of a category (size / traces_in_category), so ranking by
# confidence x frequency reduces to size^2 / traces_in_category and rewards
# whichever category is rarest. Measured 2026-08-26: the top 8 for both opus-5
# and fable-5 were 8/8 "verification"; only the per-category cap kept the
# injected block from being five of the same thing.


async def test_top_slots_are_shared_across_categories(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    # A bank dominated by one category, with more categories than slots. Taking
    # the globally best five spends two slots on the dominant category before
    # reaching the fourth, so the two weakest categories never appear at all.
    for i in range(3):
        await _add_pattern(
            storage, "claude-fable-5", "verification", f"verification: v{i}", "s", 1.0, 100 - i
        )
    for name, freq in (
        ("debugging", 50),
        ("planning", 40),
        ("research", 30),
        ("refactoring", 20),
        ("architecture", 10),
    ):
        await _add_pattern(storage, "claude-fable-5", name, f"{name}: only", "s", 1.0, freq)

    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    # Five slots, five different categories — one strategy each, not two of the
    # loudest and none of the quietest.
    assert block.count("verification:") == 1
    for name in ("debugging", "planning", "research", "refactoring"):
        assert f"{name}: only" in block, name


async def test_the_per_category_safety_belt_still_holds(tmp_path: Path) -> None:
    # Round-robin provides the diversity; _MAX_PER_CATEGORY stays as the belt.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    for i in range(8):
        await _add_pattern(
            storage, "claude-fable-5", "verification", f"verification: v{i}", "s", 1.0, 100 - i
        )
    await _add_pattern(storage, "claude-fable-5", "debugging", "debugging: d", "s", 0.4, 2)

    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert block.count("verification:") == 2
    assert "debugging: d" in block


async def test_a_single_category_bank_is_unchanged(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    for i in range(4):
        await _add_pattern(
            storage, "claude-fable-5", "debugging", f"debugging: {i}", "s", 1.0, 10 - i
        )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    # Nothing to round-robin with: the two best of the only category, as before.
    assert block.count("debugging:") == 2
    assert "debugging: 0" in block and "debugging: 1" in block


async def test_selection_is_deterministic_when_ranks_tie(tmp_path: Path) -> None:
    # Same confidence x frequency for every pattern: without an explicit
    # tie-break the row order from storage would decide what gets injected.
    blocks = []
    for _ in range(3):
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        for name in ("c", "a", "b", "d"):
            await _add_pattern(
                storage, "claude-fable-5", "debugging", f"debugging: {name}", "s", 0.5, 4
            )
        cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
        blocks.append(await build_injection_context(storage, "claude-opus-4-8", cfg))

    assert blocks[0] == blocks[1] == blocks[2]
    # Ties resolve by title, so the choice is explainable rather than incidental.
    assert "debugging: a" in blocks[0] and "debugging: b" in blocks[0]
    assert "debugging: c" not in blocks[0]


# ── Legacy patterns must not claim a measured order (defect 2 residue) ───────
#
# Found by the U2 checker: patterns distilled before segment_moves walked the
# text still sit in the bank rendering "Moves: a -> b -> c", and re-distillation
# will not overwrite them (their signature keys on the trace set). On the live
# corpus that rendered order disagreed with the text in 67.5% of traces that had
# one, so the injected block was teaching a sequence that may never have run.


async def test_a_pattern_without_a_measured_chain_does_not_claim_one(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    # No _reasoning_chain in metadata → distilled before the chain was measured.
    await _add_pattern(
        storage,
        "claude-fable-5",
        "debugging",
        "debugging: verify, restate-goal",
        "Moves: verify -> restate-goal\nmedoid text",
        1.0,
        3,
    )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert "Moves (order unverified): verify, restate-goal" in block
    assert "->" not in block
    assert "medoid text" in block  # the rest of the strategy survives


async def test_a_measured_chain_still_renders_as_a_chain(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage,
        "claude-fable-5",
        "debugging",
        "debugging: verify, restate-goal",
        "Moves: verify -> restate-goal\nmedoid text",
        1.0,
        3,
    )
    # Mark it as measured, exactly as the distiller now does.
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=10)
    await storage.update_fiber_metadata(
        fibers[0].id,
        {"_reasoning_chain": ["verify", "restate-goal"], "_reasoning_chain_source": "measured"},
    )

    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert "Moves: verify -> restate-goal" in block
    assert "order unverified" not in block


async def test_an_unordered_legacy_line_is_left_alone(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage,
        "claude-fable-5",
        "debugging",
        "debugging: verify",
        "Moves (unordered): verify, backtrack\nmedoid",
        1.0,
        3,
    )
    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    # It already says it has no order; nothing to downgrade.
    assert "Moves (unordered): verify, backtrack" in block


async def test_a_chain_recovered_by_parsing_is_not_a_measured_chain(tmp_path: Path) -> None:
    # The retro-merge stamps _reasoning_chain on legacy fibers too, recovered by
    # parsing their rendered line. That parse cannot distinguish a real chain
    # from the old top-3 fallback, so it must not buy the fiber its arrows back.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage,
        "claude-fable-5",
        "debugging",
        "debugging: verify, restate-goal",
        "Moves: verify -> restate-goal\nmedoid",
        1.0,
        3,
    )
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=10)
    await storage.update_fiber_metadata(
        fibers[0].id,
        {
            "_reasoning_chain": ["verify", "restate-goal"],
            "_reasoning_chain_source": "legacy-parsed",
        },
    )

    cfg = _ucfg(tmp_path, injection_map=(("claude-opus-*", "claude-fable-5"),))
    block = await build_injection_context(storage, "claude-opus-4-8", cfg)

    assert "Moves (order unverified): verify, restate-goal" in block
    assert "->" not in block


async def test_two_sources_do_not_deliver_the_same_strategy_twice(tmp_path: Path) -> None:
    # A sonnet/haiku session is mapped to two source models. Both banks hold a
    # pattern with the same title, and the block-level dedup could not see it:
    # each block was fine on its own while the session got the strategy twice in
    # one prompt. Measured on the live bank before this fix.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    shared = "debugging: restate-goal, verify, backtrack"
    await _add_pattern(storage, "claude-fable-5", "debugging", shared, "s", 1.0, 9)
    await _add_pattern(storage, "claude-opus-5", "debugging", shared, "s", 1.0, 9)
    await _add_pattern(storage, "claude-opus-5", "planning", "planning: distinct", "s", 1.0, 8)

    cfg = _ucfg(tmp_path, injection_map=(("claude-sonnet-5", "claude-fable-5,claude-opus-5"),))
    block = await build_injection_context(storage, "claude-sonnet-5", cfg)

    assert block.count(shared) == 1
    # The second source still contributes — it just spends the slot on
    # something the session does not already have.
    assert "planning: distinct" in block
    assert block.count("## Reasoning strategies") == 2


async def test_a_source_whose_every_pattern_was_already_delivered_adds_no_header(
    tmp_path: Path,
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    shared = "debugging: only one"
    await _add_pattern(storage, "claude-fable-5", "debugging", shared, "s", 1.0, 9)
    await _add_pattern(storage, "claude-opus-5", "debugging", shared, "s", 1.0, 9)

    cfg = _ucfg(tmp_path, injection_map=(("claude-sonnet-5", "claude-fable-5,claude-opus-5"),))
    block = await build_injection_context(storage, "claude-sonnet-5", cfg)

    # No empty second header (the existing rule: a source with nothing to say
    # contributes nothing).
    assert block.count("## Reasoning strategies") == 1
    assert block.count(shared) == 1


async def test_the_same_moves_in_a_different_order_are_not_two_strategies(
    tmp_path: Path,
) -> None:
    # Two source banks hold the same three moves in different orders, and
    # NEITHER has a measured chain — both render "(order unverified)". Serving
    # both spends two slots on one strategy, which is what the block-level and
    # then the literal-title dedup both missed. Measured on the live bank:
    # four of ten slots on a sonnet route.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage,
        "claude-fable-5",
        "debugging",
        "debugging: restate-goal, verify, backtrack",
        "Moves: a -> b\nx",
        1.0,
        9,
    )
    await _add_pattern(
        storage,
        "claude-opus-5",
        "debugging",
        "debugging: backtrack, restate-goal, verify",
        "Moves: b -> a\nx",
        1.0,
        9,
    )
    await _add_pattern(storage, "claude-opus-5", "planning", "planning: distinct", "s", 1.0, 8)

    cfg = _ucfg(tmp_path, injection_map=(("claude-sonnet-5", "claude-fable-5,claude-opus-5"),))
    block = await build_injection_context(storage, "claude-sonnet-5", cfg)

    assert "debugging: restate-goal, verify, backtrack" in block
    assert "debugging: backtrack, restate-goal, verify" not in block
    assert "planning: distinct" in block  # the second source still contributes


async def test_a_measured_order_is_still_its_own_strategy(tmp_path: Path) -> None:
    # The mirror case: when the order WAS measured, a different route through
    # the same moves is a genuinely different strategy and must survive.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _add_pattern(
        storage, "claude-fable-5", "debugging", "debugging: verify, backtrack", "s", 1.0, 9
    )
    await _add_pattern(
        storage, "claude-opus-5", "debugging", "debugging: backtrack, verify", "s", 1.0, 9
    )
    for f in await storage.find_fibers(metadata_key="_reasoning_pattern", limit=10):
        await storage.update_fiber_metadata(f.id, {"_reasoning_chain_source": "measured"})

    cfg = _ucfg(tmp_path, injection_map=(("claude-sonnet-5", "claude-fable-5,claude-opus-5"),))
    block = await build_injection_context(storage, "claude-sonnet-5", cfg)

    assert "debugging: verify, backtrack" in block
    assert "debugging: backtrack, verify" in block
