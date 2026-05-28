"""Unit tests for hooks.task_context."""

from __future__ import annotations

from argparse import Namespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.hooks import task_context
from surreal_memory.hooks.task_context import _read_note, save_task_context


def test_read_note_prefers_text_arg() -> None:
    """--text takes precedence over stdin."""
    assert _read_note(Namespace(text="hello note")) == "hello note"


async def test_save_task_context_blocked_by_firewall() -> None:
    """Firewall-blocked content returns saved=0 without touching storage."""
    with patch(
        "surreal_memory.safety.input_firewall.check_content",
        return_value=MagicMock(blocked=True, reason="content too short", sanitized=""),
    ):
        result = await save_task_context("garbage", "proj")

    assert result["saved"] == 0
    assert "blocked" in result["message"].lower()


async def test_save_task_context_saves_project_scoped_memory() -> None:
    """A valid note is encoded, summarised on its fiber, and saved scoped."""
    fiber = MagicMock()
    fiber.id = "fiber-1"
    fiber.with_summary = MagicMock(return_value=fiber)
    encode_result = MagicMock(fiber=fiber)

    encoder = MagicMock()
    encoder.encode = AsyncMock(return_value=encode_result)

    storage = AsyncMock()
    storage.get_brain = AsyncMock(return_value=MagicMock(config=MagicMock()))
    storage.disable_auto_save = MagicMock()
    storage.update_fiber = AsyncMock()
    storage.add_typed_memory = AsyncMock()
    storage.batch_save = AsyncMock()
    storage.close = AsyncMock()

    config = MagicMock()
    config.current_brain = "brain-1"
    config.safety.auto_redact_min_severity = "high"

    with (
        patch(
            "surreal_memory.safety.input_firewall.check_content",
            return_value=MagicMock(blocked=False, sanitized=""),
        ),
        patch(
            "surreal_memory.safety.sensitive.auto_redact_content",
            return_value=("redacted note text", [], None),
        ),
        patch("surreal_memory.unified_config.get_config", return_value=config),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            AsyncMock(return_value=storage),
        ),
        patch("surreal_memory.engine.encoder.MemoryEncoder", return_value=encoder),
    ):
        result = await save_task_context("a sufficiently long task note here", "myproj")

    assert result["saved"] == 1
    assert result["project"] == "myproj"
    fiber.with_summary.assert_called_once_with("redacted note text")
    storage.update_fiber.assert_awaited_once()
    storage.add_typed_memory.assert_awaited_once()
    # The saved typed memory carries the project scope.
    typed_mem = storage.add_typed_memory.await_args.args[0]
    assert typed_mem.project_id == "myproj"
    assert "project:myproj" in typed_mem.tags


async def test_save_task_context_no_brain_returns_error() -> None:
    """Missing brain is reported, not crashed."""
    storage = AsyncMock()
    storage.get_brain = AsyncMock(return_value=None)
    storage.close = AsyncMock()

    config = MagicMock()
    config.current_brain = "brain-1"

    with (
        patch(
            "surreal_memory.safety.input_firewall.check_content",
            return_value=MagicMock(blocked=False, sanitized=""),
        ),
        patch("surreal_memory.unified_config.get_config", return_value=config),
        patch(
            "surreal_memory.unified_config.get_shared_storage",
            AsyncMock(return_value=storage),
        ),
    ):
        result = await save_task_context("a sufficiently long task note here", "myproj")

    assert result["saved"] == 0
    assert "brain" in result["error"].lower()


def test_main_skips_short_note(monkeypatch) -> None:
    """A note below MIN_NOTE_CHARS exits cleanly without saving."""
    monkeypatch.setattr("sys.argv", ["smem-hook-task-context", "--text", "too short"])
    sentinel = MagicMock()
    monkeypatch.setattr(task_context, "save_task_context", sentinel)

    with pytest.raises(SystemExit) as exc:
        task_context.main()

    assert exc.value.code == 0
    sentinel.assert_not_called()
