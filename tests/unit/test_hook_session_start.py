"""Tests for the SessionStart Claude Code hook."""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.hooks.session_start import get_recent_memories, main, read_hook_input


@pytest.mark.asyncio
async def test_get_recent_memories_empty_brain() -> None:
    """Empty brain returns empty string — no context injected."""
    mock_storage = AsyncMock()
    mock_storage.get_fibers = AsyncMock(return_value=[])
    mock_storage.close = AsyncMock()

    mock_config = MagicMock()
    mock_config.current_brain = "test"

    with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            return_value=mock_storage,
        ):
            result = await get_recent_memories()

    assert result == ""


@pytest.mark.asyncio
async def test_get_recent_memories_returns_formatted_bullets() -> None:
    """Fibers with summary/essence are formatted as a bullet list."""
    fiber_with_summary = MagicMock()
    fiber_with_summary.summary = "Fixed auth bug in login.py"
    fiber_with_summary.essence = None

    fiber_with_essence_only = MagicMock()
    fiber_with_essence_only.summary = None
    fiber_with_essence_only.essence = "Auth bug"

    fiber_empty = MagicMock()
    fiber_empty.summary = None
    fiber_empty.essence = None

    mock_storage = AsyncMock()
    mock_storage.get_fibers = AsyncMock(
        return_value=[fiber_with_summary, fiber_with_essence_only, fiber_empty]
    )
    mock_storage.close = AsyncMock()

    mock_config = MagicMock()
    mock_config.current_brain = "test"

    with patch("surreal_memory.unified_config.get_config", return_value=mock_config):
        with patch(
            "surreal_memory.unified_config.get_shared_storage",
            return_value=mock_storage,
        ):
            result = await get_recent_memories()

    lines = result.splitlines()
    assert len(lines) == 2
    assert lines[0] == "- Fixed auth bug in login.py"
    assert lines[1] == "- Auth bug"


def test_main_malformed_stdin_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    """Malformed or empty stdin is handled gracefully — hook never blocks Claude Code."""
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch("asyncio.run", side_effect=RuntimeError("storage unavailable")):
            with pytest.raises(SystemExit) as exc_info:
                main()

    # Must exit 0 — never block Claude Code
    assert exc_info.value.code == 0


def test_read_hook_input_empty_stdin() -> None:
    """Empty stdin returns an empty dict without raising."""
    with patch("sys.stdin", io.StringIO("")):
        result = read_hook_input()
    assert result == {}


def test_read_hook_input_valid_json() -> None:
    """Valid JSON on stdin is parsed and returned."""
    payload = {"session_id": "abc123", "turn": 1}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        result = read_hook_input()
    assert result == payload


def test_main_outputs_context_json_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    """When memories exist, main() writes a JSON context response to stdout."""
    with patch("sys.stdin", io.StringIO("{}")):
        with patch("asyncio.run", return_value="- Fixed auth bug\n- Added retry logic"):
            main()  # happy path — no sys.exit, just returns

    captured = capsys.readouterr()
    response = json.loads(captured.out.strip())
    assert response["type"] == "context"
    assert "Fixed auth bug" in response["content"]


def test_main_exits_silently_when_no_memories(capsys: pytest.CaptureFixture[str]) -> None:
    """When the brain has no memories, main() exits 0 with no stdout output."""
    with patch("sys.stdin", io.StringIO("{}")):
        with patch("asyncio.run", return_value=""):
            with pytest.raises(SystemExit) as exc_info:
                main()

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
