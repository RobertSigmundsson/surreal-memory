"""Tests for the UserPromptSubmit Claude Code hook.

The hook emits the reasoning-strategies block inside a hookSpecificOutput JSON
envelope (additionalContext) — the only channel Claude Code adds to the model's
context for this event — and always exits 0 so it can never block the prompt.
The block itself is produced by the shared
engine.reasoning_injection.get_reasoning_context orchestrator, which is patched
here — its own behavior (resolve/build/marker) is covered in
test_reasoning_injection.py.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest

from surreal_memory.hooks.user_prompt_submit import main, read_hook_input

_ORCHESTRATOR = "surreal_memory.engine.reasoning_injection.get_reasoning_context"
_REASONING_BLOCK = "## Reasoning strategies (learned from claude-fable-5)\n\n1. **plan**"


def test_read_hook_input_empty_stdin() -> None:
    with patch("sys.stdin", io.StringIO("")):
        assert read_hook_input() == {}


def test_read_hook_input_valid_json() -> None:
    payload = {"session_id": "s1", "transcript_path": "/x/t.jsonl", "prompt": "hi"}
    with patch("sys.stdin", io.StringIO(json.dumps(payload))):
        assert read_hook_input() == payload


def test_read_hook_input_malformed_json() -> None:
    with patch("sys.stdin", io.StringIO("not json")):
        assert read_hook_input() == {}


def test_main_emits_hook_specific_output_json(capsys: pytest.CaptureFixture[str]) -> None:
    # Context reaches the model ONLY via hookSpecificOutput.additionalContext —
    # the hook must emit that JSON envelope, not raw stdout.
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value=_REASONING_BLOCK)):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    hook_out = payload["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "UserPromptSubmit"
    assert hook_out["additionalContext"] == _REASONING_BLOCK


def test_main_no_block_prints_nothing_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value="")):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""  # nothing injected into the prompt
    # Komunikat zmieniony świadomie: hook ma teraz DWA źródła (recall + strategie),
    # więc "brak strategii" przestało opisywać stan.
    assert "Nothing to inject" in captured.err


def test_main_injection_failure_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    # An orchestrator exception must never block the prompt — exit 0, no stdout.
    with patch("sys.stdin", io.StringIO("{}")):
        with patch(_ORCHESTRATOR, new=AsyncMock(side_effect=RuntimeError("boom"))):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "failed" in captured.err.lower()


def test_main_malformed_stdin_does_not_crash(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("sys.stdin", io.StringIO("not valid json")):
        with patch(_ORCHESTRATOR, new=AsyncMock(return_value="")):
            with pytest.raises(SystemExit) as exc:
                main()

    assert exc.value.code == 0


# ── Per-prompt memory recall ─────────────────────────────────────────────────
#
# SessionStart can only LIST the newest project memories — it fires before the
# user has said anything, so it has nothing to search with. This hook holds the
# question, so it is the only place a topic-keyed query is possible. Measured on
# the live brain before this existed: 51% of neurons never accessed, recall
# confidence 20%.

_RECALL = "surreal_memory.hooks.user_prompt_submit.get_prompt_recall"


def _cfg(**kw: object):
    from surreal_memory.unified_config import PromptRecallConfig

    return PromptRecallConfig.from_dict({"enabled": True, **kw})


async def _pipeline_returning(context: str):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.context = context
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=result)
    return pipeline


@pytest.mark.asyncio
async def test_recall_is_keyed_on_the_prompt_not_on_recency() -> None:
    """The whole point: the query is the user's own words."""
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    pipeline = await _pipeline_returning("- coś o rclone")
    storage = AsyncMock()
    storage.brain_id = "b1"
    storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())
    prompt = "co ustaliliśmy o bisync i sierocym locku rclone?"

    with (
        patch("surreal_memory.unified_config.get_config") as gc,
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
    ):
        gc.return_value.prompt_recall = _cfg()
        gc.return_value.current_brain = "b1"
        out = await get_prompt_recall({"prompt": prompt, "session_id": "s1"})

    assert pipeline.query.await_args.kwargs["query"] == prompt
    assert "coś o rclone" in out
    assert out.startswith("## Relevant memory")


@pytest.mark.asyncio
async def test_recall_is_off_by_default() -> None:
    """Opt-in: an unconfigured brain must not pay latency on every turn."""
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall
    from surreal_memory.unified_config import PromptRecallConfig

    with patch("surreal_memory.unified_config.get_config") as gc:
        gc.return_value.prompt_recall = PromptRecallConfig()
        assert await get_prompt_recall({"prompt": "x" * 200}) == ""


@pytest.mark.asyncio
async def test_short_prompt_spends_no_query() -> None:
    """ "ok" / "dalej" carry no question — searching on them is noise, not recall."""
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    with patch("surreal_memory.unified_config.get_config") as gc:
        gc.return_value.prompt_recall = _cfg(min_prompt_chars=40)
        assert await get_prompt_recall({"prompt": "ok, dalej"}) == ""


@pytest.mark.asyncio
async def test_a_slow_brain_yields_nothing_rather_than_stalling_the_prompt() -> None:
    """Memory that makes the user wait is worse than memory that stays quiet."""
    import asyncio

    from surreal_memory.hooks.user_prompt_submit import _recall_within_timeout

    async def _never(*_a: object, **_k: object) -> str:
        await asyncio.sleep(10)
        return "too late"

    with patch(_RECALL, _never):
        assert await _recall_within_timeout({"prompt": "x" * 100}, seconds=0.05) == ""


def test_main_injects_recall_and_strategies_together(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("sys.stdin", io.StringIO(json.dumps({"prompt": "p" * 60}))),
        patch(_RECALL, AsyncMock(return_value="## Relevant memory\n\n- fakt")),
        patch(_ORCHESTRATOR, AsyncMock(return_value=_REASONING_BLOCK)),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 0
    ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "## Relevant memory" in ctx
    assert "## Reasoning strategies" in ctx


def test_main_recall_failure_still_delivers_strategies(capsys: pytest.CaptureFixture[str]) -> None:
    """A broken recall must not take the reasoning block down with it."""
    with (
        patch("sys.stdin", io.StringIO(json.dumps({"prompt": "p" * 60}))),
        patch(_RECALL, AsyncMock(side_effect=RuntimeError("brain down"))),
        patch(_ORCHESTRATOR, AsyncMock(return_value=_REASONING_BLOCK)),
        pytest.raises(SystemExit) as exc,
    ):
        main()

    assert exc.value.code == 0
    ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "## Reasoning strategies" in ctx


def test_the_gap_this_closes_prompt_never_reached_memory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reprodukcja LUKI, nie tylko nowego API.

    Przed tą zmianą hook wołał wyłącznie orkiestrator strategii — treść promptu
    nie docierała do pamięci ŻADNĄ drogą. Ten test tego pilnuje od strony
    zachowania: prompt musi trafić do zapytania. Na kodzie sprzed zmiany pada
    merytorycznie (zero wywołań pipeline'u), a nie na braku symbolu.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    result = MagicMock()
    result.context = "- zapamiętany fakt"
    pipeline = MagicMock()
    pipeline.query = AsyncMock(return_value=result)
    storage = AsyncMock()
    storage.brain_id = "b1"
    storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())
    prompt = "o czym rozmawialiśmy przy strażniku rekoncyliacji?"

    with (
        patch("sys.stdin", io.StringIO(json.dumps({"prompt": prompt}))),
        patch("surreal_memory.unified_config.get_config") as gc,
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
        patch(_ORCHESTRATOR, AsyncMock(return_value="")),
    ):
        # Kaczo-typowany config, NIE import nowej klasy — dzięki temu na kodzie
        # sprzed zmiany test pada na braku ZAPYTANIA, a nie na braku symbolu.
        gc.return_value.prompt_recall = SimpleNamespace(
            enabled=True, min_prompt_chars=40, max_tokens=600, timeout_seconds=5.0
        )
        gc.return_value.current_brain = "b1"
        with pytest.raises(SystemExit):
            main()

    # TO jest asercja o zachowaniu: prompt dotarł do pamięci jako ZAPYTANIE.
    assert pipeline.query.await_count == 1, "prompt nigdy nie trafił do pamięci"
    assert pipeline.query.await_args.kwargs["query"] == prompt
    payload = json.loads(capsys.readouterr().out)
    assert "zapamiętany fakt" in payload["hookSpecificOutput"]["additionalContext"]


@pytest.mark.asyncio
async def test_max_tokens_is_a_ceiling_not_a_suggestion() -> None:
    """Zmierzone: ReflexPipeline traktuje max_tokens jako CEL i przestrzeliwuje
    o ~70% (600 -> ~1009 tokenów). Pole nazwane max_tokens musi ciąć, inaczej
    obiecuje limit, którego nie ma — a to leci przy KAŻDEJ turze."""
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    pipeline = await _pipeline_returning("x" * 10_000)
    storage = AsyncMock()
    storage.brain_id = "b1"
    storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())

    with (
        patch("surreal_memory.unified_config.get_config") as gc,
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
    ):
        gc.return_value.prompt_recall = _cfg(max_tokens=100)
        gc.return_value.current_brain = "b1"
        out = await get_prompt_recall({"prompt": "p" * 100})

    assert len(out) < 700, "sufit nie zadziałał — wstrzyk zalałby kontekst"
    assert "przycięte do 100 tokenów" in out
