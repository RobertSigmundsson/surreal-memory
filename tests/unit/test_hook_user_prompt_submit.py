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
from pathlib import Path
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


@pytest.mark.asyncio
async def test_the_pipelines_own_heading_is_not_stacked_under_ours() -> None:
    """Dwa nagłówki na każdym prompcie to szum, za który użytkownik płaci co turę."""
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    pipeline = await _pipeline_returning("## Relevant Memories\n\n- fakt o rclone")
    storage = AsyncMock()
    storage.brain_id = "b1"
    storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())

    with (
        patch("surreal_memory.unified_config.get_config") as gc,
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
        patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
    ):
        gc.return_value.prompt_recall = _cfg()
        gc.return_value.current_brain = "b1"
        out = await get_prompt_recall({"prompt": "p" * 100})

    assert out.lower().count("## relevant memor") == 1
    assert "- fakt o rclone" in out


# ---------------------------------------------------------------------------
# R1 (program jev-uzycie-wdrozenie): Claude Code system content is not a question.
# Measured 2026-09-24: 48 % of real Jev traffic came from task notifications and
# bash-mode input reaching this hook; 62 % of it judged irrelevant.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPTS = {
    "<task-notification": "<task-notification>\n<task-id>b1</task-id>\n<status>completed</status>\n"
    "<summary>Background command finished</summary>\n</task-notification> smem recall Jev brama klucz",
    "<bash-input": "<bash-input>git -C ~/repos/github/uruboros push origin master</bash-input>"
    "<bash-stdout>Everything up-to-date</bash-stdout><bash-stderr></bash-stderr> smem recall Jev",
    "<local-command": "<local-command-stdout>Goal set: program ukończony</local-command-stdout> smem Jev",
    "<command-": "<command-name>/goal</command-name><command-message>goal</command-message> smem Jev",
}


def _mocked_recall(prompt: str, **cfg: object):
    """Runs get_prompt_recall with storage/pipeline spies; returns (out, storage_mock, pipeline)."""
    import asyncio

    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    async def _run():
        pipeline = await _pipeline_returning("- trafienie z pamięci")
        storage = AsyncMock()
        storage.brain_id = "b1"
        storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())
        shared = AsyncMock(return_value=storage)
        with (
            patch("surreal_memory.unified_config.get_config") as gc,
            patch("surreal_memory.unified_config.get_shared_storage", shared),
            patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
        ):
            gc.return_value.prompt_recall = _cfg(min_prompt_chars=40, **cfg)
            gc.return_value.current_brain = "b1"
            out = await get_prompt_recall({"prompt": prompt, "session_id": "sesja-test"})
        return out, shared, pipeline

    return asyncio.run(_run())


def _skip_lines(tmp_path: Path) -> list[dict]:
    p = tmp_path / "prompt_recall_pominiete.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


@pytest.mark.parametrize("prefix", list(_SYSTEM_PROMPTS))
def test_system_content_skips_recall_and_is_recorded(
    prefix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, shared, pipeline = _mocked_recall(_SYSTEM_PROMPTS[prefix])
    assert out == ""
    shared.assert_not_awaited()  # no storage connection, hence no Jev call
    pipeline.query.assert_not_awaited()
    lines = _skip_lines(tmp_path)
    assert len(lines) == 1
    assert set(lines[0]) == {"ts", "powod", "prefiks", "dlugosc", "sesja"}
    assert lines[0]["prefiks"] == prefix and lines[0]["powod"] == "prefiks"
    assert lines[0]["sesja"] == "sesja-test"


def test_skip_record_never_contains_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    znacznik = "UNIKALNY-ZNACZNIK-TRESCI-7731"
    _mocked_recall("<bash-input>echo " + znacznik + "</bash-input>" + "x" * 60)
    _mocked_recall("\n   <task-notification>" + znacznik + "</task-notification>" + "y" * 60)
    tekst = (tmp_path / "prompt_recall_pominiete.jsonl").read_text(encoding="utf-8")
    assert znacznik not in tekst
    assert [r["prefiks"] for r in _skip_lines(tmp_path)] == ["<bash-input", "<task-notification"]


@pytest.mark.parametrize(
    "prompt",
    [
        "Robert pyta o wynik, a w środku cytuje <task-notification> z poprzedniej tury — co z tym?",
        "<div> jak ustawić wyrównanie w tym komponencie strony, bo się rozjeżdża na telefonie?",
        "task-notification: dlaczego przyszło powiadomienie o zadaniu w tle, które się nie udało?",
        "<Task-Notification> wielkie litery to nie format Claude Code, więc to człowiek coś wkleił",
    ],
)
def test_human_prompt_is_never_filtered(
    prompt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control (checker mandate a): only a prompt STARTING with a system tag is skipped."""
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, _shared, pipeline = _mocked_recall(prompt)
    assert pipeline.query.await_args.kwargs["query"] == prompt
    assert out.startswith("## Relevant memory")
    assert _skip_lines(tmp_path) == []


def test_empty_prefix_list_disables_the_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, _shared, pipeline = _mocked_recall(
        _SYSTEM_PROMPTS["<task-notification"], system_prefixes=[]
    )
    pipeline.query.assert_awaited()
    assert _skip_lines(tmp_path) == []


def test_invalid_prefixes_never_filter_human_prompts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty string would match every prompt — it must not survive config parsing."""
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    prompt = "zwykłe pytanie Roberta o stan programu jev-uzycie i klucze per kanał na bramie"
    out, _shared, pipeline = _mocked_recall(prompt, system_prefixes=["", "  ", "zwykle", "<ok"])
    pipeline.query.assert_awaited()
    assert out.startswith("## Relevant memory")


def test_unwritable_skip_log_still_skips_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plik = tmp_path / "to-jest-plik"
    plik.write_text("x")
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(plik))
    out, shared, _pipeline = _mocked_recall(_SYSTEM_PROMPTS["<bash-input"])
    assert out == ""
    shared.assert_not_awaited()
    assert "zapis nieudany" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# smem-cli-slad-i-pody-claude (V-GATE r1 F-3): the hook's recall leaves a trace
# (tor ``cli``) with its own agent_id, like every other host recall path.
# ---------------------------------------------------------------------------

_PERSIST = "surreal_memory.engine.recall_api.persist_trace"


def _recall_with_trace(
    prompt: str,
    *,
    env: dict[str, str],
    trace_enabled: bool = True,
    persist: AsyncMock | None = None,
    session: str = "sesja-hook-1",
):
    """Runs get_prompt_recall with a real-shaped [trace]; returns (out, persist_spy, result)."""
    import asyncio
    from types import SimpleNamespace

    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    spy = persist or AsyncMock(
        side_effect=lambda sink, *a, **k: sink.update({"trace_id": "retrieval_trace:t1"}) or "sync"
    )

    async def _run():
        pipeline = await _pipeline_returning("- trafienie z pamięci")
        result = pipeline.query.return_value
        storage = AsyncMock()
        storage.brain_id = "b1"
        storage.get_brain = AsyncMock(return_value=type("B", (), {"config": object()})())
        with (
            patch.dict("os.environ", env, clear=False),
            patch("surreal_memory.unified_config.get_config") as gc,
            patch(
                "surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)
            ),
            patch("surreal_memory.engine.retrieval.ReflexPipeline", return_value=pipeline),
            patch(_PERSIST, spy),
        ):
            gc.return_value.prompt_recall = _cfg(min_prompt_chars=40)
            gc.return_value.current_brain = "b1"
            gc.return_value.trace = SimpleNamespace(enabled=trace_enabled, sample_rate=1.0)
            out = await get_prompt_recall({"prompt": prompt, "session_id": session})
        return out, spy, result

    return asyncio.run(_run())


_HOOK_ENV = {"CLAUDE_CODE_ENTRYPOINT": "claude-desktop", "SMEM_AGENT_ID": ""}
_LONG = "co ustaliliśmy o śladzie recallu z hooka i torze cli w smem? " * 2


def test_hook_recall_writes_one_trace_tor_cli_with_hook_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, spy, _ = _recall_with_trace(_LONG, env=_HOOK_ENV)
    assert spy.await_count == 1
    kw = spy.await_args.kwargs
    assert kw["tor"] == "cli"
    assert kw["agent_id"] == "claude-code-hook:claude-desktop"
    assert kw["args"]["session_id"] == "sesja-hook-1"
    assert kw["query"] == _LONG.strip()
    assert "trafienie z pamięci" in out
    assert not (tmp_path / "prompt_recall_slad_bledy.jsonl").exists()


def test_hook_agent_id_is_apart_from_explicit_cli_recall() -> None:
    from surreal_memory.hooks.user_prompt_submit import resolve_hook_identity

    assert resolve_hook_identity({"session_id": "s"}, {"SMEM_AGENT_ID": "gpu-tryb"}) == (
        "gpu-tryb:hook",
        "s",
    )
    assert resolve_hook_identity({}, {"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"})[0] == (
        "claude-code-hook:sdk-cli"
    )
    assert resolve_hook_identity({}, {}) == ("cli-hook", None)
    # session from the hook input wins; the env session is the fallback
    assert resolve_hook_identity({}, {"CLAUDE_CODE_SESSION_ID": "env-s"})[1] == "env-s"


def test_hook_trace_off_when_trace_section_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, spy, _ = _recall_with_trace(_LONG, env=_HOOK_ENV, trace_enabled=False)
    assert spy.await_count == 0
    assert "trafienie z pamięci" in out


def test_hook_invalid_identity_writes_no_trace_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    out, spy, _ = _recall_with_trace(_LONG, env={**_HOOK_ENV, "SMEM_AGENT_ID": "zły agent"})
    assert spy.await_count == 0
    assert "SMEM-SLAD-BLAD tor=cli status=identity_error" in capsys.readouterr().err
    rows = [
        json.loads(x)
        for x in (tmp_path / "prompt_recall_slad_bledy.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 and "identity_error" in rows[0]["blad"]
    assert "zły agent" not in json.dumps(rows)  # the value never lands in the log
    assert "trafienie z pamięci" in out  # the prompt still gets its memory


def test_hook_trace_failure_never_blocks_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    boom = AsyncMock(side_effect=RuntimeError("baza padła"))
    out, spy, _ = _recall_with_trace(_LONG, env=_HOOK_ENV, persist=boom)
    assert spy.await_count == 1
    assert "status=sync_error" in capsys.readouterr().err
    assert (tmp_path / "prompt_recall_slad_bledy.jsonl").exists()
    assert "trafienie z pamięci" in out


def test_hook_trace_does_not_change_what_the_prompt_receives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    with_trace, _, r1 = _recall_with_trace(_LONG, env=_HOOK_ENV, trace_enabled=True)
    without, _, r2 = _recall_with_trace(_LONG, env=_HOOK_ENV, trace_enabled=False)
    assert with_trace == without
    assert r1.context == r2.context == "- trafienie z pamięci"


def test_system_content_and_short_prompts_write_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    for prompt in (_SYSTEM_PROMPTS["<task-notification"], "ok, dalej"):
        _, spy, _ = _recall_with_trace(prompt, env=_HOOK_ENV)
        assert spy.await_count == 0


def test_timed_out_recall_is_recorded_not_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No trace for a cancelled recall — so the durable log must say why."""
    import asyncio

    from surreal_memory.hooks.user_prompt_submit import _recall_within_timeout

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))

    async def _slow(_hook_input: dict) -> str:
        await asyncio.sleep(5)
        return "nigdy"

    with patch(_RECALL, _slow):
        out = asyncio.run(_recall_within_timeout({"session_id": "s-timeout"}, 0.05))
    assert out == ""
    rows = [
        json.loads(x)
        for x in (tmp_path / "prompt_recall_slad_bledy.jsonl").read_text().splitlines()
    ]
    assert rows == [
        {
            "ts": rows[0]["ts"],
            "blad": "SMEM-SLAD-BLAD tor=cli status=timeout powod=recall+slad>0.05s",
            "sesja": "s-timeout",
        }
    ]
