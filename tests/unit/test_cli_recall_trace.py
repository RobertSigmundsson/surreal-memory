"""CLI recall writes a retrieval trace (tor "cli") without changing the recall itself.

The real Typer app (CliRunner) runs the real ``ReflexPipeline`` on a real ``InMemoryStorage`` filled
by ``MemoryEncoder``; only the storage factory and the unified config are patched (no live DB).
Every test sets the identity environment explicitly — the test process may itself run inside a
Claude Code session that exports ``CLAUDE_CODE_*``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from surreal_memory.cli import recall_trace as rt
from surreal_memory.cli.main import app
from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine import recall_api
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.recall_http import RecallIn
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import TraceConfig, UnifiedConfig

runner = CliRunner()
MEM = "surreal_memory.cli.commands.memory"
SHORT = "surreal_memory.cli.commands.shortcuts"
SESSION = "0f1e2d3c-4b5a-6978-8796-a5b4c3d2e1f0"
MEMORIES = (
    "Met with Alice at the coffee shop to discuss API design",
    "Alice suggested adding rate limiting to the API",
    "Completed the authentication module for the API gateway",
)
QUERY = "What did Alice suggest for the API?"


def _build(storage: InMemoryStorage | None = None, texts: tuple[str, ...] = MEMORIES) -> Any:
    async def _go() -> InMemoryStorage:
        s = storage or InMemoryStorage()
        cfg = BrainConfig(activation_threshold=0.1, max_spread_hops=4)
        brain = Brain.create(name="t", config=cfg)
        await s.save_brain(brain)
        s.set_brain(brain.id)
        enc = MemoryEncoder(s, cfg)
        for i, text in enumerate(texts):
            await enc.encode(text, timestamp=datetime(2024, 2, 3, 15, i))
        return s

    return asyncio.run(_go())


def _traces(storage: InMemoryStorage) -> list[Any]:
    return list(storage._retrieval_traces[storage.brain_id or ""])


def _ucfg(tmp_path: Path, enabled: bool = True, sample_rate: float = 1.0) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="t",
        trace=TraceConfig(enabled=enabled, sample_rate=sample_rate),
    )


def _env(monkeypatch: pytest.MonkeyPatch, **env: str | None) -> None:
    for name in (rt.ENV_AGENT_ID, rt.ENV_ENTRYPOINT, rt.ENV_SESSION_ID):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        if value is not None:
            monkeypatch.setenv(name, value)


def _invoke(
    storage: InMemoryStorage,
    ucfg: UnifiedConfig,
    argv: list[str],
    module: str = MEM,
) -> Any:
    with (
        patch(f"{module}.get_config", MagicMock()),
        patch(f"{module}.get_storage", new=AsyncMock(return_value=storage)),
        patch("surreal_memory.unified_config.get_config", return_value=ucfg),
    ):
        return runner.invoke(app, argv)


def _json(result: Any) -> dict[str, Any]:
    assert result.exit_code == 0, result.output
    out: dict[str, Any] = json.loads(result.stdout)
    return out


# ── T1-T3: identity ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("env", "agent_id", "source"),
    [
        (
            {"SMEM_AGENT_ID": "k3-test", "CLAUDE_CODE_ENTRYPOINT": "sdk-cli"},
            "k3-test",
            "SMEM_AGENT_ID",
        ),
        (
            {"SMEM_AGENT_ID": "", "CLAUDE_CODE_ENTRYPOINT": "claude-desktop"},
            "claude-code:claude-desktop",
            "CLAUDE_CODE_ENTRYPOINT",
        ),
        ({"CLAUDE_CODE_ENTRYPOINT": "sdk-cli"}, "claude-code:sdk-cli", "CLAUDE_CODE_ENTRYPOINT"),
        ({}, "cli", "default"),
    ],
)
def test_identity_precedence(env: dict[str, str], agent_id: str, source: str) -> None:
    ident = rt.resolve_cli_identity({**env, rt.ENV_SESSION_ID: SESSION})
    assert (ident.agent_id, ident.agent_source, ident.session_id) == (agent_id, source, SESSION)
    assert rt.resolve_cli_identity(env).session_id is None


@pytest.mark.parametrize(
    ("env", "variable"),
    [
        ({"SMEM_AGENT_ID": "zly agent"}, "SMEM_AGENT_ID"),
        ({"SMEM_AGENT_ID": "zażółć"}, "SMEM_AGENT_ID"),
        ({"SMEM_AGENT_ID": "x" * 121}, "SMEM_AGENT_ID"),
        ({"CLAUDE_CODE_ENTRYPOINT": "bad entry"}, "CLAUDE_CODE_ENTRYPOINT"),
        ({"CLAUDE_CODE_ENTRYPOINT": "e" * 109}, "CLAUDE_CODE_ENTRYPOINT"),
        ({"CLAUDE_CODE_SESSION_ID": "s" * 129}, "CLAUDE_CODE_SESSION_ID"),
        ({"CLAUDE_CODE_SESSION_ID": "sesja z spacja"}, "CLAUDE_CODE_SESSION_ID"),
    ],
)
def test_identity_rejects_invalid_without_echoing_the_value(
    env: dict[str, str], variable: str
) -> None:
    with pytest.raises(rt.CliIdentityError) as exc:
        rt.resolve_cli_identity(env)
    assert exc.value.variable == variable
    for value in env.values():
        assert value not in str(exc.value)


def test_identity_limits_are_inclusive() -> None:
    assert rt.resolve_cli_identity({"SMEM_AGENT_ID": "a" * 120}).agent_id == "a" * 120
    ident = rt.resolve_cli_identity({"CLAUDE_CODE_ENTRYPOINT": "e" * 108})
    assert len(ident.agent_id) == 120
    assert rt.resolve_cli_identity({"CLAUDE_CODE_SESSION_ID": "s" * 128}).session_id == "s" * 128


@pytest.mark.parametrize(
    "sample",
    [
        "agent:s_p_1",
        "claude-code:sdk-cli",
        "a.b@c-d",
        "x" * 120,
        "x" * 121,
        "zly agent",
        "ą",
        "a|b",
    ],
)
def test_agent_id_pattern_matches_shim(sample: str) -> None:
    ours = recall_api.AGENT_ID_PATTERN.match(sample) is not None
    try:
        RecallIn(query="q", agent_id=sample, tor="http:t")
        shim = True
    except ValidationError:
        shim = False
    assert ours == shim


@pytest.mark.parametrize("sample", ["s1", "http:test|s1", "s" * 128, "s" * 129, "z spacja", "ą"])
def test_session_id_pattern_matches_shim(sample: str) -> None:
    ours = recall_api.SESSION_ID_PATTERN.match(sample) is not None
    try:
        RecallIn(query="q", agent_id="a", tor="http:t", session_id=sample)
        shim = True
    except ValidationError:
        shim = False
    assert ours == shim


# ── T4: flag / config gate ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("flag", "enabled", "rate", "draw", "want"),
    [
        (None, True, 1.0, 0.99, True),
        (False, True, 1.0, 0.0, False),
        (None, False, 1.0, 0.0, False),
        (True, False, 1.0, 0.99, True),
        (None, True, 0.5, 0.2, True),
        (None, True, 0.5, 0.7, False),
    ],
)
def test_trace_gate_truth_table(
    flag: bool | None, enabled: bool, rate: float, draw: float, want: bool
) -> None:
    cfg = TraceConfig(enabled=enabled, sample_rate=rate)
    assert rt.cli_trace_wanted(cfg, flag, draw=lambda: draw) is want


# ── T5-T9, T11-T13: the CLI command ──────────────────────────────────────────────────────────


def test_cli_recall_writes_one_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, CLAUDE_CODE_ENTRYPOINT="claude-desktop", CLAUDE_CODE_SESSION_ID=SESSION)
    storage = _build()
    out = _json(_invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"]))
    traces = _traces(storage)
    assert len(traces) == 1
    t = traces[0]
    assert (t.tor, t.agent_id, t.session_id) == ("cli", "claude-code:claude-desktop", SESSION)
    assert out["trace_status"] == "sync" and out["trace_id"] == t.id
    assert list(t.fiber_ids) == list(out["fibers_matched"])[:10]
    assert out["fibers_matched"], "positive control: the query must match something"


def test_cli_recall_no_trace_writes_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, CLAUDE_CODE_ENTRYPOINT="claude-desktop")
    storage = _build()
    out = _json(_invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json", "--no-trace"]))
    assert _traces(storage) == []
    assert out["trace_status"] == "disabled" and "trace_id" not in out


def test_cli_recall_config_off_and_force(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    storage = _build()
    off = _json(_invoke(storage, _ucfg(tmp_path, enabled=False), ["recall", QUERY, "--json"]))
    assert _traces(storage) == [] and off["trace_status"] == "off"
    forced = _json(
        _invoke(storage, _ucfg(tmp_path, enabled=False), ["recall", QUERY, "--json", "--trace"])
    )
    assert len(_traces(storage)) == 1 and forced["trace_status"] == "sync"
    assert _traces(storage)[0].agent_id == "cli"


class _FailingTraceStorage(InMemoryStorage):
    async def add_retrieval_trace(self, trace: Any) -> str:
        raise RuntimeError("trace table unavailable")


def test_trace_failure_is_visible_and_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch)
    storage = _build(_FailingTraceStorage())
    res = _invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"])
    out = _json(res)
    assert out["trace_status"] == "sync_error" and out["trace_error"]
    assert out["fibers_matched"]
    assert f"{rt.MARKER} tor=cli status=sync_error" in res.stderr


def test_invalid_identity_is_visible_and_writes_no_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, SMEM_AGENT_ID="zly agent", CLAUDE_CODE_ENTRYPOINT="claude-desktop")
    storage = _build()
    res = _invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"])
    out = _json(res)
    assert _traces(storage) == []
    assert out["trace_status"] == "identity_error"
    assert "SMEM_AGENT_ID" in res.stderr and "zly agent" not in res.stderr + res.stdout


def test_pipeline_call_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch, CLAUDE_CODE_ENTRYPOINT="sdk-cli", CLAUDE_CODE_SESSION_ID=SESSION)
    seen: list[dict[str, Any]] = []
    orig = ReflexPipeline.query

    async def spy(self: ReflexPipeline, *a: Any, **kw: Any) -> Any:
        seen.append({"args": a, "kw": kw})
        return await orig(self, *a, **kw)

    monkeypatch.setattr(ReflexPipeline, "query", spy)
    for extra in ([], ["--trace"], ["--no-trace"]):
        _json(_invoke(_build(), _ucfg(tmp_path), ["recall", QUERY, "--json", *extra]))
    assert len(seen) == 3
    for call in seen:
        assert call["args"] == ()
        assert set(call["kw"]) == {"query", "depth", "max_tokens", "reference_time"}
        assert call["kw"]["query"] == QUERY


def test_trace_does_not_touch_the_printed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trace is written from the pipeline result without mutating it.

    Two in-process recalls are not a fair same-input pair (engine state is process-global and
    recall mutates storage), so this measures the mechanism; the effect on real, per-process CLI
    calls is K2 (0 rank differences before/after on fresh brain copies).
    """
    _env(monkeypatch, CLAUDE_CODE_ENTRYPOINT="claude-desktop")
    import surreal_memory.cli.commands.memory as mem

    orig = rt.persist_cli_trace
    seen: list[tuple[str, str]] = []

    def _snap(res: Any) -> str:
        return repr((res.context, res.confidence, res.fibers_matched, res.neurons_activated))

    async def spy(storage: Any, result: Any, **kw: Any) -> Any:
        before = _snap(result)
        out = await orig(storage, result, **kw)
        seen.append((before, _snap(result)))
        return out

    monkeypatch.setattr(mem, "persist_cli_trace", spy)
    res = _invoke(_build(), _ucfg(tmp_path), ["recall", QUERY])
    assert res.exit_code == 0, res.output
    assert len(seen) == 1 and seen[0][0] == seen[0][1]

    async def mutating(storage: Any, result: Any, **kw: Any) -> Any:
        out = await orig(storage, result, **kw)
        result.context = "zmieniony"
        return out

    seen.clear()
    monkeypatch.setattr(mem, "persist_cli_trace", mutating)

    async def spy2(storage: Any, result: Any, **kw: Any) -> Any:
        before = _snap(result)
        out = await mutating(storage, result, **kw)
        seen.append((before, _snap(result)))
        return out

    monkeypatch.setattr(mem, "persist_cli_trace", spy2)
    _invoke(_build(), _ucfg(tmp_path), ["recall", QUERY])
    assert seen[0][0] != seen[0][1], "control: the detector must see a mutation"


def test_json_keys_additive_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    out = _json(_invoke(_build(), _ucfg(tmp_path), ["recall", QUERY, "--json"]))
    keys = list(out)
    base = [k for k in keys if k not in ("trace_status", "trace_id", "trace_error")]
    assert keys[: len(base)] == base, "trace keys are appended after every existing key"
    assert base[:6] == [
        "answer",
        "confidence",
        "depth_used",
        "neurons_activated",
        "fibers_matched",
        "latency_ms",
    ]
    assert set(keys) - set(base) == {"trace_status", "trace_id"}


def test_below_threshold_is_still_traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _env(monkeypatch)
    storage = _build()
    out = _json(_invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json", "-c", "1.1"]))
    assert out["below_threshold"] is True and out["trace_status"] == "sync"
    assert _traces(storage)[0].filters.get("min_confidence") == 1.1
    storage2 = _build()
    _json(_invoke(storage2, _ucfg(tmp_path), ["recall", QUERY, "--json"]))
    assert "min_confidence" not in _traces(storage2)[0].filters


# ── T14, T17: persist_cli_trace directly ─────────────────────────────────────────────────────


def _pipeline_result(storage: InMemoryStorage, query: str) -> tuple[Any, Any]:
    async def _go() -> tuple[Any, Any]:
        brain = await storage.get_brain(storage.brain_id or "")
        assert brain is not None
        res = await ReflexPipeline(storage, brain.config).query(query=query, max_tokens=500)
        return brain, res

    return asyncio.run(_go())


def test_trace_query_is_sanitized_copy(tmp_path: Path) -> None:
    storage = _build()
    brain, res = _pipeline_result(storage, QUERY)
    out = asyncio.run(
        rt.persist_cli_trace(
            storage,
            res,
            brain=brain,
            query="alice \ud800 api",
            depth=1,
            max_tokens=500,
            min_confidence=0.0,
            flag=None,
            env={},
            config=_ucfg(tmp_path),
        )
    )
    assert out.status == "sync"
    q = _traces(storage)[0].query
    assert "\ud800" not in q and "\ufffd" in q and q.startswith("alice ") and q.endswith(" api")


def test_config_without_trace_section_is_an_error_not_silence(tmp_path: Path) -> None:
    storage = _build()
    brain, res = _pipeline_result(storage, QUERY)
    out = asyncio.run(
        rt.persist_cli_trace(
            storage,
            res,
            brain=brain,
            query=QUERY,
            depth=1,
            max_tokens=500,
            min_confidence=0.0,
            flag=None,
            env={},
            config=SimpleNamespace(),
        )
    )
    assert (out.status, out.error) == ("sync_error", "brak-sekcji-trace")
    assert out.stderr_line() is not None and _traces(storage) == []


# ── T18: smem q ──────────────────────────────────────────────────────────────────────────────


def test_quick_recall_writes_trace_and_no_trace_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _env(monkeypatch, CLAUDE_CODE_ENTRYPOINT="sdk-cli")
    storage = _build()
    res = _invoke(storage, _ucfg(tmp_path), ["q", QUERY], module=SHORT)
    assert res.exit_code == 0, res.output
    assert [(t.tor, t.agent_id) for t in _traces(storage)] == [("cli", "claude-code:sdk-cli")]
    storage2 = _build()
    res2 = _invoke(storage2, _ucfg(tmp_path), ["q", QUERY, "--no-trace"], module=SHORT)
    assert res2.exit_code == 0 and _traces(storage2) == []
    assert "Relevant" in res.stdout and rt.MARKER not in res.stderr + res2.stderr
