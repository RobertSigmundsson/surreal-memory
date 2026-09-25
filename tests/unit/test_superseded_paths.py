"""The superseded filter on every non-MCP recall path: ``smem recall``, ``smem q``, the function
behind ``POST /v1/recall-cli`` (``recall_like_cli``) and the ``UserPromptSubmit`` hook.

Real Typer app, real ``ReflexPipeline``, real ``InMemoryStorage`` filled by ``MemoryEncoder``; the
only patches are the storage factory and the unified config (no live DB). Each path is checked
three ways: the superseded fiber is gone (list and text), the escape hatch brings it back, and
the pipeline itself really returns it (positive control — otherwise "gone" proves nothing).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from surreal_memory.cli.main import app
from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from surreal_memory.engine import cli_recall_api
from surreal_memory.engine.cli_recall_api import TraceOutcome
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.engine.superseded_filter import DISABLE_ENV
from surreal_memory.hooks import user_prompt_submit as ups
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import PromptRecallConfig, TraceConfig, UnifiedConfig
from surreal_memory.utils.timeutils import utcnow

runner = CliRunner()
MEM = "surreal_memory.cli.commands.memory"
SHORT = "surreal_memory.cli.commands.shortcuts"
OLD = "Alice suggested adding rate limiting to the API"
MEMORIES = (
    "Met with Alice at the coffee shop to discuss API design",
    OLD,
    "Completed the authentication module for the API gateway",
)
QUERY = "What did Alice suggest for the API?"


def _build(*, supersede: bool) -> tuple[InMemoryStorage, str]:
    """Brain with the three memories; ``OLD``'s fiber gets a typed_memory, closed if ``supersede``."""

    async def _go() -> tuple[InMemoryStorage, str]:
        s = InMemoryStorage()
        cfg = BrainConfig(activation_threshold=0.1, max_spread_hops=4)
        brain = Brain.create(name="t", config=cfg)
        await s.save_brain(brain)
        s.set_brain(brain.id)
        enc = MemoryEncoder(s, cfg)
        for i, text in enumerate(MEMORIES):
            await enc.encode(text, timestamp=datetime(2024, 2, 3, 15, i))
        old_fiber = ""
        for fiber in await s.get_fibers(limit=100):
            anchor = await s.get_neuron(fiber.anchor_neuron_id)
            if anchor is not None and anchor.content == OLD:
                old_fiber = fiber.id
        assert old_fiber, "fixture: fiber of OLD not found"
        now = utcnow()
        await s.add_typed_memory(
            TypedMemory(
                fiber_id=old_fiber,
                memory_type=MemoryType.FACT,
                priority=Priority.from_int(5),
                provenance=Provenance(source="test"),
                created_at=now - timedelta(days=30),
                valid_from=now - timedelta(days=30),
                valid_until=now if supersede else None,
            )
        )
        return s, old_fiber

    return asyncio.run(_go())


def _ucfg(tmp_path: Path) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="t",
        trace=TraceConfig(enabled=False),
        prompt_recall=PromptRecallConfig(enabled=True, min_prompt_chars=10, max_tokens=600),
    )


def _invoke(storage: InMemoryStorage, ucfg: UnifiedConfig, argv: list[str], module: str) -> Any:
    with (
        patch(f"{module}.get_config", MagicMock(return_value=ucfg)),
        patch(f"{module}.get_storage", new=AsyncMock(return_value=storage)),
        patch("surreal_memory.unified_config.get_config", return_value=ucfg),
    ):
        return runner.invoke(app, argv)


@pytest.fixture(autouse=True)
def _filter_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DISABLE_ENV, raising=False)
    for name in ("SMEM_AGENT_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)


def test_control_pipeline_returns_the_old_fiber() -> None:
    """Positive control: without any filter the pipeline DOES return OLD for this query."""
    storage, old_fiber = _build(supersede=True)

    async def _go() -> Any:
        brain = await storage.get_brain(storage.brain_id or "")
        return await ReflexPipeline(storage, brain.config).query(query=QUERY, max_tokens=500)

    res = asyncio.run(_go())
    assert old_fiber in res.fibers_matched
    assert OLD in res.context


# ── smem recall (CLI) ────────────────────────────────────────────────────────────────────────


def test_cli_recall_drops_superseded(tmp_path: Path) -> None:
    storage, old_fiber = _build(supersede=True)
    r = _invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"], MEM)
    assert r.exit_code == 0, r.output
    out = json.loads(r.stdout)
    assert old_fiber not in out["fibers_matched"]
    assert OLD not in out["answer"]
    assert out["superseded_excluded_count"] == 1
    assert out["fibers_matched"], "other fibers stay"


def test_cli_recall_escape_hatch_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    storage, old_fiber = _build(supersede=True)
    out = json.loads(_invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"], MEM).stdout)
    assert old_fiber in out["fibers_matched"]
    assert "superseded_excluded_count" not in out


def test_cli_recall_open_fact_unchanged(tmp_path: Path) -> None:
    storage, old_fiber = _build(supersede=False)
    out = json.loads(_invoke(storage, _ucfg(tmp_path), ["recall", QUERY, "--json"], MEM).stdout)
    assert old_fiber in out["fibers_matched"]
    assert "superseded_excluded_count" not in out


# ── smem q ───────────────────────────────────────────────────────────────────────────────────


def test_quick_recall_drops_superseded(tmp_path: Path) -> None:
    storage, _ = _build(supersede=True)
    r = _invoke(storage, _ucfg(tmp_path), ["q", QUERY, "--no-trace"], SHORT)
    assert r.exit_code == 0, r.output
    assert OLD not in r.stdout
    assert "Alice" in r.stdout


def test_quick_recall_escape_hatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    storage, _ = _build(supersede=True)
    r = _invoke(storage, _ucfg(tmp_path), ["q", QUERY, "--no-trace"], SHORT)
    assert OLD in r.stdout


# ── recall_like_cli = POST /v1/recall-cli ────────────────────────────────────────────────────


def _recall_like_cli(storage: InMemoryStorage, ucfg: UnifiedConfig) -> tuple[dict[str, Any], Any]:
    seen: list[Any] = []

    async def _po(result: Any, depth: int) -> TraceOutcome:
        seen.append(result)
        return TraceOutcome("disabled")

    async def _go() -> dict[str, Any]:
        brain = await storage.get_brain(storage.brain_id or "")
        with patch("surreal_memory.unified_config.get_config", return_value=ucfg):
            body, _ = await cli_recall_api.recall_like_cli(
                storage,
                brain,
                query=QUERY,
                depth=None,
                max_tokens=500,
                min_confidence=0.0,
                show_routing=False,
                show_age=False,
                po_zapytaniu=_po,
            )
        return body

    return asyncio.run(_go()), seen


def test_recall_cli_function_drops_superseded_and_traces_filtered_list(tmp_path: Path) -> None:
    storage, old_fiber = _build(supersede=True)
    body, seen = _recall_like_cli(storage, _ucfg(tmp_path))
    assert old_fiber not in body["fibers_matched"] and OLD not in body["answer"]
    assert body["superseded_excluded_count"] == 1
    assert len(seen) == 1 and old_fiber not in seen[0].fibers_matched  # trace = filtered list


def test_recall_cli_function_escape_hatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    storage, old_fiber = _build(supersede=True)
    body, _ = _recall_like_cli(storage, _ucfg(tmp_path))
    assert old_fiber in body["fibers_matched"]


# ── UserPromptSubmit hook ────────────────────────────────────────────────────────────────────


def _hook(storage: InMemoryStorage, ucfg: UnifiedConfig) -> str:
    async def _go() -> str:
        with (
            patch("surreal_memory.unified_config.get_config", return_value=ucfg),
            patch(
                "surreal_memory.unified_config.get_shared_storage",
                AsyncMock(return_value=storage),
            ),
            patch.object(ups, "_persist_hook_trace", AsyncMock(return_value=None)),
        ):
            return await ups.get_prompt_recall({"prompt": QUERY, "session_id": "s-1"})

    return asyncio.run(_go())


def test_hook_context_has_no_superseded_text(tmp_path: Path) -> None:
    storage, _ = _build(supersede=True)
    ctx = _hook(storage, _ucfg(tmp_path))
    assert ctx.startswith("## Relevant memory")
    assert OLD not in ctx
    assert "Alice" in ctx


def test_hook_escape_hatch_restores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    storage, _ = _build(supersede=True)
    assert OLD in _hook(storage, _ucfg(tmp_path))
