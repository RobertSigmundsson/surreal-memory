"""Characterization of ``smem remember`` — the CLI write path, pinned BEFORE its logic moves to the engine.

Real Typer app (CliRunner) on a real ``InMemoryStorage``; only the storage factory is patched. These
tests describe today's behaviour (output, exit codes, what is stored) and must stay green, unchanged,
after the write logic is shared with the recall-http shim.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from surreal_memory.cli.main import app
from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.memory_types import MemoryType
from surreal_memory.safety.sensitive import check_sensitive_content, format_sensitive_warning
from surreal_memory.storage.memory_store import InMemoryStorage

runner = CliRunner()
MEM = "surreal_memory.cli.commands.memory"
TEXT = "Zespol wybral PostgreSQL jako baze dla modulu rozliczen"
SECRET_TEXT = "moje haslo password=Sup3rTajne!2026 do bazy"  # noqa: S105 — fixture for the sensitive-content gate


def _storage() -> InMemoryStorage:
    async def _go() -> InMemoryStorage:
        s = InMemoryStorage()
        brain = Brain.create(name="t", config=BrainConfig(activation_threshold=0.1))
        await s.save_brain(brain)
        s.set_brain(brain.id)
        return s

    return asyncio.run(_go())


def _invoke(storage: InMemoryStorage, argv: list[str], stdin: str | None = None) -> Any:
    with (
        patch(f"{MEM}.get_config", MagicMock()),
        patch(f"{MEM}.get_storage", new=AsyncMock(return_value=storage)),
    ):
        return runner.invoke(app, argv, input=stdin)


def _typed(storage: InMemoryStorage) -> list[Any]:
    return asyncio.run(storage.find_typed_memories(limit=100))


def test_remember_json_shape_and_stored_attribution() -> None:
    storage = _storage()
    res = _invoke(storage, ["remember", TEXT, "--type", "decision", "--tag", "k4", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert list(out) == [
        "message",
        "fiber_id",
        "memory_type",
        "priority",
        "neurons_created",
        "neurons_linked",
        "synapses_created",
    ]
    assert out["message"] == f"Remembered: {TEXT[:50]}..."
    assert (out["memory_type"], out["priority"]) == ("decision", "normal")
    # #252 (contract #196): an auto-classified DECISION carries no implicit expiry.
    assert out["neurons_created"] > 0 and "expires_in_days" not in out
    typed = _typed(storage)
    assert len(typed) == 1
    tm = typed[0]
    assert (tm.fiber_id, tm.memory_type, tm.source) == (
        out["fiber_id"],
        MemoryType.DECISION,
        "user_input",
    )
    assert tm.provenance.created_by == "user"
    assert set(tm.tags) == {"k4"}
    assert tm.expires_at is None


def test_remember_explicit_expiry_still_applies() -> None:
    storage = _storage()
    res = _invoke(storage, ["remember", TEXT, "--type", "decision", "--expires", "7", "--json"])
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["expires_in_days"] in (6, 7)
    assert _typed(storage)[0].expires_at is not None


def test_remember_fact_without_default_expiry_has_no_expiry_key() -> None:
    out = json.loads(_invoke(_storage(), ["remember", TEXT, "--type", "fact", "--json"]).stdout)
    assert "expires_in_days" not in out and out["memory_type"] == "fact"


def test_remember_explicit_priority_and_ephemeral() -> None:
    storage = _storage()
    out = json.loads(
        _invoke(
            storage, ["remember", TEXT, "--type", "fact", "-p", "8", "--ephemeral", "--json"]
        ).stdout
    )
    assert out["priority"] == "high" and out["ephemeral"] is True
    assert out["message"].endswith(" [ephemeral — auto-expires in 24h]")
    assert out["expires_in_days"] in (0, 1)


def test_sensitive_content_refused_without_flags() -> None:
    storage = _storage()
    res = _invoke(storage, ["remember", SECRET_TEXT, "--type", "fact"])
    assert res.exit_code == 1
    warning = format_sensitive_warning(check_sensitive_content(SECRET_TEXT, min_severity=2))
    assert warning.strip() in res.stdout
    assert _typed(storage) == []


def test_sensitive_content_redact_and_force() -> None:
    storage = _storage()
    red = _invoke(storage, ["remember", SECRET_TEXT, "--type", "fact", "--redact", "--json"])
    assert red.exit_code == 0, red.output
    assert "Redacted 1 sensitive item(s)" in red.stdout
    contents = [n.content for n in storage._neurons[storage.brain_id or ""].values()]
    assert contents and not any("Sup3rTajne" in c for c in contents)
    storage2 = _storage()
    forced = _invoke(storage2, ["remember", SECRET_TEXT, "--type", "fact", "--force", "--json"])
    assert forced.exit_code == 0, forced.output
    out = json.loads(forced.stdout)
    assert out["warnings"] == ["[!] Stored with 1 sensitive item(s) - consider using --redact"]
    forced_contents = [n.content for n in storage2._neurons[storage2.brain_id or ""].values()]
    assert any("Sup3rTajne" in c for c in forced_contents), "control: --force stores it verbatim"


def test_invalid_type_message_and_exit() -> None:
    storage = _storage()
    res = _invoke(storage, ["remember", TEXT, "--type", "nieznany"])
    assert res.exit_code == 1
    valid = ", ".join(t.value for t in MemoryType)
    assert f"Invalid memory type. Valid types: {valid}" in res.stdout
    assert _typed(storage) == []


def test_stdin_is_stripped_and_empty_is_refused() -> None:
    storage = _storage()
    out = json.loads(
        _invoke(
            storage, ["remember", "--stdin", "--type", "fact", "--json"], stdin=f"  {TEXT}\n\n"
        ).stdout
    )
    assert out["message"] == f"Remembered: {TEXT[:50]}..."
    empty = _invoke(_storage(), ["remember", "--stdin"], stdin="   \n")
    assert empty.exit_code == 1
    assert "content is required" in empty.stderr
