"""C2 r2: ``smem decay --set-not-before`` proves it changed nothing but the epoch key.

Read-back of the epoch alone (and a dataclass ``config`` comparison) cannot see a
``save_brain`` that rewrites other parts of the row. The CLI therefore takes a sha256 of
the raw ``brain`` row before and after, ignoring only ``metadata.decay_not_before``,
and exits non-zero on any difference. Also pins the read-back check (checker A, M8).
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import typer

from surreal_memory.core.brain import Brain
from surreal_memory.storage.memory_store import InMemoryStorage

if TYPE_CHECKING:
    import pytest


class _Store(InMemoryStorage):
    def __init__(self, mode: str = "ok") -> None:
        super().__init__()
        self.set_brain("fp")
        self._target_mode = mode
        self.mode = "ok"  # seeding saves are clean; armed by arm()

    def arm(self) -> _Store:
        self.mode = self._target_mode
        return self

    async def get_brain(self, brain_id: str) -> Brain | None:
        # A real database hands back a detached copy, never the stored object itself.
        found = await super().get_brain(brain_id)
        return None if found is None else deepcopy(found)

    async def save_brain(self, brain: Brain) -> None:
        if self.mode == "noop":  # reports success but persists nothing
            return
        if self.mode == "lossy":  # persists the epoch but also clobbers another key
            brain.metadata.pop("_maintenance_op_count", None)
        if self.mode == "config":
            brain = dc_replace(brain, config=type(brain.config)())  # defaults over stored config
        await super().save_brain(brain)


def _brain() -> Brain:
    return Brain(id="fp", name="fp", metadata={"_maintenance_op_count": 292})


def _run(monkeypatch: pytest.MonkeyPatch, storage: _Store) -> int | None:
    from surreal_memory.cli.commands import tools

    async def _fake_storage(*_a: object, **_k: object) -> _Store:
        return storage

    monkeypatch.setattr(tools, "get_storage", _fake_storage)
    monkeypatch.setattr(tools, "resolve_brain", lambda *_a, **_k: "fp")
    monkeypatch.setattr(tools, "run_async", lambda coro: asyncio.run(coro))
    try:
        tools.decay(
            brain=None,
            dry_run=False,
            prune_threshold=0.01,
            not_before=None,
            set_not_before="now",
        )
    except typer.Exit as exc:
        return exc.exit_code
    return None


def test_clean_save_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _Store()
    asyncio.run(st.save_brain(_brain()))
    assert _run(monkeypatch, st) in (None, 0)
    stored = asyncio.run(st.get_brain("fp"))
    assert stored is not None and "decay_not_before" in stored.metadata
    assert stored.metadata["_maintenance_op_count"] == 292


def test_save_that_persists_nothing_fails_the_readback(monkeypatch: pytest.MonkeyPatch) -> None:
    """M8: without the post-save read-back this would exit 0 with no epoch stored."""
    st = _Store("noop")
    asyncio.run(st.save_brain(_brain()))
    assert _run(monkeypatch, st.arm()) == 1


def test_save_that_clobbers_another_metadata_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _Store("lossy")
    asyncio.run(st.save_brain(_brain()))
    assert _run(monkeypatch, st.arm()) == 1


def test_save_that_rewrites_config_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    st = _Store("config")
    b = _brain()
    b = dc_replace(b, config=type(b.config)(decay_rate=0.123))
    asyncio.run(st.save_brain(b))
    assert _run(monkeypatch, st.arm()) == 1


def test_backend_that_cannot_fingerprint_is_refused_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    st = _Store()
    asyncio.run(st.save_brain(_brain()))
    st.brain_row_fingerprint = AsyncMock(side_effect=NotImplementedError())  # type: ignore[method-assign]
    assert _run(monkeypatch, st.arm()) == 1
    stored = asyncio.run(st.get_brain("fp"))
    assert stored is not None and "decay_not_before" not in stored.metadata


async def test_surreal_fingerprint_ignores_only_the_epoch_key() -> None:
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    s = SurrealDBStorage()
    s._current_brain_id = "uruboros"
    s._conn = AsyncMock()

    def _row(meta: dict[str, Any], cfg: Any = None) -> dict[str, Any]:
        return {
            "id": "brain:uruboros",
            "name": "uruboros",
            "config": cfg or {"decay_rate": 0.1},
            "metadata": meta,
        }

    async def fp(row: dict[str, Any]) -> str | None:
        s._query = AsyncMock(return_value=[row])  # type: ignore[method-assign]
        return await s.brain_row_fingerprint("uruboros", ignore_metadata_key="decay_not_before")

    base = await fp(_row({"_maintenance_op_count": 292}))
    assert base is not None and len(base) == 64
    assert await fp(_row({"_maintenance_op_count": 292, "decay_not_before": "2026-11-03"})) == base
    assert await fp(_row({"_maintenance_op_count": 291})) != base
    assert await fp(_row({"_maintenance_op_count": 292}, {"decay_rate": 0.2})) != base
    s._query = AsyncMock(return_value=[])  # type: ignore[method-assign]
    assert await s.brain_row_fingerprint("uruboros", ignore_metadata_key="x") is None
