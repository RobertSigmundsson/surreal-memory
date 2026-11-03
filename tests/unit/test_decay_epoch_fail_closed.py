"""C2 r2 / B-3: the decay epoch is read fail-closed.

``SurrealDBStorage.get_brain`` swallows every exception and returns ``None`` (store.py,
``except Exception: pass``), which ``_resolve_epoch`` used to read as "no epoch" and so
charged the whole life of every row. The epoch now has an explicit read path that tells
"no key" from "read failed", a ``--require-epoch`` guard, and the summary always prints
the epoch line (``none`` included) so the operator sees it before trusting a run.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.lifecycle import DecayEpochError, DecayManager, DecayReport
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


class _Counting(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.set_brain("b3")
        self.writes = 0

    async def update_synapse(self, synapse: Any) -> Any:
        self.writes += 1
        return await super().update_synapse(synapse)

    async def update_neuron_decay_fields(self, *a: Any, **k: Any) -> None:
        self.writes += 1
        return await super().update_neuron_decay_fields(*a, **k)


async def _seed(storage: InMemoryStorage) -> None:
    a = Neuron.create(type=NeuronType.ENTITY, content="a")
    b = Neuron.create(type=NeuronType.ENTITY, content="b")
    await storage.add_neuron(a)
    await storage.add_neuron(b)
    s = Synapse.create(source_id=a.id, target_id=b.id, type=SynapseType.RELATED_TO)
    await storage.add_synapse(dc_replace(s, last_activated=utcnow() - timedelta(days=9)))


def _surreal_with_failing_reads() -> Any:
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    s = SurrealDBStorage()
    s._current_brain_id = "uruboros"
    s._conn = AsyncMock()
    s._query = AsyncMock(side_effect=RuntimeError("brain row read failed"))
    return s


async def test_surreal_get_brain_still_swallows_but_strict_read_raises() -> None:
    s = _surreal_with_failing_reads()
    assert await s.get_brain("uruboros") is None  # legacy contract untouched
    with pytest.raises(RuntimeError):
        await s.get_brain_strict("uruboros")


async def test_read_error_is_refusal_not_no_epoch() -> None:
    """Checker A probe S2, inverted: a failing brain read must refuse, in dry-run too."""
    s = _surreal_with_failing_reads()
    with pytest.raises(DecayEpochError):
        await DecayManager()._resolve_epoch(s)
    for dry in (True, False):
        with pytest.raises(DecayEpochError):
            await DecayManager().apply_decay(s, dry_run=dry)
    assert s._conn.merge.await_count == 0


async def test_read_error_on_in_memory_style_backend_also_refuses() -> None:
    storage = _Counting()
    await _seed(storage)
    storage.get_brain_strict = AsyncMock(side_effect=OSError("boom"))  # type: ignore[method-assign]
    with pytest.raises(DecayEpochError):
        await DecayManager(min_age_days=0.0).apply_decay(storage)
    assert storage.writes == 0


async def test_missing_row_or_key_is_still_no_epoch_by_default() -> None:
    storage = _Counting()
    await _seed(storage)
    assert await DecayManager()._resolve_epoch(storage) is None  # no brain row
    await storage.save_brain(Brain(id="b3", name="b3"))
    assert await DecayManager()._resolve_epoch(storage) is None  # row, no key


async def test_require_epoch_refuses_when_none_and_writes_nothing() -> None:
    storage = _Counting()
    await _seed(storage)
    with pytest.raises(DecayEpochError):
        await DecayManager(min_age_days=0.0, require_epoch=True).apply_decay(storage)
    assert storage.writes == 0
    # no brain row at all is "no epoch" too
    await storage.save_brain(Brain(id="b3", name="b3"))
    with pytest.raises(DecayEpochError):
        await DecayManager(min_age_days=0.0, require_epoch=True).apply_decay(storage)
    assert storage.writes == 0


async def test_require_epoch_passes_with_stored_or_explicit_epoch() -> None:
    storage = _Counting()
    await _seed(storage)
    epoch = utcnow() - timedelta(days=3)
    await storage.save_brain(
        Brain(id="b3", name="b3", metadata={"decay_not_before": epoch.isoformat()})
    )
    rep = await DecayManager(min_age_days=0.0, require_epoch=True).apply_decay(storage)
    assert rep.not_before == epoch and rep.synapses_decayed == 1

    storage2 = _Counting()
    await _seed(storage2)
    rep2 = await DecayManager(min_age_days=0.0, require_epoch=True, not_before=epoch).apply_decay(
        storage2
    )
    assert rep2.not_before == epoch


def test_summary_always_prints_epoch_line() -> None:
    assert "Decay epoch (not_before): none" in DecayReport().summary()
    e = utcnow()
    assert f"Decay epoch (not_before): {e.isoformat()}" in DecayReport(not_before=e).summary()


class TestCliEpochGuards:
    def _wire(self, monkeypatch: pytest.MonkeyPatch, storage: _Counting, events: list[str]) -> None:
        from surreal_memory.cli.commands import tools

        async def _fake_storage(*_a: object, **_k: object) -> _Counting:
            return storage

        monkeypatch.setattr(tools, "get_storage", _fake_storage)
        monkeypatch.setattr(tools, "resolve_brain", lambda *_a, **_k: storage.brain_id)
        monkeypatch.setattr(tools, "run_async", lambda coro: asyncio.run(coro))
        real_echo = tools.typer.echo
        monkeypatch.setattr(
            tools.typer,
            "echo",
            lambda msg="", *a, **k: (events.append(f"echo:{msg}"), real_echo(msg, *a, **k))[1],
        )
        orig = storage.update_synapse

        async def _spy(syn: Any) -> Any:
            events.append("write")
            return await orig(syn)

        storage.update_synapse = _spy  # type: ignore[method-assign]

    def _call(self, **kw: object) -> None:
        from surreal_memory.cli.commands import tools

        args: dict[str, object] = {
            "brain": None,
            "dry_run": False,
            "prune_threshold": 0.01,
            "not_before": None,
            "set_not_before": None,
        }
        args.update(kw)
        tools.decay(**args)  # type: ignore[arg-type]

    def test_require_epoch_flag_refuses_rc3_without_epoch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typer

        storage = _Counting()
        asyncio.run(_seed(storage))
        asyncio.run(storage.save_brain(Brain(id="b3", name="b3")))
        events: list[str] = []
        self._wire(monkeypatch, storage, events)
        with pytest.raises(typer.Exit) as exc:
            self._call(require_epoch=True)
        assert exc.value.exit_code == 3
        assert "write" not in events

    def test_epoch_line_is_printed_before_any_write_even_when_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        storage = _Counting()
        asyncio.run(_seed(storage))
        asyncio.run(storage.save_brain(Brain(id="b3", name="b3")))
        events: list[str] = []
        self._wire(monkeypatch, storage, events)
        self._call()
        first_write = events.index("write")
        epoch_lines = [i for i, e in enumerate(events) if "Decay epoch (not_before): none" in e]
        assert epoch_lines and epoch_lines[0] < first_write

    def test_unreadable_brain_row_makes_cli_exit_3_before_any_write(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import typer

        storage = _Counting()
        asyncio.run(_seed(storage))
        storage.get_brain_strict = AsyncMock(side_effect=RuntimeError("down"))  # type: ignore[method-assign]
        events: list[str] = []
        self._wire(monkeypatch, storage, events)
        with pytest.raises(typer.Exit) as exc:
            self._call()
        assert exc.value.exit_code == 3
        assert "write" not in events
