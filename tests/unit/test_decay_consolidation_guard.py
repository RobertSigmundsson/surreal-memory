"""P5 (C2): a real decay pass refuses to run next to an incomplete consolidation run.

A decay pass rewrites ~95k synapses one by one; against the strict ``semantic_link``
source fence that turns a running consolidation into ``Changed`` (or ``Unavailable``
once the changefeed exceeds its row cap). Dry runs write nothing and stay allowed.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.engine.lifecycle import DecayManager, DecayRefusedError
from surreal_memory.storage.memory_store import InMemoryStorage


class _Storage(InMemoryStorage):
    """InMemoryStorage whose consolidation-run lookup is scripted."""

    def __init__(self, run_ids: list[str] | Exception) -> None:
        super().__init__()
        self.set_brain("decay-guard-test")
        self._run_ids = run_ids
        self.lookups: list[Any] = []
        self.writes = 0

    async def get_consolidation_run_ids(  # type: ignore[override]
        self, *, statuses: Any, brain_id: str | None = None
    ) -> list[str]:
        self.lookups.append(set(statuses))
        if isinstance(self._run_ids, Exception):
            raise self._run_ids
        return list(self._run_ids)

    async def update_synapse(self, synapse: Any) -> Any:
        self.writes += 1
        return await super().update_synapse(synapse)

    async def update_neuron_state(self, state: Any) -> None:
        self.writes += 1
        return await super().update_neuron_state(state)

    async def update_neuron_decay_fields(self, *a: Any, **k: Any) -> None:
        self.writes += 1
        return await super().update_neuron_decay_fields(*a, **k)


async def test_real_pass_refused_when_run_incomplete() -> None:
    storage = _Storage(["run-1"])
    with pytest.raises(DecayRefusedError):
        await DecayManager().apply_decay(storage)
    assert storage.writes == 0
    assert storage.lookups == [{"queued", "running", "paused", "failed"}]


async def test_dry_run_allowed_when_run_incomplete() -> None:
    storage = _Storage(["run-1"])
    report = await DecayManager().apply_decay(storage, dry_run=True)
    assert report is not None
    assert storage.lookups == []
    assert storage.writes == 0


async def test_real_pass_proceeds_when_no_run_incomplete() -> None:
    storage = _Storage([])
    report = await DecayManager().apply_decay(storage)
    assert report is not None
    assert len(storage.lookups) == 1


async def test_adapter_without_durable_state_is_not_blocked() -> None:
    """NotImplementedError = backend has no consolidation runs at all (in-memory)."""
    storage = _Storage(NotImplementedError())
    assert await DecayManager().apply_decay(storage) is not None


async def _seed_eligible(storage: _Storage) -> None:
    """One idle neuron state and one idle synapse: a pass that gets past the guard WRITES."""
    from dataclasses import replace as dc_replace
    from datetime import timedelta

    from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
    from surreal_memory.core.synapse import Synapse, SynapseType
    from surreal_memory.utils.timeutils import utcnow

    old = utcnow() - timedelta(days=30)
    a = Neuron.create(type=NeuronType.ENTITY, content="a")
    b = Neuron.create(type=NeuronType.ENTITY, content="b")
    await storage.add_neuron(a)
    await storage.add_neuron(b)
    await InMemoryStorage.update_neuron_state(
        storage,
        NeuronState(neuron_id=a.id, activation_level=1.0, last_activated=old, created_at=old),
    )
    syn = Synapse.create(source_id=a.id, target_id=b.id, type=SynapseType.RELATED_TO)
    await storage.add_synapse(dc_replace(syn, last_activated=old))


async def test_refusal_leaves_zero_writes_even_with_eligible_rows() -> None:
    """Kills the 'check moved behind the neuron loop' mutant (checker A, M9).

    The earlier zero-write assertion ran on an empty storage, where it is vacuous.
    """
    storage = _Storage(["run-1"])
    await _seed_eligible(storage)
    with pytest.raises(DecayRefusedError):
        await DecayManager(min_age_days=0.0).apply_decay(storage)
    assert storage.writes == 0
    # control: with the guard satisfied the very same data IS written
    clear = _Storage([])
    await _seed_eligible(clear)
    await DecayManager(min_age_days=0.0).apply_decay(clear)
    assert clear.writes >= 2


async def test_not_implemented_is_refused_for_non_local_backends() -> None:
    """Only the explicit local list may skip the guard on NotImplementedError."""
    from unittest.mock import AsyncMock

    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    surreal = SurrealDBStorage()
    surreal._current_brain_id = "x"
    surreal._conn = AsyncMock()
    surreal.get_consolidation_run_ids = AsyncMock(side_effect=NotImplementedError())  # type: ignore[method-assign]
    surreal.get_brain_strict = AsyncMock(return_value=None)  # type: ignore[method-assign]
    surreal.get_pinned_neuron_ids = AsyncMock(return_value=set())  # type: ignore[method-assign]
    with pytest.raises(DecayRefusedError):
        await DecayManager().apply_decay(surreal)
    assert surreal._conn.merge.await_count == 0

    unknown = AsyncMock()
    unknown.get_brain_strict = AsyncMock(return_value=None)
    unknown.get_consolidation_run_ids = AsyncMock(side_effect=NotImplementedError())
    with pytest.raises(DecayRefusedError):
        await DecayManager().apply_decay(unknown)


async def test_lookup_failure_is_fail_closed() -> None:
    """Any other failure means we cannot prove the coast is clear: refuse."""
    storage = _Storage(RuntimeError("db down"))
    with pytest.raises(DecayRefusedError):
        await DecayManager().apply_decay(storage)
    assert storage.writes == 0


def test_cli_decay_exits_nonzero_when_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """`smem decay` turns the refusal into a visible failure, not a silent rc 0."""
    import typer

    from surreal_memory.cli.commands import tools

    storage = _Storage(["run-1"])

    async def _fake_storage(*_a: Any, **_k: Any) -> _Storage:
        return storage

    monkeypatch.setattr(tools, "get_storage", _fake_storage)
    monkeypatch.setattr(tools, "resolve_brain", lambda *_a, **_k: "decay-guard-test")
    monkeypatch.setattr(tools, "run_async", lambda coro: __import__("asyncio").run(coro))
    with pytest.raises(typer.Exit) as excinfo:
        tools.decay(
            brain=None,
            dry_run=False,
            prune_threshold=0.01,
            not_before=None,
            set_not_before=None,
        )
    assert excinfo.value.exit_code != 0
    assert storage.writes == 0
