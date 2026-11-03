"""C2 r2 / B-1: trial instruments of ``smem decay``: ``--reference-time`` and ``--rate``.

Without them KC-4 ("at ``reference_time = T0`` nothing is decayed") cannot be expressed
through the CLI. A real (writing) pass with either deviation needs the extra explicit
``--allow-reference-time``; dry runs need nothing.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import typer

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.storage.memory_store import InMemoryStorage

if TYPE_CHECKING:
    from collections.abc import Iterator

T0 = datetime(2026, 11, 3, 12, 0, 0)


class _Store(InMemoryStorage):
    def __init__(self) -> None:
        super().__init__()
        self.set_brain("trial")
        self.writes = 0

    async def update_synapse(self, synapse: Any) -> Any:
        self.writes += 1
        return await super().update_synapse(synapse)


@pytest.fixture
def storage() -> _Store:
    st = _Store()

    async def seed() -> None:
        await st.save_brain(Brain(id="trial", name="trial"))
        a = Neuron.create(type=NeuronType.ENTITY, content="a")
        b = Neuron.create(type=NeuronType.ENTITY, content="b")
        await st.add_neuron(a)
        await st.add_neuron(b)
        syn = Synapse.create(source_id=a.id, target_id=b.id, type=SynapseType.RELATED_TO)
        await st.add_synapse(dc_replace(syn, last_activated=T0 - timedelta(days=20), weight=0.5))

    asyncio.run(seed())
    return st


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    rec: list[dict[str, Any]] = []
    orig = DecayManager.apply_decay

    async def _spy(
        self: DecayManager, storage: Any, reference_time: Any = None, dry_run: bool = False
    ) -> Any:
        report = await orig(self, storage, reference_time=reference_time, dry_run=dry_run)
        rec.append(
            {"rate": self.decay_rate, "ref": reference_time, "dry": dry_run, "report": report}
        )
        return report

    monkeypatch.setattr(DecayManager, "apply_decay", _spy)
    return rec


def _call(monkeypatch: pytest.MonkeyPatch, storage: _Store, **kw: object) -> int | None:
    from surreal_memory.cli.commands import tools

    async def _fake_storage(*_a: object, **_k: object) -> _Store:
        return storage

    monkeypatch.setattr(tools, "get_storage", _fake_storage)
    monkeypatch.setattr(tools, "resolve_brain", lambda *_a, **_k: "trial")
    monkeypatch.setattr(tools, "run_async", lambda coro: asyncio.run(coro))
    args: dict[str, object] = {
        "brain": None,
        "dry_run": True,
        "prune_threshold": 0.01,
        "not_before": None,
        "set_not_before": None,
    }
    args.update(kw)
    try:
        tools.decay(**args)  # type: ignore[arg-type]
    except typer.Exit as exc:
        return exc.exit_code
    return None


def test_kc4_reference_time_at_epoch_decays_nothing(
    monkeypatch: pytest.MonkeyPatch, storage: _Store, calls: list[dict[str, Any]]
) -> None:
    rc = _call(
        monkeypatch,
        storage,
        not_before=T0.isoformat(),
        reference_time=T0.isoformat(),
        rate=0.005,
    )
    assert rc in (None, 0)
    rep = calls[0]["report"]
    assert rep.synapses_decayed == 0 and rep.synapses_skipped_bookmark == 1
    assert calls[0]["ref"] == T0 and calls[0]["rate"] == 0.005 and calls[0]["dry"] is True
    assert storage.writes == 0


def test_reference_time_after_window_decays_with_the_given_rate(
    monkeypatch: pytest.MonkeyPatch, storage: _Store, calls: list[dict[str, Any]]
) -> None:
    ref = T0 + timedelta(days=7)
    _call(
        monkeypatch, storage, not_before=T0.isoformat(), reference_time=ref.isoformat(), rate=0.005
    )
    rep = calls[0]["report"]
    assert rep.synapses_decayed == 1
    assert rep.config_snapshot["decay_rate"] == 0.005


def test_real_pass_with_reference_time_or_rate_needs_the_extra_flag(
    monkeypatch: pytest.MonkeyPatch, storage: _Store, calls: list[dict[str, Any]]
) -> None:
    ref = (T0 + timedelta(days=7)).isoformat()
    assert _call(monkeypatch, storage, dry_run=False, reference_time=ref) == 2
    assert _call(monkeypatch, storage, dry_run=False, rate=0.005) == 2
    assert storage.writes == 0 and calls == []
    assert _call(
        monkeypatch,
        storage,
        dry_run=False,
        reference_time=ref,
        rate=0.005,
        allow_reference_time=True,
    ) in (None, 0)
    assert storage.writes == 1 and calls[0]["dry"] is False


def test_plain_real_pass_needs_no_extra_flag(
    monkeypatch: pytest.MonkeyPatch, storage: _Store, calls: list[dict[str, Any]]
) -> None:
    assert _call(monkeypatch, storage, dry_run=False) in (None, 0)
    assert calls[0]["ref"] is None


@pytest.mark.parametrize(
    "kw",
    [
        {"reference_time": "junk"},
        {"reference_time": ""},
        {"rate": 0.0},
        {"rate": -0.1},
        {"rate": float("nan")},
        {"rate": float("inf")},
    ],
)
def test_bad_trial_values_exit_2_without_touching_anything(
    monkeypatch: pytest.MonkeyPatch,
    storage: _Store,
    calls: list[dict[str, Any]],
    kw: dict[str, Any],
) -> None:
    assert _call(monkeypatch, storage, **kw) == 2
    assert calls == [] and storage.writes == 0


def test_trial_options_cannot_be_combined_with_set_not_before(
    monkeypatch: pytest.MonkeyPatch, storage: _Store
) -> None:
    rc = _call(monkeypatch, storage, dry_run=False, set_not_before="now", rate=0.005)
    assert rc == 2
    brain = asyncio.run(storage.get_brain("trial"))
    assert brain is not None and "decay_not_before" not in brain.metadata
