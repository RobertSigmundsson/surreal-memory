"""P2 (C2): neuron decay must not stamp ``last_activated``.

``last_activated`` is the freshness source of truth for heat/tier/dormancy. Decay used
to write ``last_activated=reference_time`` on every neuron it touched, turning a
maintenance pass into fake usage (7 892 such stamps from the 2026-09-19 run). Decay now
keeps its own bookmark, ``NeuronState.last_decayed``.
"""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta

import pytest

from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

RATE = 0.1


def _mgr(**kw: object) -> DecayManager:
    return DecayManager(decay_rate=RATE, prune_threshold=0.0, min_age_days=1.0, **kw)  # type: ignore[arg-type]


async def _neuron(
    storage: InMemoryStorage, now: datetime, *, idle_days: float, activated: bool = True
) -> str:
    n = Neuron.create(type=NeuronType.ENTITY, content=f"n-{idle_days}-{activated}")
    await storage.add_neuron(n)
    stamp = now - timedelta(days=idle_days)
    await storage.update_neuron_state(
        NeuronState(
            neuron_id=n.id,
            activation_level=1.0,
            decay_rate=RATE,
            last_activated=stamp if activated else None,
            created_at=stamp,
        )
    )
    return n.id


async def _state(storage: InMemoryStorage, nid: str) -> NeuronState:
    st = await storage.get_neuron_state(nid)
    assert st is not None
    return st


def _store() -> InMemoryStorage:
    s = InMemoryStorage()
    s.set_brain("neuron-stamp")
    return s


async def test_last_activated_is_left_alone_and_last_decayed_set() -> None:
    now = utcnow()
    storage = _store()
    nid = await _neuron(storage, now, idle_days=10)
    before = await _state(storage, nid)

    await _mgr().apply_decay(storage, reference_time=now)

    after = await _state(storage, nid)
    assert after.last_activated == before.last_activated
    assert after.last_decayed == now
    assert after.activation_level == pytest.approx(math.exp(-RATE * 10))


async def test_ten_daily_runs_equal_one_ten_day_run() -> None:
    now = utcnow()
    start = now - timedelta(days=10)
    daily, one_shot = _store(), _store()
    a = await _neuron(daily, now, idle_days=10)
    b = await _neuron(one_shot, now, idle_days=10)

    for day in range(1, 11):
        await _mgr().apply_decay(daily, reference_time=start + timedelta(days=day))
    await _mgr().apply_decay(one_shot, reference_time=now)

    assert (await _state(daily, a)).activation_level == pytest.approx(
        (await _state(one_shot, b)).activation_level
    )
    assert (await _state(daily, a)).activation_level == pytest.approx(math.exp(-RATE * 10))


async def test_legacy_row_first_run_charges_the_old_way() -> None:
    """No ``last_decayed`` yet: charge from last_activated (or created_at), as before."""
    now = utcnow()
    storage = _store()
    with_stamp = await _neuron(storage, now, idle_days=6)
    never_stamped = await _neuron(storage, now, idle_days=4, activated=False)

    await _mgr().apply_decay(storage, reference_time=now)

    assert (await _state(storage, with_stamp)).activation_level == pytest.approx(
        math.exp(-RATE * 6)
    )
    st = await _state(storage, never_stamped)
    assert st.activation_level == pytest.approx(math.exp(-RATE * 4))
    assert st.last_activated is None  # still "never activated"


async def test_use_after_last_decay_is_not_charged() -> None:
    now = utcnow()
    storage = _store()
    nid = await _neuron(storage, now, idle_days=10)
    await _mgr().apply_decay(storage, reference_time=now)
    level = (await _state(storage, nid)).activation_level

    used = await _state(storage, nid)
    await storage.update_neuron_state(dc_replace(used, last_activated=now + timedelta(days=2)))
    await _mgr().apply_decay(storage, reference_time=now + timedelta(days=3))

    assert (await _state(storage, nid)).activation_level == pytest.approx(level * math.exp(-RATE))


async def test_dry_run_writes_nothing() -> None:
    now = utcnow()
    storage = _store()
    nid = await _neuron(storage, now, idle_days=10)
    before = await _state(storage, nid)

    report = await _mgr().apply_decay(storage, reference_time=now, dry_run=True)

    assert report.neurons_decayed == 1
    assert await _state(storage, nid) == before


async def test_idle_gate_still_measures_real_idleness() -> None:
    """A neuron used 0.5 d ago is skipped even though it was decayed earlier."""
    now = utcnow()
    storage = _store()
    nid = await _neuron(storage, now, idle_days=0.5)
    report = await _mgr().apply_decay(storage, reference_time=now)
    assert report.neurons_decayed == 0
    assert (await _state(storage, nid)).activation_level == 1.0


def test_surreal_row_mapper_and_writers_carry_last_decayed() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from surreal_memory.storage.surrealdb.store import SurrealDBStorage, _row_to_neuron_state

    stamp = utcnow()
    row = {"neuron_id": "neuron:x", "last_decayed": stamp}
    assert _row_to_neuron_state(row).last_decayed == stamp
    assert _row_to_neuron_state({"neuron_id": "neuron:x"}).last_decayed is None

    s = SurrealDBStorage()
    s._current_brain_id = "default"
    s._query = AsyncMock(return_value=[])
    s._conn = AsyncMock()
    state = NeuronState(neuron_id="neuron:x", last_decayed=stamp)
    asyncio.run(s.update_neuron_states_batch([state]))
    assert s._query.await_args.kwargs["d0"]["last_decayed"] == stamp
    asyncio.run(s.update_neuron_state(state))
    assert s._conn.merge.await_args.args[1]["last_decayed"] == stamp
    # unset stays omitted (a write can never null the bookmark)
    asyncio.run(s.update_neuron_state(NeuronState(neuron_id="neuron:y")))
    assert "last_decayed" not in s._conn.merge.await_args.args[1]


def test_schema_declares_last_decayed() -> None:
    import re

    from surreal_memory.storage.surrealdb import schema

    assert re.search(
        r"DEFINE FIELD last_decayed\s+ON neuron_state TYPE option<datetime>;", schema.SCHEMA_SQL
    )
