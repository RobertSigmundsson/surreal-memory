"""C2 r2 / B-2: neuron decay writes ONLY ``activation_level`` and ``last_decayed``.

Decay used to write the whole neuron state it read at the start of the pass (a snapshot),
so a reinforce landing mid-pass was rolled back: ``last_activated`` went back in time and
``access_frequency`` dropped. The write must be a narrow merge of the two fields decay owns.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import timedelta
from unittest.mock import AsyncMock

from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow


async def test_reinforce_during_decay_pass_is_not_rolled_back() -> None:
    """Sonda S3 of checker A: reinforce between snapshot and write must survive."""
    now = utcnow()
    st = InMemoryStorage()
    st.set_brain("narrow-write")
    n = Neuron.create(type=NeuronType.ENTITY, content="x")
    await st.add_neuron(n)
    old = now - timedelta(days=20)
    await st.update_neuron_state(
        NeuronState(
            neuron_id=n.id,
            activation_level=1.0,
            access_frequency=3,
            last_activated=old,
            created_at=old,
        )
    )
    orig = st.get_all_neuron_states

    async def _snapshot_then_recall() -> list[NeuronState]:
        snap = await orig()
        cur = await st.get_neuron_state(n.id)
        assert cur is not None
        await st.update_neuron_state(
            dc_replace(cur, last_activated=now, access_frequency=4, activation_level=1.0)
        )
        return snap

    st.get_all_neuron_states = _snapshot_then_recall  # type: ignore[method-assign]
    await DecayManager(decay_rate=0.1, prune_threshold=0.0).apply_decay(st, reference_time=now)

    after = await st.get_neuron_state(n.id)
    assert after is not None
    assert after.last_activated == now  # reinforce stamp survived
    assert after.access_frequency == 4
    assert after.last_decayed == now
    assert after.activation_level < 1.0  # decay itself still applied


async def test_surreal_decay_write_is_a_two_field_merge() -> None:
    from surreal_memory.storage.surrealdb.store import SurrealDBStorage

    s = SurrealDBStorage()
    s._current_brain_id = "default"
    s._conn = AsyncMock()
    stamp = utcnow()
    await s.update_neuron_decay_fields("neuron:x", activation_level=0.25, last_decayed=stamp)
    assert s._conn.merge.await_count == 1
    payload = s._conn.merge.await_args.args[1]
    assert payload == {"activation_level": 0.25, "last_decayed": stamp}


async def test_decay_goes_through_the_narrow_writer_only() -> None:
    """The pass must not call the full-state writer for neurons at all."""
    now = utcnow()
    st = InMemoryStorage()
    st.set_brain("narrow-only")
    n = Neuron.create(type=NeuronType.ENTITY, content="y")
    await st.add_neuron(n)
    old = now - timedelta(days=10)
    await st.update_neuron_state(
        NeuronState(neuron_id=n.id, activation_level=1.0, last_activated=old, created_at=old)
    )
    full: list[NeuronState] = []
    orig_full = st.update_neuron_state

    async def _spy(state: NeuronState) -> None:
        full.append(state)
        await orig_full(state)

    st.update_neuron_state = _spy  # type: ignore[method-assign]
    await DecayManager(decay_rate=0.1, prune_threshold=0.0).apply_decay(st, reference_time=now)
    assert full == []
