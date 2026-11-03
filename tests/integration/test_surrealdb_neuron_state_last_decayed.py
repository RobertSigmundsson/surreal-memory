"""Live SurrealDB: the SCHEMAFULL ``neuron_state`` accepts and returns ``last_decayed``,
and a decay pass leaves ``last_activated`` alone (C2/P2). Loopback test server only."""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

TEST_SURREALDB_URL = os.getenv("SMEM_TEST_SURREALDB_URL")
TEST_AUTH = ("root", "root")


def _is_loopback_test_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme in {"http", "https", "ws", "wss"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
        )
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_test_url(TEST_SURREALDB_URL),
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL with port",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_ci",
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="last-decayed-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def test_schemafull_accepts_last_decayed_and_decay_keeps_last_activated(store) -> None:
    now = utcnow()
    neuron = Neuron.create(type=NeuronType.ENTITY, content="decay-stamp-it")
    await store.add_neuron(neuron)
    stamp = now - timedelta(days=10)
    await store.update_neuron_state(
        NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            decay_rate=0.1,
            last_activated=stamp,
            created_at=stamp,
        )
    )

    report = await DecayManager(prune_threshold=0.0, min_age_days=1.0).apply_decay(
        store, reference_time=now
    )
    assert report.neurons_decayed >= 1

    state = await store.get_neuron_state(neuron.id)
    assert state is not None
    assert state.last_decayed is not None
    assert abs((state.last_decayed - now).total_seconds()) < 1
    assert state.last_activated is not None
    assert abs((state.last_activated - stamp).total_seconds()) < 1  # decay did not stamp it
    assert state.activation_level < 1.0


async def test_brain_epoch_round_trip_keeps_config_and_drives_decay(store) -> None:
    """P1 on the live backend: save_brain/get_brain carry ``decay_not_before`` without
    touching the brain config, and DecayManager reads it from there."""
    brain_id = store.brain_id
    brain = await store.get_brain(brain_id)
    assert brain is not None
    config_before = brain.config
    epoch = utcnow() - timedelta(days=2)
    brain.metadata["decay_not_before"] = epoch.isoformat()
    await store.save_brain(brain)

    check = await store.get_brain(brain_id)
    assert check is not None
    assert check.metadata["decay_not_before"] == epoch.isoformat()
    assert check.config == config_before

    now = utcnow()
    neuron = Neuron.create(type=NeuronType.ENTITY, content="epoch-it")
    await store.add_neuron(neuron)
    stamp = now - timedelta(days=30)
    await store.update_neuron_state(
        NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            decay_rate=0.1,
            last_activated=stamp,
            created_at=stamp,
        )
    )
    report = await DecayManager(prune_threshold=0.0, min_age_days=1.0).apply_decay(
        store, reference_time=now
    )
    assert report.not_before == epoch
    state = await store.get_neuron_state(neuron.id)
    assert state is not None
    # Charged 2 days (epoch), not 30: exp(-0.2) ~ 0.8187
    assert abs(state.activation_level - 0.8187) < 0.01
