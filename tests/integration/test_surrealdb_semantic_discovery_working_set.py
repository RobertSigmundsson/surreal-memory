"""Disposable local-DB integration tests for the semantic discovery working set (P4).

Real SurrealQL on SurrealDB 3.2+: the digests built from SDK rows must equal the digests built from
changefeed after-images, otherwise every recall would abort the run and the whole tolerance would
be dead letter. Loopback only - never run this against a shared or production endpoint.
"""

from __future__ import annotations

import ipaddress
import json
import os
import uuid
from dataclasses import replace
from datetime import UTC, timedelta
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.consolidation_progress import (
    CONSOLIDATION_ENGINE_VERSION,
    PROGRESS_FORMAT_VERSION,
)
from surreal_memory.engine.semantic_discovery import (
    SemanticSourceStaleError,
    discover_semantic_synapses,
    staged_source_state_ref,
    verify_staged_discovery_source,
)
from surreal_memory.storage.surrealdb.schema import SCHEMA_VERSION
from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.timeutils import utcnow

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_fence_it")


def _is_loopback_url(url: str | None) -> bool:
    if not url:
        return False
    try:
        hostname = urlparse(url).hostname
        if hostname == "localhost":
            return True
        return bool(hostname and ipaddress.ip_address(hostname).is_loopback)
    except ValueError:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _is_loopback_url(SURREALDB_URL),
        reason="requires explicit loopback SURREALDB_URL for a disposable SurrealDB >= 3.2",
    ),
]


@pytest_asyncio.fixture
async def store():
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="wset_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="semantic-working-set-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


_DIMENSIONS = 3072  # the vector column of the schema is typed to the production embedding width


def _vec(first: float, second: float) -> list[float]:
    return [first, second] + [0.0] * (_DIMENSIONS - 2)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )


async def _world(store, monkeypatch):
    """Six old candidates with vectors (one well connected) and a reference time after them."""
    monkeypatch.setattr(
        semantic_discovery_module, "_effective_embedding", lambda _c: (True, "fixture", "fixture")
    )
    server_now = store._parse_datetime(await store._query_response("RETURN time::now()"))
    old = server_now - timedelta(days=1)
    neurons = [
        replace(
            Neuron.create(
                NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
                f"integration candidate {index}",
                metadata={"_embedding": _vec(1.0 - index * 0.01, index * 0.01)},
                neuron_id=f"wset-node-{index:03d}",
            ),
            created_at=old,
        )
        for index in range(6)
    ]
    for neuron in neurons:
        await store.add_neuron(neuron)
    edge = replace(
        Synapse.create(
            neurons[0].id, neurons[5].id, SynapseType.RELATED_TO, synapse_id="wset-edge-0-5"
        ),
        created_at=old,
    )
    await store.add_synapse(edge)
    reference_time = store._parse_datetime(await store._query_response("RETURN time::now()"))
    return neurons, edge, reference_time.replace(tzinfo=UTC)


async def _run(store, reference_time, during, run_id=None):
    fired = False

    async def checkpoint(stage: str, _cursor, _details) -> None:
        nonlocal fired
        if stage == "similarity" and not fired:
            fired = True
            await during()

    result = await discover_semantic_synapses(
        store,
        _config(),
        checkpoint=checkpoint,
        resume_state=None,
        budget_check=None,
        reference_time=reference_time,
        run_id=run_id or "wset-run-" + uuid.uuid4().hex,
        owner_token="wset-owner-" + uuid.uuid4().hex,
    )
    assert fired, "the mutation must land while the run is in flight"
    return result


@pytest.mark.asyncio
async def test_recall_like_writes_do_not_abort_the_run_on_a_real_changefeed(
    store, monkeypatch
) -> None:
    neurons, edge, reference_time = await _world(store, monkeypatch)

    async def nothing(_stage, _cursor, _details) -> None:
        return None

    silent = await discover_semantic_synapses(
        store,
        _config(),
        checkpoint=nothing,
        resume_state=None,
        budget_check=None,
        reference_time=reference_time,
        run_id="wset-run-" + uuid.uuid4().hex,
        owner_token="wset-owner-" + uuid.uuid4().hex,
    )
    assert silent.synapses

    async def recall() -> None:
        # reconsolidation: counters and a trail in metadata, vector column untouched
        await store.update_neuron(
            replace(
                neurons[0], metadata={"_context_trail": [{"ts": "x"}], "_reconsolidation_count": 1}
            )
        )
        # the same write carrying the vector back (values identical)
        await store.update_neuron(
            replace(neurons[1], metadata={**neurons[1].metadata, "_reconsolidation_count": 2})
        )
        # reinforce: weight on an old synapse
        await store.update_synapse(replace(edge, weight=0.9))

    result = await _run(store, reference_time, recall)

    assert sorted((s.source_id, s.target_id) for s in result.synapses) == sorted(
        (s.source_id, s.target_id) for s in silent.synapses
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("what", ["content", "vector"])
async def test_a_content_or_vector_change_of_a_candidate_is_detected_on_a_real_changefeed(
    store, monkeypatch, what: str
) -> None:
    neurons, _edge, reference_time = await _world(store, monkeypatch)

    async def change() -> None:
        if what == "content":
            await store.update_neuron(replace(neurons[2], content="a different meaning"))
        else:
            await store.update_neuron(
                replace(neurons[2], metadata={"_embedding": _vec(0.05, 0.95)})
            )

    with pytest.raises(SemanticSourceChangedError):
        await _run(store, reference_time, change)


@pytest.mark.asyncio
async def test_a_vanished_synapse_of_a_candidate_is_caught_by_the_real_degree_guard(
    store, monkeypatch
) -> None:
    _neurons, edge, reference_time = await _world(store, monkeypatch)

    async def vanish() -> None:
        await store.delete_synapse(edge.id)

    with pytest.raises(SemanticSourceStaleError):
        await _run(store, reference_time, vanish)


async def _finished_discovery(store, monkeypatch):
    """Discovery run to the end on a real store, the way the apply stage will find it afterwards."""
    neurons, edge, reference_time = await _world(store, monkeypatch)
    lazy = replace(
        Neuron.create(NeuronType.CONCEPT, "not embedded yet", neuron_id="wset-lazy"),
        created_at=reference_time - timedelta(days=1),
    )
    await store.add_neuron(lazy)
    run_id = "wset-run-" + uuid.uuid4().hex

    async def nothing() -> None:
        return None

    result = await _run(store, reference_time, nothing, run_id=run_id)
    state_ref = staged_source_state_ref(result)
    assert state_ref is not None
    # the apply stage's world: a live lease and a progress record whose pending manifest names the
    # snapshot exactly as the engine writes it - the store releases a snapshot on no other terms
    brain_id = store._get_brain_id()
    owner = "wset-owner-" + uuid.uuid4().hex
    assert await store.acquire_consolidation_lease(brain_id, owner, lease_seconds=300)
    manifest = {
        "kind": "semantic_link_synapses",
        "version": 2,
        "source_token": result.source_token,
        "source_state_ref": state_ref,
        "synapses": [],
    }
    await store.create_consolidation_progress(
        {
            "brain_id": brain_id,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "engine_version": CONSOLIDATION_ENGINE_VERSION,
            "format_version": PROGRESS_FORMAT_VERSION,
            "requested_strategies": ["semantic_link"],
            "completed_strategies": [],
            "options_fingerprint": "wset-options",
            "reference_time": utcnow(),
            "owner_token": owner,
            "status": "paused",
            "current_strategy": "semantic_link",
            "phase": "semantic_link_apply",
            "cursor": None,
            "strategy_states": {
                "semantic_link": {"pending": [json.dumps(manifest, sort_keys=True)]}
            },
            "counters": {},
            "started_at": utcnow(),
            "updated_at": utcnow(),
            "last_error": None,
        }
    )
    kwargs = {
        "source_token": result.source_token,
        "reference_time": reference_time,
        "run_id": run_id,
    }
    return neurons, edge, lazy, state_ref, kwargs


@pytest.mark.asyncio
async def test_staged_verification_passes_through_recall_and_worker_traffic_on_a_real_changefeed(
    store, monkeypatch
) -> None:
    neurons, edge, lazy, state_ref, kwargs = await _finished_discovery(store, monkeypatch)
    assert await verify_staged_discovery_source(store, state_ref, **kwargs) is True

    await store.update_neuron(
        replace(neurons[0], metadata={"_context_trail": [{"ts": "x"}], "_reconsolidation_count": 1})
    )
    await store.update_synapse(replace(edge, weight=0.9, reinforced_count=3))
    await store.update_neuron_embeddings([(lazy.id, _vec(0.5, 0.5))])  # the worker's write path

    assert await verify_staged_discovery_source(store, state_ref, **kwargs) is True


@pytest.mark.asyncio
async def test_staged_verification_catches_vector_and_degree_changes_on_a_real_changefeed(
    store, monkeypatch
) -> None:
    neurons, edge, _lazy, state_ref, kwargs = await _finished_discovery(store, monkeypatch)
    await store.delete_synapse(edge.id)
    with pytest.raises(SemanticSourceStaleError):
        await verify_staged_discovery_source(store, state_ref, **kwargs)

    await store.update_neuron(replace(neurons[3], metadata={"_embedding": _vec(0.1, 0.9)}))
    with pytest.raises(SemanticSourceChangedError):
        await verify_staged_discovery_source(store, state_ref, **kwargs)
