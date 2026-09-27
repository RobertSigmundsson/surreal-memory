"""Live SurrealDB regression: the merge fiber census must survive fibers with empty fields.

The Python SDK sends ``None`` as NONE, and SurrealDB drops NONE-valued keys from the
stored record (nested ones inside ``metadata`` too). The census used to fingerprint the
encoded page *with* those keys and re-fingerprint the page it read back *without* them,
so every page holding a fiber with an empty field failed validation and merge stopped with
"fiber census staged page is invalid" — on a real brain, where almost every fiber has an
empty ``summary`` or ``last_ghost_shown_at``, merge never ran at all.

Runs only against the explicitly opted-in loopback SMEM_TEST_SURREALDB_URL; the fixture
creates a unique database and uses the disposable test credentials.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationStrategy,
)
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

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
        reason="requires an explicit loopback SMEM_TEST_SURREALDB_URL",
    ),
]


@pytest_asyncio.fixture
async def store() -> AsyncIterator[SurrealDBStorage]:
    assert TEST_SURREALDB_URL is not None
    storage = SurrealDBStorage(
        url=TEST_SURREALDB_URL,
        user=TEST_AUTH[0],
        password=TEST_AUTH[1],
        namespace="smem_ci",
        database="it_" + uuid.uuid4().hex[:12],
    )
    await storage.initialize()
    brain = Brain.create(name="fiber-census-none-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def test_merge_census_completes_with_fibers_that_have_empty_fields(
    store: SurrealDBStorage,
) -> None:
    for i in range(3):
        neurons = [Neuron.create(NeuronType.CONCEPT, f"census none {i} {s}") for s in ("a", "b")]
        for neuron in neurons:
            await store.add_neuron(neuron)
        fiber = Fiber.create(
            neuron_ids={n.id for n in neurons},
            synapse_ids=set(),
            anchor_neuron_id=neurons[0].id,
            summary=None,  # empty field — dropped by the SDK/SurrealDB round trip
        )
        fiber.metadata["note"] = None  # nested empty value — dropped as well
        await store.add_fiber(fiber)

    report = await ConsolidationEngine(store, ConsolidationConfig()).run(
        strategies=[ConsolidationStrategy.MERGE]
    )

    assert report.extra.get("failed_strategies") in (None, []), report.extra
    assert report.extra.get("consolidation_status") == "completed", report.extra
