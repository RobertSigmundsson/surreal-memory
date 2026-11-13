"""W-6 live twin of ``test_dedup_content_hash.py`` -- real SurrealDB, real encoder, real index.

STANOWISKO: kopia bazy (pula 18680-18689), wlasna przestrzen nazw ``higiena_test``;
skipped without ``SURREALDB_URL``, asserts (does not skip) on a production port.

KA.11: after the FIRST write exactly one neuron of the text's ``content_hash`` is an anchor
(positive control); after the SECOND identical write -- with more than 30 anchors sharing the
first word, so the first-word pool misses -- still exactly one, no alias neuron, and both fibers
anchor on it. ``find_neurons_by_content_hash`` is also measured on its own (oldest first, capped).

MUTANT (must turn this file red): ``DedupPipeline._get_candidates`` without the ``content_hash``
query (the old first-word pool) -- the second write mints a second anchor.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from surreal_memory.core.brain import BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.dedup.config import DedupConfig
from surreal_memory.engine.dedup.pipeline import DedupPipeline
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.utils.simhash import simhash
from tests.unit._higiena_live import close_live_store, open_live_store, requires_live

pytestmark = requires_live

CONTENTS = [
    "TODO: przepisac skrypt kopii zapasowej tak, zeby sprawdzal sume kontrolna archiwum",
    "Decision: konfiguracje bramki trzymamy w jednym pliku, bo dwa pliki sie rozjezdzaly",
    "/home/robert/notatki/plan-okna.md zawiera harmonogram okna konserwacyjnego bazy",
    "Kubernetes nie jest potrzebny na tym hoscie, wystarczy podman z jednostkami systemd",
]


@pytest.fixture
async def store() -> Any:
    st, brain = await open_live_store("w6-dedup-hash-live")
    try:
        yield st
    finally:
        await close_live_store(st, brain)


async def _seed_fillers(store: Any, term: str, n: int = 35) -> None:
    for i in range(n):
        text = f"{term} wypelniacz {uuid4()} o zupelnie innej sprawie {i * 7919}"
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT,
                content=text,
                metadata={"is_anchor": True},
                neuron_id=f"0000fill-0000-0000-0000-{i:012d}",
                content_hash=simhash(text),
            )
        )


async def test_find_neurons_by_content_hash_oldest_first_and_capped(store: Any) -> None:
    h = 123456789
    ids = [str(uuid4()) for _ in range(4)]
    for nid in ids:
        await store.add_neuron(
            Neuron.create(
                type=NeuronType.CONCEPT, content=f"c {nid}", neuron_id=nid, content_hash=h
            )
        )
    await store.add_neuron(
        Neuron.create(type=NeuronType.CONCEPT, content="inny", content_hash=h + 1)
    )

    found = await store.find_neurons_by_content_hash(h, limit=10)
    assert [n.id for n in found] == ids  # kontrola pozytywna: 4 trafienia, kolejność utworzenia
    assert len(await store.find_neurons_by_content_hash(h, limit=2)) == 2
    assert await store.find_neurons_by_content_hash(h + 99, limit=10) == []  # kontrola zerowa


@pytest.mark.parametrize("content", CONTENTS, ids=["todo", "decision", "sciezka", "zdanie"])
async def test_drugi_identyczny_zapis_jedna_kotwica(store: Any, content: str) -> None:
    await _seed_fillers(store, content.split()[0].strip(".,;:!?").lower())
    pipeline = DedupPipeline(DedupConfig(enabled=True, max_candidates=30), store)
    encoder = MemoryEncoder(store, BrainConfig(), dedup_pipeline=pipeline)

    first = await encoder.encode(content)
    after_first = await store.find_neurons_by_content_hash(simhash(content), limit=50)
    assert [n.id for n in after_first if n.metadata.get("is_anchor")] == [
        first.fiber.anchor_neuron_id
    ]  # kontrola pozytywna: po pierwszym zapisie dokładnie 1 kotwica

    second = await encoder.encode(content)
    after_second = await store.find_neurons_by_content_hash(simhash(content), limit=50)

    assert second.fiber.anchor_neuron_id == first.fiber.anchor_neuron_id
    assert [n.id for n in after_second] == [
        n.id for n in after_first
    ]  # żadnego aliasu ani drugiej kotwicy
    assert second.fiber.id != first.fiber.id
    stored = await store.get_fiber(second.fiber.id)
    assert stored is not None and stored.anchor_neuron_id == first.fiber.anchor_neuron_id
