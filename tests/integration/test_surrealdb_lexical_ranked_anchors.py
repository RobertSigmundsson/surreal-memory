"""Live integration test for the ranked lexical (keyword-anchor) path (skipped unless
SURREALDB_URL is set — see tests/integration/test_surrealdb_query_shapes.py's header for how to
run this against a real SurrealDB >= 3.2.0).

The defect this pins: ``find_neurons(content_contains=...)`` matches through the BM25 full-text
index (``content @@ $content_contains``) but then does ``ORDER BY id LIMIT {limit}`` — the BM25
score the index just computed is never read. For the keyword-anchor retriever in
``engine/retrieval.py`` this means a small ``limit`` (2, when IDF weighting is off) returns
whichever two matching neurons happen to sort first by id, regardless of how well they actually
match the term — measured on the production brain: the term "reranker" matched 100 neurons, and
the two returned were a 972-character document mentioning it once and a 455-character one, while
the two eight-character neurons whose entire content IS "RERANKER"/"Reranker" never surfaced.

A unit test with a mocked storage cannot tell the difference between "the BM25 index ranked this"
and "the mock said so" — only a real full-text index, searched with a real term matching more
candidates than the limit, proves the ranked ordering actually reaches a short, exact match that
id ordering would cut off.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

from surreal_memory.core.brain import Brain
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

SURREALDB_URL = os.getenv("SURREALDB_URL")
SURREALDB_USER = os.getenv("SURREALDB_USER", "root")
SURREALDB_PASS = os.getenv("SURREALDB_PASS", "root")
SURREALDB_NS = os.getenv("SURREALDB_NS", "smem_it")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not SURREALDB_URL, reason="requires SURREALDB_URL (live SurrealDB >= 3.2.0)"
    ),
]

_TERM = "widget"
# Ids sorted BEFORE the target's, like the KNN test's filler pattern — `ORDER BY id LIMIT 2`
# is guaranteed to return these two, never the target, regardless of relevance.
_DECOY_COUNT = 5
_FILLER_COUNT = 50
_TARGET_ID = "zzzzzzzz-target-neuron"
_SHORT_ID = "zzzzzzzy-short-neuron"


@pytest_asyncio.fixture
async def store():
    """A fresh store, scoped to its own throwaway database."""
    storage = SurrealDBStorage(
        url=SURREALDB_URL,
        user=SURREALDB_USER,
        password=SURREALDB_PASS,
        namespace=SURREALDB_NS,
        database="it_" + uuid.uuid4().hex[:12],
        embedding_dim=4,
    )
    await storage.initialize()
    brain = Brain.create(name="lexical-ranked-it")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    try:
        yield storage
    finally:
        await storage.close()


async def _seed(store: SurrealDBStorage) -> None:
    """Fifty fillers that never mention the term, five long decoys (the term buried once in an
    unrelated paragraph, ids sorting first), and a target whose ENTIRE content is the bare term
    (BM25 rewards term frequency / short field length — the same pattern measured on production
    for "reranker"). The fillers matter: BM25's IDF component is degenerate (every score becomes
    exactly 0.0, and ties fall back to insertion order) when the term appears in 100% of the
    corpus, which five decoys + one target alone would do — measured directly against this test
    server while designing this test, not assumed. A sixth neuron, `_SHORT_ID`, exists only for
    the length-gate test below and is not otherwise a match target here."""
    fillers = [
        Neuron.create(
            type=NeuronType.CONCEPT,
            content=f"Completely unrelated filler content number {i} about gardening and weather.",
            neuron_id=f"filler-{i:03d}",
        )
        for i in range(_FILLER_COUNT)
    ]
    decoys = [
        Neuron.create(
            type=NeuronType.CONCEPT,
            content=(
                f"Meeting notes {i}: we discussed the quarterly roadmap, budget allocation, "
                f"staffing, and in passing someone mentioned a {_TERM} as an aside before moving "
                "on to the next agenda item entirely unrelated to it."
            ),
            neuron_id=f"aaaaaaaa-decoy-{i:02d}",
        )
        for i in range(_DECOY_COUNT)
    ]
    target = Neuron.create(type=NeuronType.CONCEPT, content=_TERM, neuron_id=_TARGET_ID)
    short = Neuron.create(type=NeuronType.CONCEPT, content=_TERM, neuron_id=_SHORT_ID)
    await store.add_neurons_batch([*fillers, *decoys, target, short], record_change=False)


@pytest.mark.timeout(120)
async def test_ranked_lookup_surfaces_the_best_match_id_ordering_would_cut_off(
    store: SurrealDBStorage,
) -> None:
    await _seed(store)

    # Baseline (unchanged method): id ordering returns the first two decoys, never the target —
    # this is the defect itself, still present after the fix because `find_neurons` is untouched.
    id_ordered = await store.find_neurons(content_contains=_TERM, limit=2)
    id_ordered_ids = {n.id for n in id_ordered}
    assert id_ordered_ids == {"aaaaaaaa-decoy-00", "aaaaaaaa-decoy-01"}
    assert _TARGET_ID not in id_ordered_ids, (
        "test setup invariant: id ordering must miss the target"
    )

    # Fixed path: BM25-ranked ordering surfaces the exact short match within the same limit=2 that
    # id ordering could not. This call does not exist before the fix (AttributeError -> test FAILS
    # on pre-fix code, exactly as required).
    ranked = await store.find_neurons_ranked(content_contains=_TERM, limit=2)
    ranked_ids = {n.id for n in ranked}
    assert _TARGET_ID in ranked_ids, "ranked ordering must surface the best BM25 match"
    assert len(ranked) == 2


@pytest.mark.timeout(120)
async def test_ranked_lookup_length_gate_drops_short_content_without_losing_the_call(
    store: SurrealDBStorage,
) -> None:
    """`min_content_len` (the L-C arm) must exclude short-content matches even when they would
    otherwise rank first, without erroring when that leaves fewer than `limit` results."""
    await _seed(store)

    gated = await store.find_neurons_ranked(content_contains=_TERM, limit=2, min_content_len=25)
    gated_ids = {n.id for n in gated}
    assert _TARGET_ID not in gated_ids, "content shorter than min_content_len must be excluded"
    assert _SHORT_ID not in gated_ids
    # Only the decoys clear the length gate; both are returned since the gate leaves exactly 5
    # candidates >= 2, and ranking among them is still by BM25 score (best two of the five decoys).
    assert gated_ids, "the length gate must not drop every candidate here"
    assert gated_ids <= {f"aaaaaaaa-decoy-{i:02d}" for i in range(_DECOY_COUNT)}
