"""The shared superseded-fact filter (engine.superseded_filter) on a REAL in-memory storage.

Real ``InMemoryStorage``, ``Neuron``, ``Fiber``, ``TypedMemory`` and ``RetrievalResult``; the
prose is produced by the real ``format_context``. The positive control (``test_control_*``) shows
the prose assertion CAN fail: without the anchor exclusion the superseded text is in the prose.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import MemoryType, Priority, Provenance, TypedMemory
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine import superseded_filter as sf
from surreal_memory.engine.retrieval_context import format_context
from surreal_memory.engine.retrieval_types import DepthLevel, RetrievalResult, Subgraph
from surreal_memory.safety.encryption import MemoryEncryptor
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

BRAIN = "testbrain"
OLD = "ROZKAZ 08-18 klienci embedduja wylacznie przez proxy 4000"
NEW = "ZMIANA 09-14 embedder omija proxy i chodzi lokalnie 18200"
PLAIN = "Wspomnienie bez typed memory o rerankerze Toniego"
STAMPED = "Stary neuron z metadanymi superseded bez dopasowanego fibra"


async def _brain(*, old_closed: bool = True) -> tuple[InMemoryStorage, dict[str, str]]:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    ids: dict[str, str] = {}
    now = utcnow()
    for key, text, meta in (
        ("old", OLD, {}),
        ("new", NEW, {}),
        ("plain", PLAIN, {}),
        ("stamped", STAMPED, {"_superseded": True}),
    ):
        neuron = Neuron.create(type=NeuronType.CONCEPT, content=text, metadata=meta)
        await storage.add_neuron(neuron)
        ids[f"n_{key}"] = neuron.id
        if key == "stamped":
            continue
        fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
        await storage.add_fiber(fiber)
        ids[f"f_{key}"] = fiber.id
    for key, closed in (("old", old_closed), ("new", False)):
        await storage.add_typed_memory(
            TypedMemory(
                fiber_id=ids[f"f_{key}"],
                memory_type=MemoryType.FACT,
                priority=Priority.from_int(5),
                provenance=Provenance(source="test"),
                created_at=now - timedelta(days=30),
                valid_from=now - timedelta(days=30),
                valid_until=now if closed else None,
                superseded_by=ids["f_new"] if closed and key == "old" else None,
            )
        )
    return storage, ids


def _result(ids: dict[str, str], fibers: list[str], *, levels: bool = True) -> RetrievalResult:
    meta = {}
    if levels:
        meta["activation_levels"] = {
            ids["n_old"]: 0.9,
            ids["n_stamped"]: 0.85,
            ids["n_new"]: 0.8,
            ids["n_plain"]: 0.7,
        }
    return RetrievalResult(
        answer=None,
        confidence=0.9,
        depth_used=DepthLevel.INSTANT,
        neurons_activated=4,
        fibers_matched=list(fibers),
        subgraph=Subgraph(neuron_ids=[], synapse_ids=[], anchor_ids=[]),
        context=f"## Relevant Memories\n\n- {OLD}\n- {NEW}\n- {PLAIN}",
        latency_ms=1.0,
        metadata=meta,
    )


@pytest.fixture(autouse=True)
def _filter_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(sf.DISABLE_ENV, raising=False)


async def test_superseded_fiber_leaves_list_others_keep_order() -> None:
    storage, ids = await _brain()
    res = _result(ids, [ids["f_old"], ids["f_new"], ids["f_plain"]])
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    assert out.result.fibers_matched == [ids["f_new"], ids["f_plain"]]
    assert out.excluded_fiber_ids == [ids["f_old"]]
    assert out.context_rebuilt is True


async def test_superseded_text_absent_from_every_prose_section() -> None:
    storage, ids = await _brain()
    res = _result(ids, [ids["f_old"], ids["f_new"], ids["f_plain"]])
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    ctx = out.result.context
    assert OLD not in ctx
    assert STAMPED not in ctx  # activated neuron stamped _superseded, no matched fiber
    assert NEW in ctx and PLAIN in ctx
    assert "## Related Information" in ctx  # the section exists, only without the old anchor


async def test_control_without_exclusion_old_text_is_in_related_information() -> None:
    """Positive control: the same prose built WITHOUT the exclusion carries the old text."""
    storage, ids = await _brain()
    res = _result(ids, [ids["f_new"], ids["f_plain"]])
    fibers = [await storage.get_fiber(ids["f_new"]), await storage.get_fiber(ids["f_plain"])]
    acts = sf._activations(res, exclude=set())
    ctx, _ = await format_context(
        storage=storage, activations=acts, fibers=[f for f in fibers if f], max_tokens=500
    )
    assert OLD in ctx and STAMPED in ctx


async def test_nothing_to_exclude_returns_the_same_object() -> None:
    storage, ids = await _brain(old_closed=False)
    res = _result(ids, [ids["f_old"], ids["f_new"], ids["f_plain"]])
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    assert out.result is res
    assert out.excluded_fiber_ids == []
    assert out.context_rebuilt is False


async def test_escape_hatch_keeps_superseded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(sf.DISABLE_ENV, "1")
    storage, ids = await _brain()
    res = _result(ids, [ids["f_old"], ids["f_new"]])
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    assert out.result is res
    assert out.enabled is False
    assert ids["f_old"] in out.result.fibers_matched and OLD in out.result.context


async def test_fiber_without_typed_memory_is_never_excluded() -> None:
    storage, ids = await _brain()
    res = _result(ids, [ids["f_plain"]])
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    assert out.result is res


async def test_everything_excluded_gives_empty_list_and_empty_prose() -> None:
    storage, ids = await _brain()
    res = _result(ids, [ids["f_old"]], levels=False)
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN)
    assert out.result.fibers_matched == []
    assert out.result.context == ""  # never the pre-filter prose


async def test_dash_and_underscore_ids_compare_equal() -> None:
    storage, ids = await _brain()
    res = _result(ids, [ids["f_old"], ids["f_new"]])
    # activation map keyed by the underscore record-id form of the old anchor
    levels = dict(res.metadata["activation_levels"])
    levels[ids["n_old"].replace("-", "_")] = levels.pop(ids["n_old"])
    res.metadata["activation_levels"] = levels
    # the exclusion set comes from the fiber anchor (dash form); the map key is underscore form
    exclude = await sf.excluded_anchor_ids(storage, [ids["f_old"]])
    acts = sf._activations(res, exclude)
    assert ids["n_old"].replace("-", "_") not in acts
    assert ids["n_new"] in acts
    assert sf.norm_id("neuron:ab_cd-ef") == sf.norm_id("ab-cd_ef")


async def test_encrypted_fiber_is_decrypted_in_rebuild(tmp_path: Path) -> None:
    storage, ids = await _brain()
    enc = MemoryEncryptor(keys_dir=tmp_path)
    plaintext = "Zaszyfrowane nowe wspomnienie o lokalnym embedderze"
    neuron = Neuron.create(
        type=NeuronType.CONCEPT, content=enc.encrypt(plaintext, BRAIN).ciphertext, metadata={}
    )
    await storage.add_neuron(neuron)
    fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
    fiber.metadata["encrypted"] = (
        True  # frozen dataclass, mutable metadata dict (as storage builds it)
    )
    await storage.add_fiber(fiber)
    res = _result(ids, [ids["f_old"], fiber.id], levels=False)
    out = await sf.filter_superseded(res, storage, max_tokens=500, brain_id=BRAIN, encryptor=enc)
    assert plaintext in out.result.context


def test_predicate_matches_recall_api_semantics() -> None:
    now = utcnow()
    tm = TypedMemory(
        fiber_id="f",
        memory_type=MemoryType.FACT,
        priority=Priority.from_int(5),
        provenance=Provenance(source="test"),
        created_at=now - timedelta(days=10),
        valid_from=now - timedelta(days=10),
        valid_until=now - timedelta(days=1),
    )
    assert sf.is_excluded_by_validity(tm, valid_at=None, include_superseded=False) is True
    assert sf.is_excluded_by_validity(tm, valid_at=None, include_superseded=True) is False
    assert sf.is_excluded_by_validity(None, valid_at=None, include_superseded=False) is False
    # point-in-time: a moment inside the window keeps it, after the window drops it
    assert (
        sf.is_excluded_by_validity(tm, valid_at=now - timedelta(days=5), include_superseded=False)
        is False
    )
    assert sf.is_excluded_by_validity(tm, valid_at=now, include_superseded=False) is True
