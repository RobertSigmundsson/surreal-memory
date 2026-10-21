"""Pinned fibers protect every neuron that compression would mutate."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.compression import CompressionConfig, CompressionEngine, CompressionTier
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

ORIGINAL = (
    "The release contains detailed migration notes and recovery instructions. "
    "The first check is green. The second check verifies graph integrity. "
    "The third check confirms checkpoints. The fourth check reads the backup. "
    "The fifth check reviews the service state."
)


async def _store_with_fiber(
    neuron_ids: set[str] | None = None,
) -> tuple[InMemoryStorage, Fiber, list[Neuron]]:
    store = InMemoryStorage()
    brain = Brain.create(name="pinned-compression-test")
    await store.save_brain(brain)
    store.set_brain(brain.id)
    neurons = [
        Neuron.create(type=NeuronType.CONCEPT, content=ORIGINAL),
        Neuron.create(type=NeuronType.ENTITY, content="Related entity details."),
    ]
    for neuron in neurons:
        await store.add_neuron(neuron)
    members = neuron_ids or {neurons[0].id}
    fiber = Fiber(
        id="f-compress-target",
        neuron_ids=members,
        synapse_ids=set(),
        anchor_neuron_id=neurons[0].id,
        compression_tier=0,
        created_at=utcnow() - timedelta(days=15),
    )
    await store.add_fiber(fiber)
    return store, fiber, neurons


async def _refresh(
    _store: InMemoryStorage, neuron: Neuron, content: str, *, brain: Brain | None = None
) -> Neuron:
    return replace(neuron, content=content)


async def _add_pinned_fiber(
    store: InMemoryStorage, fiber_id: str, neuron_ids: set[str], anchor_id: str
) -> Fiber:
    pinned = Fiber(
        id=fiber_id,
        neuron_ids=neuron_ids,
        synapse_ids=set(),
        anchor_neuron_id=anchor_id,
        created_at=utcnow() - timedelta(days=200),
        pinned=True,
    )
    await store.add_fiber(pinned)
    return pinned


@pytest.mark.asyncio
async def test_shared_pinned_anchor_skips_before_backup_or_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    store, fiber, neurons = await _store_with_fiber()
    await _add_pinned_fiber(store, "f-pinned", {neurons[0].id}, neurons[0].id)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    result = await CompressionEngine(
        store, CompressionConfig(tier1_max_sentences=1)
    ).compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    saved = await store.get_fiber(fiber.id)
    assert result.skipped and result.skip_reason == "pinned_neuron_protected"
    assert saved is not None and "_compression_pending" not in saved.metadata
    assert await store.get_compression_backup(fiber.id) is None
    assert (await store.get_neuron(neurons[0].id)).content == ORIGINAL


@pytest.mark.asyncio
async def test_direct_dry_run_skips_shared_pinned_anchor() -> None:
    store, fiber, _neurons = await _store_with_fiber()
    await _add_pinned_fiber(
        store, "f-pinned-preview", {fiber.anchor_neuron_id}, fiber.anchor_neuron_id
    )

    result = await CompressionEngine(store).compress_fiber(
        fiber, CompressionTier.EXTRACTIVE, dry_run=True
    )

    assert result.skipped and result.skip_reason == "pinned_neuron_protected"
    saved = await store.get_fiber(fiber.id)
    assert saved is not None and saved.compression_tier == 0
    assert "_compression_pending" not in saved.metadata


@pytest.mark.asyncio
async def test_graph_only_protects_pinned_nonanchor_member(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, _, neurons = await _store_with_fiber({"unused-anchor", "unused-member"})
    # Rebuild the test fiber with two real neurons; GRAPH_ONLY mutates both.
    fiber = Fiber(
        id="f-graph-target",
        neuron_ids={neuron.id for neuron in neurons},
        synapse_ids=set(),
        anchor_neuron_id=neurons[0].id,
        created_at=utcnow() - timedelta(days=200),
    )
    await store.add_fiber(fiber)
    await _add_pinned_fiber(store, "f-pinned-member", {neurons[1].id}, neurons[1].id)

    async def _contents_refreshed(_storage, updates, **kwargs):
        return [replace(neuron, content=content) for neuron, content in updates]

    monkeypatch.setattr(compression, "contents_refreshed", _contents_refreshed)

    result = await CompressionEngine(store).compress_fiber(fiber, CompressionTier.GRAPH_ONLY)

    assert result.skipped and result.skip_reason == "pinned_neuron_protected"
    assert (await store.get_neuron(neurons[0].id)).content == ORIGINAL
    assert (await store.get_neuron(neurons[1].id)).content == "Related entity details."
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_unshared_fiber_still_compresses(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, fiber, neurons = await _store_with_fiber()
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    result = await CompressionEngine(
        store, CompressionConfig(tier1_max_sentences=1)
    ).compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    assert not result.skipped and result.new_tier == CompressionTier.EXTRACTIVE
    assert (await store.get_neuron(neurons[0].id)).content != ORIGINAL
    assert await store.get_compression_backup(fiber.id) is not None


@pytest.mark.asyncio
async def test_direct_call_skips_fiber_that_is_itself_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    store, fiber, neurons = await _store_with_fiber()
    pinned = replace(fiber, pinned=True)
    await store.update_fiber(pinned)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    result = await CompressionEngine(store).compress_fiber(pinned, CompressionTier.EXTRACTIVE)

    assert result.skipped and result.skip_reason == "pinned_fiber"
    assert (await store.get_neuron(neurons[0].id)).content == ORIGINAL
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_pinned_lookup_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, fiber, neurons = await _store_with_fiber()

    async def fail_lookup() -> set[str]:
        raise RuntimeError("pin census unavailable")

    monkeypatch.setattr(store, "get_pinned_neuron_ids", fail_lookup)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    with pytest.raises(RuntimeError, match="pin census unavailable"):
        await CompressionEngine(store).compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    saved = await store.get_fiber(fiber.id)
    assert saved is not None and "_compression_pending" not in saved.metadata
    assert (await store.get_neuron(neurons[0].id)).content == ORIGINAL
    assert await store.get_compression_backup(fiber.id) is None


@pytest.mark.asyncio
async def test_replay_preserves_intent_when_neuron_becomes_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from surreal_memory.engine import compression

    store, fiber, neurons = await _store_with_fiber()
    update_neuron = store.update_neuron

    async def interrupt(_updated: Neuron) -> None:
        raise RuntimeError("interrupted after intent")

    monkeypatch.setattr(store, "update_neuron", interrupt)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)
    engine = CompressionEngine(store, CompressionConfig(tier1_max_sentences=1))
    with pytest.raises(RuntimeError, match="interrupted after intent"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)
    monkeypatch.setattr(store, "update_neuron", update_neuron)
    await _add_pinned_fiber(store, "f-new-pinned", {neurons[0].id}, neurons[0].id)

    with pytest.raises(RuntimeError, match=r"(?i)pinned neuron.*compression replay"):
        await engine.compress_fiber(fiber, CompressionTier.EXTRACTIVE)

    with pytest.raises(RuntimeError, match=r"(?i)pinned neuron.*compression replay"):
        await engine.run()

    saved = await store.get_fiber(fiber.id)
    assert saved is not None and "_compression_pending" in saved.metadata
    assert saved.compression_tier == 0
    assert (await store.get_neuron(neurons[0].id)).content == ORIGINAL


@pytest.mark.asyncio
async def test_run_loads_pinned_neurons_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.engine import compression

    store, _, neurons = await _store_with_fiber()
    second = Fiber(
        id="f-compress-second",
        neuron_ids={neurons[1].id},
        synapse_ids=set(),
        anchor_neuron_id=neurons[1].id,
        created_at=utcnow() - timedelta(days=15),
    )
    await store.add_fiber(second)
    original_lookup = store.get_pinned_neuron_ids
    calls = 0

    async def count_lookup() -> set[str]:
        nonlocal calls
        calls += 1
        return await original_lookup()

    async def _contents_refreshed(_storage, updates, **kwargs):
        return [replace(neuron, content=content) for neuron, content in updates]

    monkeypatch.setattr(store, "get_pinned_neuron_ids", count_lookup)
    monkeypatch.setattr(compression, "contents_refreshed", _contents_refreshed)
    monkeypatch.setattr(compression, "content_refreshed", _refresh)
    monkeypatch.setattr("surreal_memory.plugins.get_compression_fn", lambda: None)

    await CompressionEngine(store, CompressionConfig(tier1_max_sentences=1)).run()

    assert calls == 1
