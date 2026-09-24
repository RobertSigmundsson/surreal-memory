"""engine.remember_api — the write shared by ``smem remember`` and ``/v1/remember`` (real InMemoryStorage)."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.memory_types import MemoryType, Priority
from surreal_memory.engine import remember_api as ra
from surreal_memory.storage.memory_store import InMemoryStorage

TEXT = "Rozliczenia przechodza na PostgreSQL po audycie modulu faktur"
TS = datetime(2026, 9, 1, 12, 0)
POD = ra.Attribution(
    source="http:claude-pod",
    created_by="agent:pod_a",
    stored_by={"agent_id": "agent:pod_a", "tor": "http:claude-pod", "kanal": "recall-http"},
)


async def _storage() -> InMemoryStorage:
    s = InMemoryStorage()
    brain = Brain.create(name="t", config=BrainConfig(activation_threshold=0.1))
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


async def _store(s: InMemoryStorage, attribution: ra.Attribution, **kw: Any) -> ra.StoredMemory:
    brain = await s.get_brain(s.brain_id or "")
    assert brain is not None
    return await ra.encode_and_store(
        s,
        brain.config,
        kw.pop("content", TEXT),
        tags=kw.pop("tags", {"k4"}),
        mem_type=MemoryType.DECISION,
        mem_priority=Priority.NORMAL,
        expiry_days=90,
        project_id=None,
        event_timestamp=TS,
        attribution=attribution,
        **kw,
    )


def _counts(s: InMemoryStorage) -> tuple[int, int, int, int]:
    b = s.brain_id or ""
    return (
        len(s._neurons[b]),
        len(s._fibers[b]),
        len(s._synapses[b]),
        len(s._typed_memories[b]),
    )


@pytest.mark.asyncio
async def test_cli_attribution_keeps_today_values() -> None:
    s = await _storage()
    seen: list[dict[str, Any]] = []
    orig = ra.MemoryEncoder.encode

    async def spy(self: Any, **kw: Any) -> Any:
        seen.append(kw)
        return await orig(self, **kw)

    with patch.object(ra.MemoryEncoder, "encode", spy):
        stored = await _store(s, ra.CLI_ATTRIBUTION)
    assert seen[0]["metadata"] is None
    tm = stored.typed_mem
    assert (tm.source, tm.provenance.created_by) == ("user_input", "user")
    fiber = await s.get_fiber(stored.fiber_id)
    anchor = await s.get_neuron(stored.anchor_neuron_id)
    assert fiber is not None and anchor is not None
    assert "stored_by" not in fiber.metadata and "stored_by" not in anchor.metadata


@pytest.mark.asyncio
async def test_pod_attribution_lands_on_anchor_fiber_and_typed_memory_without_graph_change() -> (
    None
):
    cli, pod = await _storage(), await _storage()
    a = await _store(cli, ra.CLI_ATTRIBUTION)
    b = await _store(pod, POD)
    assert _counts(cli) == _counts(pod)
    assert (a.neurons_created, a.neurons_linked, a.synapses_created) == (
        b.neurons_created,
        b.neurons_linked,
        b.synapses_created,
    )
    fiber = await pod.get_fiber(b.fiber_id)
    anchor = await pod.get_neuron(b.anchor_neuron_id)
    assert fiber is not None and anchor is not None
    assert fiber.metadata["stored_by"]["agent_id"] == "agent:pod_a"
    assert anchor.metadata["stored_by"]["tor"] == "http:claude-pod"
    assert anchor.content == (await cli.get_neuron(a.anchor_neuron_id)).content  # type: ignore[union-attr]
    stored_tm = await pod.get_typed_memory(b.fiber_id)
    assert stored_tm is not None
    assert (stored_tm.source, stored_tm.provenance.created_by) == ("http:claude-pod", "agent:pod_a")


@pytest.mark.asyncio
async def test_explicit_priority_and_attribution_merge_in_encoder_metadata() -> None:
    s = await _storage()
    seen: list[dict[str, Any]] = []
    orig = ra.MemoryEncoder.encode

    async def spy(self: Any, **kw: Any) -> Any:
        seen.append(kw)
        return await orig(self, **kw)

    with patch.object(ra.MemoryEncoder, "encode", spy):
        await _store(s, POD, priority_was_explicit=True)
    assert seen[0]["metadata"]["priority"] == Priority.NORMAL.value
    assert seen[0]["metadata"]["stored_by"]["agent_id"] == "agent:pod_a"


def test_check_content_gate_never_carries_the_match() -> None:
    secret = "moje haslo password=Sup3rTajne!2026 do bazy"  # noqa: S105 — gate fixture
    with pytest.raises(ra.SensitiveContentError) as exc:
        ra.check_content(secret, force=False, redact=False)
    assert "Sup3rTajne" not in str(exc.value) and exc.value.types == ["password"]
    red = ra.check_content(secret, force=False, redact=True)
    assert red.redacted and "Sup3rTajne" not in red.content and len(red.matches) == 1
    forced = ra.check_content(secret, force=True, redact=False)
    assert forced.content == secret and not forced.redacted and len(forced.matches) == 1
    clean = ra.check_content(TEXT, force=False, redact=False)
    assert (clean.content, clean.matches, clean.redacted) == (TEXT, (), False)


def test_type_expiry_priority_resolution() -> None:
    assert ra.resolve_memory_type("DECISION", TEXT) is MemoryType.DECISION
    with pytest.raises(ra.InvalidMemoryTypeError) as exc:
        ra.resolve_memory_type("nieznany", TEXT)
    assert "fact" in exc.value.valid
    assert isinstance(ra.resolve_memory_type(None, TEXT), MemoryType)
    assert ra.resolve_expiry_days(MemoryType.DECISION, None, ephemeral=False) == 90
    assert ra.resolve_expiry_days(MemoryType.FACT, None, ephemeral=False) is None
    assert ra.resolve_expiry_days(MemoryType.FACT, None, ephemeral=True) == 1
    assert ra.resolve_expiry_days(MemoryType.DECISION, 7, ephemeral=True) == 7
    assert ra.resolve_priority(None) == (Priority.NORMAL, False)
    assert ra.resolve_priority(8) == (Priority.from_int(8), True)


@pytest.mark.asyncio
async def test_auto_save_restored_even_when_encoding_fails() -> None:
    s = await _storage()
    calls: list[str] = []
    s.disable_auto_save = lambda: calls.append("off")  # type: ignore[method-assign]
    s.enable_auto_save = lambda: calls.append("on")  # type: ignore[method-assign]

    async def boom(self: Any, **kw: Any) -> Any:
        raise RuntimeError("encoder down")

    with patch.object(ra.MemoryEncoder, "encode", boom), pytest.raises(RuntimeError):
        await _store(s, POD)
    assert calls == ["off", "on"]
