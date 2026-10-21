"""No automatic restart once the apply stage may have written edges (independent review r1, P1).

The durable edge counter trails the writes by up to one checkpoint batch, so a crash inside the apply
stage leaves edges the counters do not know about. A restart that rediscovers the pairs would build
the same deterministic edge ids with a new ``created_at`` and every later resume would then fail on
"changed after its checkpoint". The restart is therefore refused as soon as the stage has begun, and
the run fails once; the next run supersedes it and rediscovers from the committed edges.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from tests.unit.test_semantic_discovery_working_set import _candidates
from tests.unit.test_semantic_link_apply_under_traffic import (
    _forget_the_durable_cursor,
    _world_with_lazy_neuron,
)
from tests.unit.test_semantic_link_auto_restart import (  # noqa: F401 - fixtures are autouse/params
    _fixture_embedding,
    _phases,
    _progress,
    _run,
    _state,
    kind,
)


@pytest.mark.asyncio
async def test_a_source_change_after_unrecorded_writes_is_not_restarted(kind: str) -> None:  # noqa: F811
    storage = _world_with_lazy_neuron(kind)
    progress = _progress(pause_phase="semantic_link_apply", pause_after_phase_calls=2)
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    assert storage.added_synapses, "the stage must have written edges before the crash"
    _forget_the_durable_cursor(progress)  # edges exist, the counters say 0
    storage.update_neuron(_candidates(kind)[0], content="rewritten while the run was down")

    with pytest.raises(ConsolidationProgressError, match="source fence failed"):
        await _run(storage, progress, restarts=2)

    assert not _state(progress).get("auto_recovery")


@pytest.mark.asyncio
async def test_the_apply_stage_is_marked_before_its_first_edge_is_written(kind: str) -> None:  # noqa: F811
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    phases_at_first_write: list[str] = []
    add_synapse = storage.add_synapse

    async def spy(synapse: Any) -> str:
        if not phases_at_first_write:
            phases_at_first_write.extend(_phases(progress))
        return await add_synapse(synapse)

    storage.add_synapse = spy  # type: ignore[method-assign]

    await _run(storage, progress, restarts=2)

    assert phases_at_first_write, "the run must write at least one edge"
    assert "semantic_link_apply_started" in phases_at_first_write


@pytest.mark.asyncio
async def test_a_budget_pause_before_the_first_edge_still_allows_the_restart(
    kind: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The marker means "an edge may exist", so a pause before any write must not set it (r2)."""
    from surreal_memory.engine.consolidation import ConsolidationEngine

    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    original = ConsolidationEngine._check_progress_budget
    seen = {"checks_after_pending": 0, "paused": 0}

    async def budget(self: Any) -> None:
        if "semantic_link_pending" in _phases(progress):
            seen["checks_after_pending"] += 1
            # 1 = the check that ends the pending checkpoint, 2 = the next one inside the apply stage
            if seen["checks_after_pending"] == 2 and not seen["paused"]:
                seen["paused"] += 1
                raise ConsolidationPausedError("strategy budget exhausted")
        await original(self)

    monkeypatch.setattr(ConsolidationEngine, "_check_progress_budget", budget)
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    assert not storage.added_synapses, "the pause must come before the first edge"
    storage.update_neuron(_candidates(kind)[0], content="edited between two installments")

    report = await _run(storage, progress, restarts=2)

    assert len(_state(progress)["auto_recovery"]) == 1
    assert report.semantic_synapses_created > 0


@pytest.mark.asyncio
async def test_no_budget_check_sits_between_the_marker_and_the_first_edge(
    kind: str,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pause there would persist the marker for a stage that has written nothing (review r2)."""
    from surreal_memory.engine.consolidation import ConsolidationEngine

    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    events: list[str] = []
    original_budget = ConsolidationEngine._check_progress_budget
    original_checkpoint = progress.checkpoint
    original_add = storage.add_synapse

    async def budget(self: Any) -> None:
        events.append("budget")
        await original_budget(self)

    async def checkpoint(strategy: str, phase: str, **kwargs: Any) -> None:
        await original_checkpoint(strategy, phase, **kwargs)
        if phase == "semantic_link_apply_started":
            events.append("marker")

    async def add(synapse: Any) -> str:
        events.append("add")
        return await original_add(synapse)

    monkeypatch.setattr(ConsolidationEngine, "_check_progress_budget", budget)
    monkeypatch.setattr(progress, "checkpoint", checkpoint)
    storage.add_synapse = add  # type: ignore[method-assign]

    await _run(storage, progress, restarts=2)

    marker, first_add = events.index("marker"), events.index("add")
    assert marker < first_add
    assert "budget" not in events[marker + 1 : first_add]
