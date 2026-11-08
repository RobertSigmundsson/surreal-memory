"""A discovery epoch resumed from a checkpoint that finishes without saving a new snapshot, then an
apply pause and an apply resume: the apply manifest must name a staged snapshot that exists.

Regression: with an eager fresh state_id per owner epoch, a pause at the final similarity checkpoint
left the manifest pointing at revision 0 of a state_id that was never saved, and the apply resume
failed with "invalid semantic_link staged source-state reference"."""

import json

import pytest

from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from tests.unit.test_consolidation_mid_strategies_resume import _engine, _Progress
from tests.unit.test_semantic_discovery_working_set import _REFERENCE_TIME, _config
from tests.unit.test_semantic_link_apply_under_traffic import (
    _fixture_embedding,  # noqa: F401 - autouse fixture
    _world_with_lazy_neuron,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "nth"),
    [
        ("semantic_link_discovery_neurons", 1),
        ("semantic_link_discovery_neurons", 2),
        ("semantic_link_discovery_synapses", 1),
        ("semantic_link_discovery_synapses", 2),
        ("semantic_link_discovery_similarity", 1),
        ("semantic_link_discovery_similarity", 2),
        (None, 0),  # null control: no discovery pause, single epoch
    ],
)
async def test_resumed_epoch_then_apply_resume_names_existing_snapshot(
    phase: str | None, nth: int
) -> None:
    storage = _world_with_lazy_neuron("plain")
    storage.brain.config = _config()
    progress = _Progress("semantic_link", pause_phase=phase, pause_after_phase_calls=max(nth, 1))
    progress.reference_time = _REFERENCE_TIME
    storage.progress = progress

    def eng():
        return _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)

    if phase is not None:
        with pytest.raises(ConsolidationPausedError):
            await eng()._semantic_link(ConsolidationReport(), dry_run=False)
    progress.pause_phase = "semantic_link_apply"
    progress.pause_after_phase_calls = progress.phase_calls.get("semantic_link_apply", 0) + 2
    with pytest.raises(ConsolidationPausedError):
        await eng()._semantic_link(ConsolidationReport(), dry_run=False)
    manifest = json.loads(progress.strategy_state("semantic_link")["pending"][0])
    ref = manifest.get("source_state_ref")
    exists = (
        ref is not None and (ref["state_id"], ref["revision"]) in storage.semantic_discovery_states
    )
    assert exists, f"phase={phase} nth={nth}: manifest names a missing snapshot {ref}"
    await eng()._semantic_link(ConsolidationReport(), dry_run=False)
    assert progress.strategy_state("semantic_link")["phase"] == "semantic_link_complete"
