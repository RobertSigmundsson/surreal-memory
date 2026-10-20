"""P5: the apply stage of semantic_link survives the traffic that arrives while it is paused.

Discovery ends with a manifest of the synapses to write. A resumed apply used to re-verify the world
the strict way: the whole source fence plus a fingerprint of every CONCEPT/ENTITY neuron and every
synapse. Any recall (rewriting anchors' metadata, reinforcing a synapse the stage had just written)
or a vector filled by the worker then failed the run. Now the manifest names the staged discovery
state; the resume re-verifies exactly what the result depends on (the candidates' digests and
degrees), and an edge already written is recognized by identity, not by every field a recall may
have touched.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.consolidation import ConsolidationReport, ConsolidationStrategy
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from surreal_memory.storage.surrealdb.semantic_discovery_state import (
    SurrealDBSemanticDiscoveryStateMixin,
)
from tests.unit.test_consolidation_mid_strategies_resume import _engine, _Progress
from tests.unit.test_semantic_discovery_working_set import (
    _OLD,
    _REFERENCE_TIME,
    _candidates,
    _config,
    _FeedStorage,
    _world,
)


@pytest.fixture(autouse=True)
def _fixture_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )
    monkeypatch.setattr(
        "surreal_memory.engine.consolidation._SEMANTIC_LINK_APPLY_CHECKPOINT_BATCH", 1
    )


@pytest.fixture(params=["plain", "bounded"])
def kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "bounded":
        monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)
    return str(request.param)


class _ApplyStorage(_FeedStorage):
    """The real store hands a snapshot back only while the run's pending manifest names it."""

    progress: _Progress | None = None

    async def add_synapse(self, synapse: Any) -> str:
        result = await super().add_synapse(synapse)
        self._log("synapse", "create", synapse.id, "brain-1", synapse)
        return result

    async def load_semantic_discovery_state(
        self, state_id: str, revision: int
    ) -> dict[str, Any] | None:
        payload = await super().load_semantic_discovery_state(state_id, revision)
        if payload is None or self.progress is None:
            return payload
        pending = self.progress.strategy_state("semantic_link").get("pending")
        named = SurrealDBSemanticDiscoveryStateMixin._manifest_references(
            pending,
            state_id=state_id,
            revision=revision,
            source_token=str(payload.get("source_token", "")),
        )
        return payload if named else None


def _world_with_lazy_neuron(kind: str) -> _ApplyStorage:
    storage = _world(kind, storage_cls=_ApplyStorage)
    # a neuron the worker has not embedded yet: CONCEPT, so part of every fingerprint, no vector
    storage.neurons["ws-lazy"] = replace(
        Neuron.create(NeuronType.CONCEPT, "not embedded yet", neuron_id="ws-lazy"), created_at=_OLD
    )
    return storage  # type: ignore[return-value]


def _progress() -> _Progress:
    progress = _Progress(
        "semantic_link", pause_phase="semantic_link_apply", pause_after_phase_calls=2
    )
    progress.reference_time = _REFERENCE_TIME  # type: ignore[attr-defined]
    return progress


async def _run_until_paused_in_apply(storage: _ApplyStorage, progress: _Progress) -> None:
    storage.brain.config = _config()
    storage.progress = progress
    with pytest.raises(ConsolidationPausedError, match="simulated interruption"):
        await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
            ConsolidationReport(), dry_run=False
        )
    assert len(storage.added_synapses) >= 2, "the stage must have written edges before the pause"


async def _resume(storage: _ApplyStorage, progress: _Progress) -> ConsolidationReport:
    report = ConsolidationReport()
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        report, dry_run=False
    )
    return report


async def _silent_edges(kind: str) -> list[str]:
    storage = _world_with_lazy_neuron(kind)
    storage.brain.config = _config()
    progress = _Progress("semantic_link")
    progress.reference_time = _REFERENCE_TIME  # type: ignore[attr-defined]
    await _engine(storage, ConsolidationStrategy.SEMANTIC_LINK, progress)._semantic_link(
        ConsolidationReport(), dry_run=False
    )
    assert storage.added_synapses
    return sorted(storage.added_synapses)


def _forget_the_durable_cursor(progress: _Progress) -> None:
    """A crash between add_synapse and its checkpoint: the edges exist, the cursor does not know."""
    state = progress.strategy_state("semantic_link")
    state["cursor"] = None
    state["counters"] = {**state["counters"], "semantic_synapses_created": 0}


def _pending_manifest(progress: _Progress) -> dict[str, Any]:
    pending = progress.strategy_state("semantic_link")["pending"]
    manifest = json.loads(pending[0])
    assert manifest["kind"] == "semantic_link_synapses"
    return manifest  # type: ignore[no-any-return]


@pytest.mark.asyncio
async def test_manifest_names_the_staged_discovery_state(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)

    ref = _pending_manifest(progress)["source_state_ref"]
    state = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]
    assert state["candidate_digests_format"] == 1
    assert state["source_token"] == _pending_manifest(progress)["source_token"]


@pytest.mark.asyncio
async def test_apply_resume_survives_recall_and_worker_traffic_while_paused(kind: str) -> None:
    expected = await _silent_edges(kind)
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    written_before_pause = list(storage.added_synapses)

    # recall: an anchor's metadata, and the reinforcement of an edge this stage had just written
    storage.touch_metadata(_candidates(kind)[0])
    storage.reinforce_synapse(written_before_pause[0])
    # the vector worker fills a neuron outside the working set
    storage.set_vector("ws-lazy", [0.5, 0.5])
    # a memory created in another brain
    storage.create_in_other_brain()

    report = await _resume(storage, progress)

    assert sorted(storage.added_synapses) == expected
    assert len(storage.added_synapses) == len(set(storage.added_synapses))
    assert report.semantic_synapses_created == len(expected)
    state = progress.strategy_state("semantic_link")
    assert state["phase"] == "semantic_link_complete"
    assert state["pending"] == []


@pytest.mark.asyncio
async def test_apply_resume_fails_when_a_candidate_vector_changed_while_paused(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    storage.set_vector(_candidates(kind)[0], [0.1, 0.9])

    with pytest.raises(ConsolidationProgressError, match="source fence failed"):
        await _resume(storage, progress)


@pytest.mark.asyncio
async def test_apply_resume_fails_when_a_candidates_synapse_vanished_while_paused(
    kind: str,
) -> None:
    storage = _world_with_lazy_neuron(kind)
    if kind == "bounded":
        # candidates 4-7 have no edges in this world; give one an edge so its deletion is a change
        from tests.unit.test_semantic_discovery_working_set import _edges

        storage.synapses.update({s.id: s for s in _edges([(4, 7)])})
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    storage.delete_synapse("ws-edge-000-005" if kind == "plain" else "ws-edge-004-007")

    with pytest.raises(ConsolidationProgressError, match="source fence failed"):
        await _resume(storage, progress)


@pytest.mark.asyncio
async def test_manifest_from_before_the_staged_reference_keeps_the_strict_path() -> None:
    storage = _world_with_lazy_neuron("plain")
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    manifest = _pending_manifest(progress)
    manifest.pop("source_state_ref")
    progress.strategy_state("semantic_link")["pending"] = [
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    ]
    storage.touch_metadata(
        _candidates("plain")[0]
    )  # harmless to the result, fatal to the strict fence

    with pytest.raises(ConsolidationProgressError):
        await _resume(storage, progress)


@pytest.mark.asyncio
async def test_an_edge_already_written_tolerates_what_reinforcement_changes(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    first = storage.added_synapses[0]
    # same pair, same type, same similarity: only what reinforcement rewrites differs
    storage.synapses[first] = replace(storage.synapses[first], weight=0.99, reinforced_count=7)
    _forget_the_durable_cursor(progress)

    report = await _resume(storage, progress)

    assert sorted(storage.added_synapses) == await _silent_edges(kind)
    assert len(set(storage.added_synapses)) == len(storage.added_synapses)
    assert report.semantic_synapses_created == len(storage.added_synapses)


@pytest.mark.asyncio
async def test_an_edge_that_is_a_different_pair_than_checkpointed_is_still_rejected(
    kind: str,
) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    first = storage.added_synapses[0]
    stored = storage.synapses[first]
    storage.synapses[first] = replace(
        stored, metadata={**stored.metadata, "cosine_similarity": 0.1234}
    )
    _forget_the_durable_cursor(progress)

    with pytest.raises(ConsolidationProgressError, match="changed after its checkpoint"):
        await _resume(storage, progress)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["source_token", "reference_time"])
async def test_apply_resume_rejects_a_staged_state_that_belongs_to_another_source(
    kind: str, field: str
) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    await _run_until_paused_in_apply(storage, progress)
    ref = _pending_manifest(progress)["source_state_ref"]
    storage.semantic_discovery_states[(ref["state_id"], ref["revision"])][field] = "elsewhere"

    with pytest.raises(ConsolidationProgressError, match=r"another source|unavailable"):
        await _resume(storage, progress)
