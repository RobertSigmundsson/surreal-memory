"""P4: the semantic discovery run tolerates changes that provably cannot alter its result.

The run's working set is the candidates it selected; what it reads from them is exactly
(id, type, content, vector). A recall rewriting a candidate's metadata, a worker filling another
neuron's vector, a neuron created in another brain, a reinforced synapse - none of those can change
the pairs it computes, so the source fence must not abort the run for them. A change that CAN alter
the result (content, vector, deletion, a vanished synapse of a candidate) must still be caught.
"""

from __future__ import annotations

import json
from dataclasses import replace as dc_replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.semantic_discovery import (
    SemanticDiscoveryResult,
    SemanticSourceStaleError,
    _neuron_projection,
    discover_semantic_synapses,
)
from surreal_memory.storage.surrealdb.semantic_source_revision import SemanticSourceChangedError
from tests.unit.test_consolidation_mid_strategies_resume import _Storage

_REFERENCE_TIME = datetime(2026, 1, 1, tzinfo=UTC)
_OLD = _REFERENCE_TIME - timedelta(days=1)
_BRAIN = "brain-1"


def _aware(moment: datetime) -> datetime:
    """The engine mixes naive (``utcnow``) and aware timestamps; the real store normalizes both."""
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


class _FeedStorage(_Storage):
    """_Storage plus a faithful miniature of the changefeed fence (order of checks as in prod)."""

    def __init__(
        self, *, neurons: list[Neuron], synapses: list[Synapse] | None = None, degrees: bool = True
    ) -> None:
        super().__init__(neurons=neurons, synapses=synapses)
        self.feed: list[dict[str, Any]] = []
        self.calls: list[tuple[bool, bool, int]] = []  # (tolerant, incremental, events examined)
        self.degree_calls = 0
        if not degrees:
            self.get_synapse_degrees = None  # type: ignore[assignment]

    # -- mutations a live system performs ---------------------------------------------------
    def _log(self, table: str, action: str, record_id: str, brain: str | None, domain: Any) -> None:
        created = getattr(domain, "created_at", _OLD)
        self.feed.append(
            {
                "table": table,
                "action": action,
                "id": record_id,
                "brain": brain,
                "domain": domain,
                "created_at": created,
            }
        )

    def update_neuron(self, neuron_id: str, **changes: Any) -> None:
        neuron = dc_replace(self.neurons[neuron_id], **changes)
        self.neurons[neuron_id] = neuron
        self._log("neuron", "update", neuron_id, _BRAIN, neuron)

    def touch_metadata(self, neuron_id: str) -> None:
        """What a recall does to an anchor: bump counters inside metadata, vector untouched."""
        metadata = dict(self.neurons[neuron_id].metadata)
        metadata["_reconsolidation_count"] = int(metadata.get("_reconsolidation_count", 0)) + 1
        self.update_neuron(neuron_id, metadata=metadata)

    def set_vector(self, neuron_id: str, vector: list[float]) -> None:
        metadata = dict(self.neurons[neuron_id].metadata)
        metadata["_embedding"] = vector
        self.update_neuron(neuron_id, metadata=metadata)

    def delete_neuron(self, neuron_id: str) -> None:
        del self.neurons[neuron_id]
        self._log("neuron", "delete", neuron_id, None, None)

    def reinforce_synapse(self, synapse_id: str) -> None:
        synapse = dc_replace(self.synapses[synapse_id], weight=0.99)
        self.synapses[synapse_id] = synapse
        self._log("synapse", "update", synapse_id, _BRAIN, synapse)

    def delete_synapse(self, synapse_id: str) -> None:
        del self.synapses[synapse_id]
        self._log("synapse", "delete", synapse_id, None, None)

    def create_in_other_brain(self) -> None:
        stranger = dc_replace(
            Neuron.create(
                NeuronType.CONCEPT, "someone else's memory", neuron_id="other-brain-node"
            ),
            created_at=_REFERENCE_TIME + timedelta(seconds=5),
        )
        self._log("neuron", "create", stranger.id, "brain-2", stranger)

    # -- the fence --------------------------------------------------------------------------
    async def capture_semantic_source_token(self) -> str:
        return str(len(self.feed))

    async def assert_semantic_source_unchanged(
        self,
        token: str,
        *,
        created_before: datetime | None = None,
        tolerate: Any = None,
        progress: Any = None,
    ) -> None:
        start = progress.cursors.get("feed", int(token)) if progress is not None else int(token)
        examined = 0
        try:
            for event in self.feed[start:]:
                examined += 1
                own = event["brain"] == _BRAIN
                pre_reference = created_before is None or _aware(event["created_at"]) <= _aware(
                    created_before
                )
                if event["action"] != "delete" and own and not pre_reference:
                    continue  # a record created after the reference time: outside the frozen set
                if tolerate is not None and tolerate(
                    event["table"], event["action"], event["id"], event["brain"], event["domain"]
                ):
                    continue
                raise SemanticSourceChangedError(
                    f"{event['table']} changed since the semantic source token was captured"
                )
            if progress is not None:
                progress.cursors["feed"] = len(self.feed)
        finally:
            self.calls.append((tolerate is not None, progress is not None, examined))

    async def get_synapses_after_id(
        self, cursor_id: str | None, *, limit: int, created_before: datetime | None = None
    ) -> list[Synapse]:
        found = sorted(self.synapses.values(), key=lambda edge: edge.id)
        if created_before is not None:
            found = [e for e in found if _aware(e.created_at) <= _aware(created_before)]
        if cursor_id is not None:
            found = [edge for edge in found if edge.id > cursor_id]
        return found[:limit]

    async def find_existing_synapse_pairs(
        self, pairs: list[tuple[str, str]], *, created_before: datetime | None = None
    ) -> set[tuple[str, str]]:
        requested = {tuple(sorted(pair)) for pair in pairs}
        found: set[tuple[str, str]] = set()
        for synapse in self.synapses.values():
            if created_before is not None and _aware(synapse.created_at) > _aware(created_before):
                continue
            key = tuple(sorted((synapse.source_id, synapse.target_id)))
            if key in requested:
                found.add(key)  # type: ignore[arg-type]
        return found

    async def get_synapse_degrees(
        self, *, created_before: datetime | None = None
    ) -> dict[str, int]:
        self.degree_calls += 1
        degrees: dict[str, int] = {}
        for synapse in self.synapses.values():
            if created_before is not None and _aware(synapse.created_at) > _aware(created_before):
                continue
            for endpoint in (synapse.source_id, synapse.target_id):
                degrees[endpoint] = degrees.get(endpoint, 0) + 1
        return degrees


class _LegacyFenceStorage(_FeedStorage):
    """A store whose fence predates the tolerance filter (no ``tolerate``/``progress``)."""

    async def assert_semantic_source_unchanged(  # type: ignore[override]
        self, token: str, *, created_before: datetime | None = None
    ) -> None:
        await super().assert_semantic_source_unchanged(token, created_before=created_before)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        embedding_enabled=True,
        embedding_provider="fixture",
        embedding_model="fixture",
        semantic_discovery_similarity_threshold=0.7,
        semantic_discovery_max_pairs=50,
        essence_generator="extractive",
    )


def _neurons(count: int) -> list[Neuron]:
    return [
        dc_replace(
            Neuron.create(
                NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
                f"working set candidate {index}",
                metadata={"_embedding": [1.0 - index * 0.01, index * 0.01]},
                neuron_id=f"ws-node-{index:03d}",
            ),
            created_at=_OLD,
        )
        for index in range(count)
    ]


def _edges(pairs: list[tuple[int, int]]) -> list[Synapse]:
    return [
        dc_replace(
            Synapse.create(
                f"ws-node-{a:03d}",
                f"ws-node-{b:03d}",
                SynapseType.RELATED_TO,
                synapse_id=f"ws-edge-{a:03d}-{b:03d}",
            ),
            created_at=_OLD,
        )
        for a, b in pairs
    ]


@pytest.fixture(autouse=True)
def _fixture_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "surreal_memory.engine.semantic_discovery._effective_embedding",
        lambda _config: (True, "fixture", "fixture"),
    )


def _world(
    kind: str = "plain", *, storage_cls: type[_FeedStorage] = _FeedStorage, degrees: bool = True
) -> _FeedStorage:
    """``plain``: all 6 neurons are candidates. ``bounded``: 8 neurons, cap 4, degree ranking."""
    if kind == "bounded":
        storage = storage_cls(
            neurons=_neurons(8),
            # nodes 0-3 are well connected, so the four least connected (4-7) are the candidates
            synapses=_edges([(0, 1), (1, 2), (2, 3), (0, 3), (0, 2), (1, 3)]),
            degrees=degrees,
        )
    else:
        storage = storage_cls(neurons=_neurons(6), synapses=_edges([(0, 5)]), degrees=degrees)
    storage.brain.config = _config()
    return storage


@pytest.fixture(params=["plain", "bounded"])
def kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "bounded":
        monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)
    return str(request.param)


def _candidates(kind: str) -> list[str]:
    return [f"ws-node-{i:03d}" for i in (range(4, 8) if kind == "bounded" else range(6))]


def _outsiders(kind: str) -> list[str]:
    return [f"ws-node-{i:03d}" for i in range(4)] if kind == "bounded" else []


async def _discover(
    storage: _FeedStorage,
    *,
    during: dict[str, Any] | None = None,
    resume_state: dict[str, Any] | None = None,
    budget_check: Any = None,
) -> tuple[SemanticDiscoveryResult, list[tuple[str, str | None, dict[str, Any]]]]:
    """Run staged discovery; ``during`` maps a checkpoint stage to a mutation fired at its first emit."""
    seen: list[tuple[str, str | None, dict[str, Any]]] = []
    fired: set[str] = set()

    async def checkpoint(stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        seen.append((stage, cursor, json.loads(json.dumps(details))))
        if during and stage in during and stage not in fired:
            fired.add(stage)
            during[stage]()

    result = await discover_semantic_synapses(
        storage,
        _config(),
        checkpoint=checkpoint,
        resume_state=resume_state,
        budget_check=budget_check,
        reference_time=_REFERENCE_TIME,
    )
    return result, seen


def _pairs(result: SemanticDiscoveryResult) -> list[tuple[str, str, float]]:
    return sorted(
        (s.source_id, s.target_id, float(s.metadata["cosine_similarity"])) for s in result.synapses
    )


async def _silent_pairs(kind: str) -> list[tuple[str, str, float]]:
    result, _ = await _discover(_world(kind))
    pairs = _pairs(result)
    assert pairs, "the fixture must produce semantic pairs or the equality below proves nothing"
    return pairs


# ---------------------------------------------------------------------------------------------
# tolerated: changes that cannot alter the result
# ---------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["neurons", "similarity"])
async def test_recall_like_update_of_a_candidate_does_not_abort_the_run(
    kind: str, stage: str
) -> None:
    expected = await _silent_pairs(kind)
    storage = _world(kind)
    victim = _candidates(kind)[0]

    def recall() -> None:
        storage.touch_metadata(victim)
        storage.reinforce_synapse("ws-edge-000-005" if kind == "plain" else "ws-edge-000-001")

    result, _ = await _discover(storage, during={stage: recall})

    assert _pairs(result) == expected
    assert any(tolerant for tolerant, _incremental, _n in storage.calls)


@pytest.mark.asyncio
async def test_content_change_of_a_neuron_outside_the_working_set_is_tolerated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)
    kind = "bounded"
    outsider = _outsiders(kind)[0]
    expected = await _silent_pairs(kind)
    storage = _world(kind)

    # fired after the neuron scan: the outsider was read and dropped, so nothing depends on it
    result, _ = await _discover(
        storage, during={"similarity": lambda: storage.update_neuron(outsider, content="rewritten")}
    )

    assert _pairs(result) == expected


@pytest.mark.asyncio
async def test_a_neuron_created_in_another_brain_does_not_abort_the_run(kind: str) -> None:
    expected = await _silent_pairs(kind)
    storage = _world(kind)

    result, _ = await _discover(storage, during={"similarity": storage.create_in_other_brain})

    assert _pairs(result) == expected


@pytest.mark.asyncio
async def test_rewrite_that_sets_the_same_values_is_not_a_change(kind: str) -> None:
    storage = _world(kind)
    victim = _candidates(kind)[1]
    result, _ = await _discover(
        storage,
        during={
            "similarity": lambda: storage.update_neuron(
                victim, content=storage.neurons[victim].content
            )
        },
    )
    assert _pairs(result) == await _silent_pairs(kind)


@pytest.mark.asyncio
async def test_change_before_the_neuron_is_read_is_simply_read(kind: str) -> None:
    storage = _world(kind)
    victim = _candidates(kind)[2]
    # fired before the first page is read: the run reads the new value, there is nothing stale
    result, _ = await _discover(
        storage, during={"neurons": lambda: storage.update_neuron(victim, content="reworded early")}
    )
    assert result.synapses


# ---------------------------------------------------------------------------------------------
# NOT tolerated: changes that can alter the result must still stop the run
# ---------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["neurons", "similarity"])
async def test_content_change_of_a_candidate_is_detected(kind: str, stage: str) -> None:
    storage = _world(kind)
    victim = _candidates(kind)[0]
    # a neuron read at least once is part of the set; the scan is not "before read" any more
    with pytest.raises(SemanticSourceChangedError):
        await _discover(
            storage,
            during={"similarity": lambda: storage.update_neuron(victim, content="changed meaning")},
        )


@pytest.mark.asyncio
async def test_content_change_of_a_candidate_between_pages_is_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 2)
    storage = _world("plain")
    budget_calls = 0

    async def budget() -> None:
        nonlocal budget_calls
        budget_calls += 1
        if budget_calls == 2:  # the second page is about to be fetched; node 000 is on the first
            storage.update_neuron("ws-node-000", content="changed after it was read")

    with pytest.raises(SemanticSourceChangedError):
        await _discover(storage, budget_check=budget)


@pytest.mark.asyncio
async def test_vector_change_of_a_candidate_is_detected(kind: str) -> None:
    storage = _world(kind)
    victim = _candidates(kind)[0]
    with pytest.raises(SemanticSourceChangedError):
        await _discover(
            storage, during={"similarity": lambda: storage.set_vector(victim, [0.1, 0.9])}
        )


@pytest.mark.asyncio
async def test_deleting_a_candidate_is_detected(kind: str) -> None:
    storage = _world(kind)
    victim = _candidates(kind)[0]
    with pytest.raises(SemanticSourceChangedError):
        await _discover(storage, during={"similarity": lambda: storage.delete_neuron(victim)})


@pytest.mark.asyncio
async def test_a_vanished_synapse_of_a_candidate_is_caught_by_the_degree_guard(kind: str) -> None:
    storage = _world(kind)
    edge = "ws-edge-000-005" if kind == "plain" else None
    if edge is None:
        # in the bounded world the candidates 4-7 have no edges; connect one so a delete matters
        storage.synapses.update({s.id: s for s in _edges([(4, 7)])})
        edge = "ws-edge-004-007"
    with pytest.raises(SemanticSourceStaleError):
        await _discover(storage, during={"similarity": lambda: storage.delete_synapse(edge)})


@pytest.mark.asyncio
async def test_without_degree_support_synapse_changes_stay_strict() -> None:
    storage = _world("plain", degrees=False)
    with pytest.raises(SemanticSourceChangedError):
        await _discover(
            storage, during={"similarity": lambda: storage.reinforce_synapse("ws-edge-000-005")}
        )
    assert storage.degree_calls == 0


@pytest.mark.asyncio
async def test_a_fence_without_tolerance_support_keeps_the_strict_behaviour() -> None:
    storage = _world("plain", storage_cls=_LegacyFenceStorage)
    victim = "ws-node-001"
    with pytest.raises(SemanticSourceChangedError):
        await _discover(storage, during={"similarity": lambda: storage.touch_metadata(victim)})
    # and nothing of the new machinery leaks into what it persists
    quiet = _world("plain", storage_cls=_LegacyFenceStorage)
    await _discover(quiet)
    assert all(
        "candidate_digests" not in state for state in quiet.semantic_discovery_states.values()
    )
    assert quiet.degree_calls == 0


# ---------------------------------------------------------------------------------------------
# snapshots, resume, the final gate
# ---------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_snapshot_carries_one_digest_per_candidate_in_candidate_order(kind: str) -> None:
    storage = _world(kind)
    _result, seen = await _discover(storage)
    stage, _cursor, details = [item for item in seen if item[0] == "similarity"][-1]
    ref = details["source_state_ref"]
    state = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]

    assert state["candidate_digests_format"] == 1
    blob = state["candidate_digests"]
    assert len(blob) == 16 * len(state["candidates"])
    for index, (candidate_id, _priority, _degree) in enumerate(state["candidates"]):
        expected = _neuron_projection(storage.neurons[candidate_id])
        assert blob[16 * index : 16 * (index + 1)] == expected


async def _pause_after_neuron_scan(storage: _FeedStorage) -> dict[str, Any]:
    """Run until the first similarity row is about to be scored; return the last manifest."""
    from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

    seen: list[dict[str, Any]] = []

    async def checkpoint(_stage: str, _cursor: str | None, details: dict[str, Any]) -> None:
        seen.append(json.loads(json.dumps(details)))

    async def pause() -> None:
        if seen and seen[-1]["stage"] == "similarity":
            raise ConsolidationPausedError("simulated interruption")

    with pytest.raises(ConsolidationPausedError):
        await discover_semantic_synapses(
            storage,
            _config(),
            checkpoint=checkpoint,
            resume_state=None,
            budget_check=pause,
            reference_time=_REFERENCE_TIME,
        )
    return seen[-1]


async def _resume(storage: _FeedStorage, manifest: dict[str, Any]) -> SemanticDiscoveryResult:
    result, _ = await _discover(storage, resume_state=manifest)
    return result


@pytest.mark.asyncio
async def test_resume_tolerates_a_recall_between_the_pause_and_the_restart(kind: str) -> None:
    expected = await _silent_pairs(kind)
    storage = _world(kind)
    manifest = await _pause_after_neuron_scan(storage)
    storage.touch_metadata(_candidates(kind)[0])
    storage.create_in_other_brain()

    result = await _resume(storage, manifest)

    assert _pairs(result) == expected


@pytest.mark.asyncio
async def test_resume_restarts_when_a_candidate_changed_while_paused(kind: str) -> None:
    storage = _world(kind)
    manifest = await _pause_after_neuron_scan(storage)
    storage.update_neuron(_candidates(kind)[0], content="rewritten while the run slept")

    with pytest.raises(SemanticSourceChangedError):
        await _resume(storage, manifest)


@pytest.mark.asyncio
async def test_restored_candidate_whose_digest_differs_is_stale(kind: str) -> None:
    storage = _world(kind)
    manifest = await _pause_after_neuron_scan(storage)
    # a write the feed never reported (a gap in the fence): the digest check must still catch it
    victim = _candidates(kind)[0]
    storage.neurons[victim] = dc_replace(storage.neurons[victim], content="silent rewrite")

    with pytest.raises(SemanticSourceStaleError):
        await _resume(storage, manifest)


@pytest.mark.asyncio
async def test_snapshot_from_before_the_digests_resumes_strictly(kind: str) -> None:
    """A run in flight when the new code is deployed has no digests; its first check is strict."""
    storage = _world(kind)
    manifest = await _pause_after_neuron_scan(storage)
    ref = manifest["source_state_ref"]
    state = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]
    state.pop("candidate_digests")
    state.pop("candidate_digests_format")
    state.pop("candidate_degree_digest", None)

    quiet = await _resume(storage, manifest)  # nothing happened since the token: it just finishes
    assert quiet.synapses

    again = _world(kind)
    manifest2 = await _pause_after_neuron_scan(again)
    ref2 = manifest2["source_state_ref"]
    state2 = again.semantic_discovery_states[(ref2["state_id"], ref2["revision"])]
    state2.pop("candidate_digests")
    state2.pop("candidate_digests_format")
    again.touch_metadata(_candidates(kind)[0])
    with pytest.raises(SemanticSourceChangedError):
        await _resume(again, manifest2)


@pytest.mark.asyncio
async def test_progress_makes_the_in_run_checks_incremental_and_the_final_gate_full() -> None:
    storage = _world("plain")

    def twenty_recalls() -> None:
        for _ in range(20):
            storage.touch_metadata("ws-node-002")

    await _discover(storage, during={"neurons": twenty_recalls})

    incremental = [examined for _tolerant, inc, examined in storage.calls if inc]
    full = [examined for _tolerant, inc, examined in storage.calls if not inc]
    # the first in-run check reads the 20 new entries; every later one only what was added since
    assert incremental[0] == 20
    assert all(examined == 0 for examined in incremental[1:])
    # the last verification is a full scan from the token, whatever the memory says
    assert full and full[-1] == 20


@pytest.mark.asyncio
async def test_result_names_the_staged_state_it_was_computed_from(kind: str) -> None:
    storage = _world(kind)
    result, seen = await _discover(storage)
    ref = [item for item in seen if item[0] == "similarity"][-1][2]["source_state_ref"]
    assert result.source_state_id == ref["state_id"]
    assert result.source_state_revision == ref["revision"]


# ---------------------------------------------------------------------------------------------
# the projection itself
# ---------------------------------------------------------------------------------------------
def _neuron(**overrides: Any) -> Neuron:
    base = Neuron.create(
        NeuronType.CONCEPT,
        "projection subject",
        metadata={"_embedding": [0.25, 0.5, 0.75], "_reconsolidation_count": 1},
        neuron_id="projection-node",
    )
    return dc_replace(base, **overrides)


def test_projection_ignores_everything_a_recall_rewrites() -> None:
    base = _neuron()
    noisy = _neuron(
        metadata={
            "_embedding": [0.25, 0.5, 0.75],
            "_reconsolidation_count": 9,
            "_context_trail": ["x"],
        },
        created_at=base.created_at + timedelta(days=3),
        content_hash=123,
    )
    assert _neuron_projection(noisy) == _neuron_projection(base)


@pytest.mark.parametrize(
    "overrides",
    [
        {"content": "projection subject!"},
        {"type": NeuronType.ENTITY},
        {"id": "projection-other"},
        {"metadata": {"_embedding": [0.25, 0.5, 0.7500001]}},
        {"metadata": {}},
        {"metadata": {"_embedding": [0.25, 0.5]}},
    ],
)
def test_projection_changes_with_anything_discovery_reads(overrides: dict[str, Any]) -> None:
    assert _neuron_projection(_neuron(**overrides)) != _neuron_projection(_neuron())


def test_projection_does_not_depend_on_how_the_vector_was_decoded() -> None:
    as_list = _neuron()
    as_tuple = _neuron(metadata={"_embedding": (0.25, 0.5, 0.75)})
    as_ints = _neuron(metadata={"_embedding": [1, 2, 3]})
    as_floats = _neuron(metadata={"_embedding": [1.0, 2.0, 3.0]})
    assert _neuron_projection(as_list) == _neuron_projection(as_tuple)
    assert _neuron_projection(as_ints) == _neuron_projection(as_floats)


@pytest.mark.asyncio
async def test_snapshot_with_digests_stays_inside_the_state_budget_for_ten_thousand_long_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digests ride in the same immutable revision; the byte cap must still hold at the limit."""
    from surreal_memory.engine.consolidation_progress import ConsolidationPausedError

    neurons = [
        dc_replace(
            Neuron.create(
                NeuronType.CONCEPT if index % 2 == 0 else NeuronType.ENTITY,
                f"long candidate {index}",
                metadata={"_embedding": [1.0, 0.0]},
                neuron_id=f"candidate-{index:05d}-{'x' * 40}",
            ),
            created_at=_OLD,
        )
        for index in range(10_001)
    ]
    monkeypatch.setattr(semantic_discovery_module, "_EMBEDDING_PAGE_SIZE", 10_000)
    storage = _FeedStorage(neurons=neurons)
    storage.brain.config = _config()
    durable: dict[str, Any] = {}

    async def checkpoint(_stage: str, cursor: str | None, details: dict[str, Any]) -> None:
        nonlocal durable
        durable = details | {"cursor": cursor}

    calls = 0

    async def one_page_budget() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConsolidationPausedError("simulated candidate state budget")

    with pytest.raises(ConsolidationPausedError):
        await discover_semantic_synapses(
            storage,
            _config(),
            checkpoint=checkpoint,
            resume_state=None,
            budget_check=one_page_budget,
            reference_time=_REFERENCE_TIME,
        )

    ref = durable["source_state_ref"]
    state = storage.semantic_discovery_states[(ref["state_id"], ref["revision"])]
    assert len(state["candidates"]) == 10_000
    assert len(state["candidate_digests"]) == 16 * 10_000
    size = len(json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode())
    assert size < semantic_discovery_module._MAX_REBUILD_STATE_BYTES, size
