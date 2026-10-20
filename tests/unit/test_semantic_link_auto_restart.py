"""P6 (C1): a semantic_link discovery that finds its source changed restarts itself, boundedly.

Today every such change ends the run ``failed`` and blocks the strategies behind it until someone
clears the record by hand. With ``semantic_auto_restarts=N`` the engine starts the discovery over
with a fresh source token, at most N times per run, and only while no edge has been written (after
the first write the stage is committed work, not a computation to redo).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import pytest

from surreal_memory.engine import semantic_discovery as semantic_discovery_module
from surreal_memory.engine.consolidation import (
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationReport,
    ConsolidationStrategy,
)
from surreal_memory.engine.consolidation_progress import (
    ConsolidationPausedError,
    ConsolidationProgressError,
)
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceFenceUnavailableError,
)
from tests.unit.test_consolidation_mid_strategies_resume import _Progress
from tests.unit.test_semantic_discovery_working_set import _candidates, _config, _world
from tests.unit.test_semantic_link_apply_under_traffic import (
    _ApplyStorage,
    _fixture_embedding,  # noqa: F401 - autouse: embeddings on, apply checkpoints of one edge
    _world_with_lazy_neuron,
)


class _HookedProgress(_Progress):
    """Fires ``hook`` right after the checkpoint of ``hook_phase`` was recorded, ``fires`` times."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hook: Callable[[], None] | None = None
        self.hook_phase: str | None = None
        self.fires = 0
        self.fired = 0

    async def checkpoint(self, strategy: str, phase: str, **kwargs: Any) -> None:
        await super().checkpoint(strategy, phase, **kwargs)
        if self.hook and phase == self.hook_phase and self.fired < self.fires:
            self.fired += 1
            self.hook()


def _progress(**kwargs: Any) -> _HookedProgress:
    from tests.unit.test_semantic_discovery_working_set import _REFERENCE_TIME as REF

    progress = _HookedProgress("semantic_link", **kwargs)
    progress.reference_time = REF  # type: ignore[attr-defined]
    return progress


def _engine(storage: Any, progress: _Progress, restarts: int) -> ConsolidationEngine:
    engine = ConsolidationEngine(storage, ConsolidationConfig(), semantic_auto_restarts=restarts)
    engine._active_strategy = ConsolidationStrategy.SEMANTIC_LINK
    engine._progress_session = progress  # type: ignore[assignment]

    async def no_reactivation(_report: ConsolidationReport, _dry_run: bool) -> None:
        return None

    engine._reactivate_dormant = no_reactivation  # type: ignore[method-assign]
    return engine


async def _run(storage: Any, progress: _Progress, restarts: int) -> ConsolidationReport:
    storage.brain.config = _config()
    storage.progress = progress
    report = ConsolidationReport()
    await _engine(storage, progress, restarts)._semantic_link(report, dry_run=False)
    return report


def _state(progress: _Progress) -> dict[str, Any]:
    return progress.strategy_state("semantic_link")


def _phases(progress: _Progress) -> list[str]:
    return [write["phase"] for write in progress.writes]


def _break_a_candidate(storage: _ApplyStorage, kind: str) -> Callable[[], None]:
    """A write the fence must refuse: a candidate's content changes while the run is in flight."""
    counter = {"n": 0}

    def hook() -> None:
        counter["n"] += 1
        storage.update_neuron(_candidates(kind)[0], content=f"rewritten {counter['n']}")

    return hook


@pytest.fixture(params=["plain", "bounded"])
def kind(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "bounded":
        monkeypatch.setattr(semantic_discovery_module, "MAX_NEURONS_TO_LINK", 4)
    return str(request.param)


@pytest.mark.asyncio
async def test_a_changed_source_restarts_the_discovery_and_the_run_completes(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    progress.hook, progress.hook_phase, progress.fires = (
        _break_a_candidate(storage, kind),
        "semantic_link_discovery_similarity",
        1,
    )

    report = await _run(storage, progress, restarts=2)

    assert progress.fired == 1
    assert len(_state(progress)["auto_recovery"]) == 1
    assert _state(progress)["auto_recovery"][0]["attempt"] == 1
    assert _state(progress)["phase"] == "semantic_link_complete"
    assert storage.added_synapses and len(set(storage.added_synapses)) == len(
        storage.added_synapses
    )
    assert "semantic_link_restart_pending" in _phases(progress)
    assert len(report.extra["semantic_link_auto_restarts"]) == 1
    assert "restarted" in report.summary()


@pytest.mark.asyncio
async def test_the_limit_is_exact_two_restarts_then_the_run_fails(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    progress.hook, progress.hook_phase, progress.fires = (
        _break_a_candidate(storage, kind),
        "semantic_link_discovery_similarity",
        99,
    )

    with pytest.raises(ConsolidationProgressError, match="safe restart"):
        await _run(storage, progress, restarts=2)

    assert progress.fired == 3  # the original attempt and exactly two restarts
    assert len(_state(progress)["auto_recovery"]) == 2
    assert not storage.added_synapses


@pytest.mark.asyncio
async def test_zero_restarts_is_todays_behaviour(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress()
    progress.hook, progress.hook_phase, progress.fires = (
        _break_a_candidate(storage, kind),
        "semantic_link_discovery_similarity",
        99,
    )

    with pytest.raises(ConsolidationProgressError, match="safe restart"):
        await _run(storage, progress, restarts=0)

    assert progress.fired == 1
    assert "auto_recovery" not in _state(progress)
    assert "semantic_link_restart_pending" not in _phases(progress)


@pytest.mark.asyncio
async def test_a_fence_that_cannot_be_read_is_not_a_reason_to_restart() -> None:
    storage = _world_with_lazy_neuron("plain")
    original = storage.assert_semantic_source_unchanged
    calls = {"n": 0}

    async def unreadable(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] >= 3:
            raise SemanticSourceFenceUnavailableError("could not inspect neuron changefeed")
        await original(*args, **kwargs)

    storage.assert_semantic_source_unchanged = unreadable  # type: ignore[method-assign]
    progress = _progress()

    with pytest.raises(ConsolidationProgressError, match="safe restart"):
        await _run(storage, progress, restarts=2)

    assert "auto_recovery" not in _state(progress)


@pytest.mark.asyncio
async def test_no_restart_once_edges_have_been_written(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress(pause_phase="semantic_link_apply", pause_after_phase_calls=2)
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    written = list(storage.added_synapses)
    assert written
    storage.set_vector(_candidates(kind)[0], [0.1, 0.9])

    with pytest.raises(ConsolidationProgressError, match="source fence failed"):
        await _run(storage, progress, restarts=2)

    assert "auto_recovery" not in _state(progress)
    assert storage.added_synapses == written


@pytest.mark.asyncio
async def test_a_changed_source_before_the_first_edge_restarts_from_the_apply_stage(
    kind: str,
) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress(pause_phase="semantic_link_pending")
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    assert not storage.added_synapses
    storage.set_vector(_candidates(kind)[0], [0.1, 0.9])

    await _run(storage, progress, restarts=2)

    assert len(_state(progress)["auto_recovery"]) == 1
    assert _state(progress)["phase"] == "semantic_link_complete"
    assert storage.added_synapses


@pytest.mark.asyncio
async def test_a_resumed_discovery_that_finds_its_source_changed_starts_over_fresh(
    kind: str,
) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress(pause_phase="semantic_link_discovery_similarity")
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    storage.update_neuron(_candidates(kind)[0], content="rewritten while the run slept")

    await _run(storage, progress, restarts=2)

    # one restart was enough: the second attempt did not resume the stale snapshot
    assert len(_state(progress)["auto_recovery"]) == 1
    assert _state(progress)["phase"] == "semantic_link_complete"
    assert storage.added_synapses


@pytest.mark.asyncio
async def test_the_restart_count_survives_a_new_invocation(kind: str) -> None:
    storage = _world_with_lazy_neuron(kind)
    progress = _progress(pause_phase="semantic_link_restart_pending")
    progress.hook, progress.hook_phase, progress.fires = (
        _break_a_candidate(storage, kind),
        "semantic_link_discovery_similarity",
        2,
    )

    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)  # restart 1, paused right after recording it
    assert len(_state(progress)["auto_recovery"]) == 1

    await _run(storage, progress, restarts=2)  # fails again (restart 2), then succeeds

    assert len(_state(progress)["auto_recovery"]) == 2
    assert _state(progress)["phase"] == "semantic_link_complete"


def test_the_option_is_not_part_of_what_must_match_to_resume() -> None:
    assert "semantic_auto_restarts" not in asdict(ConsolidationConfig())
    assert (
        ConsolidationEngine(_world("plain"), semantic_auto_restarts=-3)._semantic_auto_restarts == 0
    )


def test_the_cli_exposes_the_option() -> None:
    import typer.main

    from surreal_memory.cli.main import app

    command = typer.main.get_command(app).commands["consolidate"]  # type: ignore[attr-defined]
    options = {opt: param for param in command.params for opt in param.opts}
    assert "--semantic-auto-restarts" in options
    assert options["--semantic-auto-restarts"].default == 0
