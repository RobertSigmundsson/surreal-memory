"""The counters of a restarted semantic_link run come from its new discovery (review r2, D-r2-2).

``ConsolidationProgressSession.strategy_state`` hands out a copy, so a state read before a restart
is stale after it. The apply stage used to take its counters from that stale read: a restart from
the ``pending`` phase carried the old manifest's ``skipped`` count into the new run's report and
into the durable counters. The fakes elsewhere return the live dict and cannot see this, so this
test uses a progress fake that copies like the real session.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from surreal_memory.engine.consolidation_progress import ConsolidationPausedError
from tests.unit.test_semantic_discovery_working_set import _candidates
from tests.unit.test_semantic_link_apply_under_traffic import (
    _fixture_embedding,  # noqa: F401 - autouse: embeddings on, apply checkpoints of one edge
    _world_with_lazy_neuron,
)
from tests.unit.test_semantic_link_auto_restart import _HookedProgress, _run, _state


class _CopyProgress(_HookedProgress):
    """Like the real session: ``strategy_state`` returns a copy, ``checkpoint`` writes the live state."""

    def _live(self, strategy: str) -> dict[str, Any]:
        live: dict[str, Any] = self.state["strategy_states"].setdefault(strategy, {})
        return live

    def strategy_state(self, strategy: str) -> dict[str, Any]:
        return copy.deepcopy(self._live(strategy))

    async def checkpoint(
        self,
        strategy: str,
        phase: str,
        *,
        cursor: str | None = None,
        pending: list[str] | None = None,
        counters: dict[str, int | float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        state = self._live(strategy)
        state.update(
            phase=phase, cursor=cursor, pending=list(pending or []), counters=dict(counters or {})
        )
        if extra:
            state.update(extra)
        self.writes.append(dict(state))
        self.phase_calls[phase] = self.phase_calls.get(phase, 0) + 1
        if phase == self.pause_phase and self.phase_calls[phase] == self.pause_after_phase_calls:
            self.pause_phase = None
            raise ConsolidationPausedError("simulated interruption")


@pytest.mark.asyncio
async def test_a_restart_from_the_pending_phase_does_not_inherit_the_old_skipped_count() -> None:
    from tests.unit.test_semantic_discovery_working_set import _REFERENCE_TIME

    storage = _world_with_lazy_neuron("plain")
    progress = _CopyProgress("semantic_link", pause_phase="semantic_link_pending")
    progress.reference_time = _REFERENCE_TIME  # type: ignore[attr-defined]
    with pytest.raises(ConsolidationPausedError):
        await _run(storage, progress, restarts=2)
    inherited = int(_state(progress)["counters"]["semantic_synapses_skipped"])
    assert inherited > 0, "the scenario needs a manifest with skipped pairs to inherit"
    storage.set_vector(_candidates("plain")[0], [0.1, 0.9])

    report = await _run(storage, progress, restarts=2)

    assert len(_state(progress)["auto_recovery"]) == 1
    assert int(_state(progress)["counters"]["semantic_synapses_skipped"]) == 0
    assert report.semantic_synapses_skipped == 0
