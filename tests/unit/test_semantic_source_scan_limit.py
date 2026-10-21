"""A changefeed longer than the bounded scan means "this token cannot be verified", not "database fault".

Tolerated recall traffic now accumulates in the feed instead of aborting the run on its first event,
so a long run, or an apply stage resumed after days, can pass the scan bound (independent review r1,
P2). The engine restarts a discovery whose token expired and refuses to restart on an unreadable
fence; the bound must therefore raise the former, not the latter.
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb import semantic_source_revision as revision_module
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceFenceExpiredError,
    SemanticSourceFenceProgress,
)
from tests.unit.test_semantic_link_apply_under_traffic import (
    _fixture_embedding,  # noqa: F401 - autouse: embeddings on, apply checkpoints of one edge
    _world_with_lazy_neuron,
)
from tests.unit.test_semantic_link_auto_restart import _progress, _run, _state
from tests.unit.test_semantic_source_revision_tolerance import (
    OLD,
    _event,
    _neuron_record,
    _setup,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("incremental", [False, True])
async def test_a_feed_longer_than_the_scan_bound_expires_the_token(
    monkeypatch: pytest.MonkeyPatch, incremental: bool
) -> None:
    monkeypatch.setattr(revision_module, "_SOURCE_CHANGEFEED_MAX_ROWS", 5)
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [
        _event(101 + i, "update", _neuron_record(f"old{i}", created=OLD)) for i in range(8)
    ]

    with pytest.raises(SemanticSourceFenceExpiredError, match="scan limit"):
        await storage.assert_semantic_source_unchanged(
            token,
            created_before=cutoff,
            tolerate=lambda *_a: True,
            progress=(
                SemanticSourceFenceProgress(token=token, created_before=cutoff)
                if incremental
                else None
            ),
        )


@pytest.mark.asyncio
async def test_the_engine_restarts_a_discovery_whose_token_outgrew_the_scan_bound() -> None:
    storage = _world_with_lazy_neuron("plain")
    original = storage.assert_semantic_source_unchanged
    calls = {"n": 0}

    async def outgrown_once(*args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise SemanticSourceFenceExpiredError(
                "neuron changefeed exceeded the bounded scan limit"
            )
        await original(*args, **kwargs)

    storage.assert_semantic_source_unchanged = outgrown_once  # type: ignore[method-assign]
    progress = _progress()

    report = await _run(storage, progress, restarts=2)

    assert len(_state(progress)["auto_recovery"]) == 1
    assert report.semantic_synapses_created > 0
