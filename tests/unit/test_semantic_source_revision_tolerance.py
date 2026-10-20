"""P3 (W1a): optional tolerance filter and incremental memory for the semantic source fence.

Default arguments keep today's strict behaviour (the existing suite covers that); the tests here
pin the new contract: a caller-supplied predicate may tolerate a change the fence would otherwise
reject, decoding failures fail closed, legacy tokens stay strict, and the incremental memory gives
the same verdict as a full scan.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import pytest

from surreal_memory.core.neuron import Neuron
from surreal_memory.storage.surrealdb._ids import _to_public_id
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SemanticSourceChangedError,
    SemanticSourceFenceProgress,
    SemanticSourceFenceUnavailableError,
)
from tests.unit.test_semantic_source_revision import _RevisionStorage, _token


class _FeedStorage(_RevisionStorage):
    """The base fake ignores the SINCE cursor; this one honours it (inclusive) like SurrealDB."""

    async def _show_table_changes(
        self, table: str, since: int | str, limit: int
    ) -> list[dict[str, Any]]:
        self.queries.append((f"FEED {table} SINCE {since} LIMIT {limit}", {}))
        return [row for row in self.change_rows[table] if row["versionstamp"] >= int(since)][:limit]


def _neuron_record(
    rid: str, *, brain: str = "brain-a", created: datetime, content: str = "c", vec: Any = None
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": f"neuron:{rid}",
        "brain_id": brain,
        "type": "concept",
        "content": content,
        "created_at": created,
        "metadata": {"k": "v"},
    }
    if vec is not None:
        record["embedding_vec"] = vec
    return record


def _event(stamp: int, action: str, record: dict[str, Any]) -> dict[str, Any]:
    return {"versionstamp": stamp, "changes": [{action: record}]}


def _setup(readonly: bool = True) -> tuple[_FeedStorage, datetime, str]:
    storage = _FeedStorage()
    cutoff = storage.now
    token = _token(captured_at=cutoff - timedelta(seconds=1), created_at_readonly=readonly)
    return storage, cutoff, token


OLD = datetime(2026, 9, 1, 0, 0)


@pytest.mark.asyncio
async def test_predicate_true_tolerates_a_pre_reference_update_and_false_does_not() -> None:
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [_event(101, "update", _neuron_record("old", created=OLD))]
    await storage.assert_semantic_source_unchanged(
        token, created_before=cutoff, tolerate=lambda *_a: True
    )
    with pytest.raises(SemanticSourceChangedError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, tolerate=lambda *_a: False
        )
    with pytest.raises(SemanticSourceChangedError):  # default stays strict
        await storage.assert_semantic_source_unchanged(token, created_before=cutoff)


@pytest.mark.asyncio
async def test_predicate_receives_public_id_brain_and_decoded_domain_object() -> None:
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [
        _event(
            101, "update", _neuron_record("old_node", created=OLD, content="hello", vec=[0.5, 0.25])
        ),
        _event(102, "delete", {"id": "neuron:gone_node"}),
    ]
    seen: list[tuple[str, str, str, str | None, Any]] = []

    def predicate(table: str, action: str, public_id: str, brain: str | None, domain: Any) -> bool:
        seen.append((table, action, public_id, brain, domain))
        return True

    await storage.assert_semantic_source_unchanged(token, created_before=cutoff, tolerate=predicate)
    assert [(t, a) for t, a, *_ in seen] == [("neuron", "update"), ("neuron", "delete")]
    table, action, public_id, brain, domain = seen[0]
    assert public_id == _to_public_id("neuron:old_node") and brain == "brain-a"
    assert isinstance(domain, Neuron) and domain.content == "hello"
    assert domain.metadata["_embedding"] == [0.5, 0.25]
    assert (
        seen[1][2] == _to_public_id("neuron:gone_node")
        and seen[1][3] is None
        and seen[1][4] is None
    )


@pytest.mark.asyncio
async def test_a_neuron_created_in_another_brain_aborts_the_strict_fence_but_can_be_tolerated() -> (
    None
):
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [
        _event(
            101,
            "create",
            _neuron_record("other", brain="brain-b", created=cutoff + timedelta(seconds=5)),
        )
    ]
    with pytest.raises(SemanticSourceChangedError):
        await storage.assert_semantic_source_unchanged(token, created_before=cutoff)
    seen: list[str | None] = []

    def predicate(_t: str, _a: str, _i: str, brain: str | None, _d: Any) -> bool:
        seen.append(brain)
        return True

    await storage.assert_semantic_source_unchanged(token, created_before=cutoff, tolerate=predicate)
    assert seen == ["brain-b"]


@pytest.mark.asyncio
async def test_legacy_tokens_stay_strict_even_when_the_predicate_tolerates_everything() -> None:
    storage, cutoff, token = _setup(readonly=False)
    storage.change_rows["neuron"] = [_event(101, "update", _neuron_record("old", created=OLD))]
    with pytest.raises(SemanticSourceChangedError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, tolerate=lambda *_a: True
        )


@pytest.mark.asyncio
async def test_unexpected_table_definition_fails_closed_despite_tolerance() -> None:
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [
        _event(
            101,
            "define_table",
            {"name": "neuron", "changefeed": {"expiry": "1d", "original": True}},
        )
    ]
    with pytest.raises(SemanticSourceFenceUnavailableError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, tolerate=lambda *_a: True
        )


@pytest.mark.asyncio
async def test_a_record_that_cannot_be_decoded_fails_closed() -> None:
    storage, cutoff, token = _setup()
    broken = {"id": "neuron:old", "brain_id": "brain-a", "created_at": OLD}  # no type/content
    storage.change_rows["neuron"] = [_event(101, "update", broken)]
    with pytest.raises(SemanticSourceFenceUnavailableError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, tolerate=lambda *_a: True
        )


@pytest.mark.asyncio
async def test_tolerance_and_progress_require_a_frozen_cutoff_and_matching_token() -> None:
    storage, cutoff, token = _setup()
    with pytest.raises(ValueError):
        await storage.assert_semantic_source_unchanged(token, tolerate=lambda *_a: True)
    mismatched = SemanticSourceFenceProgress(token="a-different-token", created_before=cutoff)  # noqa: S106
    with pytest.raises(ValueError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, progress=mismatched
        )


@pytest.mark.asyncio
async def test_progress_remembers_post_reference_ids_and_reads_only_new_events() -> None:
    storage, cutoff, token = _setup()
    late = cutoff + timedelta(seconds=3)
    storage.change_rows["neuron"] = [_event(101, "create", _neuron_record("late", created=late))]
    progress = SemanticSourceFenceProgress(token=token, created_before=cutoff)
    await storage.assert_semantic_source_unchanged(token, created_before=cutoff, progress=progress)
    storage.change_rows["neuron"].append(
        _event(102, "update", _neuron_record("late", created=late))
    )
    storage.queries.clear()
    await storage.assert_semantic_source_unchanged(token, created_before=cutoff, progress=progress)
    feeds = [q for q, _p in storage.queries if q.startswith("FEED neuron")]
    assert feeds and all(
        "SINCE 101" in q or "SINCE 102" in q for q in feeds
    )  # resumed, not from the token
    # Without the memory the second call is a fresh scan from the token and sees the same history:
    await storage.assert_semantic_source_unchanged(token, created_before=cutoff)


@pytest.mark.asyncio
async def test_progress_does_not_advance_when_the_scan_fails() -> None:
    storage, cutoff, token = _setup()
    storage.change_rows["neuron"] = [_event(101, "update", _neuron_record("old", created=OLD))]
    progress = SemanticSourceFenceProgress(token=token, created_before=cutoff)
    with pytest.raises(SemanticSourceChangedError):
        await storage.assert_semantic_source_unchanged(
            token, created_before=cutoff, progress=progress
        )
    assert progress.cursors == {} and progress.post_reference_ids == {}


def _random_feed(rng: random.Random, cutoff: datetime) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    known_late: list[str] = []
    for stamp in range(101, 101 + rng.randint(1, 14)):
        roll = rng.random()
        if roll < 0.35 or (not known_late and roll < 0.6):
            rid = f"late{stamp}"
            known_late.append(rid)
            events.append(
                _event(
                    stamp, "create", _neuron_record(rid, created=cutoff + timedelta(seconds=stamp))
                )
            )
        elif roll < 0.55 and known_late:
            rid = rng.choice(known_late)
            events.append(
                _event(
                    stamp, "update", _neuron_record(rid, created=cutoff + timedelta(seconds=stamp))
                )
            )
        elif roll < 0.7 and known_late:
            events.append(_event(stamp, "delete", {"id": f"neuron:{rng.choice(known_late)}"}))
        elif roll < 0.8:
            events.append(_event(stamp, "update", _neuron_record(f"old{stamp}", created=OLD)))
        elif roll < 0.9:
            events.append(_event(stamp, "delete", {"id": f"neuron:ghost{stamp}"}))
        else:
            events.append(
                _event(
                    stamp,
                    "create",
                    _neuron_record(
                        f"other{stamp}", brain="brain-b", created=cutoff + timedelta(seconds=stamp)
                    ),
                )
            )
    return events


async def _verdict(storage: _FeedStorage, token: str, cutoff: datetime, **kwargs: Any) -> str:
    try:
        await storage.assert_semantic_source_unchanged(token, created_before=cutoff, **kwargs)
    except SemanticSourceChangedError:
        return "changed"
    except SemanticSourceFenceUnavailableError:
        return "unavailable"
    return "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("tolerant", [False, True])
async def test_incremental_memory_gives_the_same_verdict_as_a_full_scan(tolerant: bool) -> None:
    """Differential test: split the feed at a random point, run with memory, compare with one full scan."""
    rng = random.Random(20261020 + int(tolerant))
    seen_verdicts: set[str] = set()
    for _ in range(200):
        storage, cutoff, token = _setup()
        feed = _random_feed(rng, cutoff)
        extra: dict[str, Any] = (
            {"tolerate": (lambda t, a, i, b, d: i.startswith("old"))} if tolerant else {}
        )
        storage.change_rows["neuron"] = feed
        full = await _verdict(storage, token, cutoff, **extra)
        split = rng.randint(0, len(feed))
        storage.change_rows["neuron"] = feed[:split]
        progress = SemanticSourceFenceProgress(token=token, created_before=cutoff)
        first = await _verdict(storage, token, cutoff, progress=progress, **extra)
        if first == "pass":
            storage.change_rows["neuron"] = feed
            second = await _verdict(storage, token, cutoff, progress=progress, **extra)
        else:
            second = first
        assert second == full, (feed, split, full, second)
        seen_verdicts.add(full)
    assert {"pass", "changed"} <= seen_verdicts  # the generator exercises both outcomes
