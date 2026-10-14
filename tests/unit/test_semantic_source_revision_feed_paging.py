"""Paging of the database changefeed in ``_show_table_changes`` (semantic-source fence).

SurrealDB 3.2.3 applies ``LIMIT`` to individual feed ENTRIES (versionstamp x table) and then merges entries that share a
versionstamp into one row, so a transaction touching two changefeed tables (e.g. DELETE neuron + cascaded synapse) makes a page
come back with LIMIT-1 rows although the feed goes on. Treating ``len(rows) < PAGE`` as the end of the feed silently truncated
the scan and the fence failed with "barrier was not found within the bounded changefeed scan" for as long as such a
transaction stayed in the 1-week retention (measured 2026-10-14, program smem-baza-odchudzenie-b, B1/R4).
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.storage.surrealdb import semantic_source_revision as ssr
from surreal_memory.storage.surrealdb.semantic_source_revision import (
    SurrealDBSemanticSourceRevisionMixin,
)


class _Feed(SurrealDBSemanticSourceRevisionMixin):
    """Fake database feed with the 3.2.3 LIMIT semantics: LIMIT counts entries, rows merge entries of one versionstamp."""

    def __init__(self, entries: list[tuple[int, str]]) -> None:
        self.entries = sorted(entries)  # (versionstamp, table)
        self.pages: list[int] = []

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        assert sql.startswith("SHOW CHANGES FOR DATABASE SINCE ")
        parts = sql.split()
        cursor, limit = int(parts[5]), int(parts[7])
        chosen = [e for e in self.entries if e[0] >= cursor][
            :limit
        ]  # inclusive SINCE, LIMIT on entries
        rows: dict[int, list[dict[str, Any]]] = {}
        for vs, table in chosen:
            rows.setdefault(vs, []).append({"update": {"id": f"{table}:r{vs}"}})
        out = [{"versionstamp": vs, "changes": ch} for vs, ch in rows.items()]
        self.pages.append(len(out))
        return out


def _feed_with_cross_table_row(n_before: int, n_after: int) -> tuple[_Feed, list[int]]:
    barrier = []
    entries: list[tuple[int, str]] = []
    vs = 100
    for i in range(n_before + n_after):
        vs += 1
        entries.append((vs, "semantic_source_barrier" if i % 3 == 0 else "neuron"))
        if entries[-1][1] == "semantic_source_barrier":
            barrier.append(vs)
        if i == n_before:  # one transaction = neuron + synapse (same versionstamp, two tables)
            vs += 1
            entries.append((vs, "neuron"))
            entries.append((vs, "synapse"))
    return _Feed(entries), barrier


@pytest.mark.asyncio
async def test_cross_table_transaction_does_not_truncate_the_scan() -> None:
    feed, barrier = _feed_with_cross_table_row(n_before=1500, n_after=2500)
    got = await feed._show_table_changes("semantic_source_barrier", 0, 100_000)
    assert [
        e["versionstamp"] for e in got
    ] == barrier  # every barrier entry, including those after the merged row
    assert (
        min(feed.pages[:-1]) < ssr._DATABASE_FEED_PAGE_SIZE
    )  # the fake really produced a short mid-feed page


@pytest.mark.asyncio
async def test_without_cross_table_row_all_barrier_entries_are_returned() -> None:
    entries = [
        (100 + i, "semantic_source_barrier" if i % 3 == 0 else "neuron") for i in range(3000)
    ]
    feed = _Feed(entries)
    got = await feed._show_table_changes("semantic_source_barrier", 0, 100_000)
    assert len(got) == sum(1 for _, t in entries if t == "semantic_source_barrier")


@pytest.mark.asyncio
async def test_limit_stops_the_scan_early_when_enough_entries_kept() -> None:
    feed, barrier = _feed_with_cross_table_row(n_before=1500, n_after=2500)
    got = await feed._show_table_changes("semantic_source_barrier", 0, 128)
    assert len(got) == 128 and [e["versionstamp"] for e in got] == barrier[:128]


@pytest.mark.asyncio
async def test_empty_feed_returns_empty_without_looping() -> None:
    assert await _Feed([])._show_table_changes("semantic_source_barrier", 0, 128) == []


@pytest.mark.asyncio
async def test_feed_end_is_detected_when_page_repeats_only_the_cursor_row() -> None:
    feed = _Feed([(101, "neuron"), (102, "semantic_source_barrier")])
    got = await feed._show_table_changes("semantic_source_barrier", 0, 128)
    assert [e["versionstamp"] for e in got] == [102]
    assert len(feed.pages) <= 3  # terminates: last page brings no new row
