"""Unit tests for the SurrealDB tool-events mixin (no live DB required).

Regression coverage: the SurrealDB backend lacked any tool-event surface, so
consolidation's process_tool_events strategy raised
``AttributeError: 'SurrealDBStorage' object has no attribute 'get_unprocessed_events'``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from surreal_memory.storage.surrealdb.tool_events import SurrealDBToolEventsMixin


class _ToolEventsStore(SurrealDBToolEventsMixin):
    """Routes _query by SQL fragment; records UPDATE/insert calls."""

    def __init__(self, unprocessed=None, total=0, ok=0, grouped=None) -> None:
        self._unprocessed = unprocessed or []
        self._total = total
        self._ok = ok
        self._grouped = grouped or []
        self.updates: list[dict[str, Any]] = []
        self.inserts: list[dict[str, Any]] = []

    def _ensure_conn(self) -> Any:
        store = self

        class _Conn:
            async def insert(self, table: str, data: dict[str, Any]) -> None:
                store.inserts.append({"table": table, "data": data})

        return _Conn()

    def _get_brain_id(self) -> str:
        return "default"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if sql.startswith("UPDATE tool_events SET processed = true"):
            self.updates.append(params)
            return []
        if "processed = false" in sql:
            return self._unprocessed
        if "AND success = true GROUP ALL" in sql:
            return [{"c": self._ok}]
        if "count() AS c FROM tool_events WHERE brain_id = $bid GROUP ALL" in sql:
            return [{"c": self._total}]
        if "GROUP BY tool_name, server_name" in sql:
            return self._grouped
        return []


async def test_get_unprocessed_events_maps_rows() -> None:
    store = _ToolEventsStore(
        unprocessed=[
            {
                "event_id": "abc",
                "tool_name": "Read",
                "server_name": "",
                "args_summary": "x",
                "success": True,
                "duration_ms": 12,
                "session_id": "s1",
                "task_context": "t",
                "created_at": datetime(2026, 5, 29, 7, 0, 0),
            }
        ]
    )
    events = await store.get_unprocessed_events("default", 200)
    assert len(events) == 1
    ev = events[0]
    assert ev["id"] == "abc"
    assert ev["tool_name"] == "Read"
    assert ev["success"] is True
    # created_at must be an ISO string (process_events calls fromisoformat on it)
    assert isinstance(ev["created_at"], str)
    datetime.fromisoformat(ev["created_at"])


async def test_mark_events_processed_issues_update_with_ids() -> None:
    store = _ToolEventsStore()
    await store.mark_events_processed("default", ["abc", "def"])
    assert len(store.updates) == 1
    assert store.updates[0]["ids"] == ["abc", "def"]


async def test_mark_events_processed_noop_on_empty() -> None:
    store = _ToolEventsStore()
    await store.mark_events_processed("default", [])
    assert store.updates == []


async def test_get_tool_stats_computes_rate_and_top_tools() -> None:
    store = _ToolEventsStore(
        total=4,
        ok=3,
        grouped=[
            {"tool_name": "Read", "server_name": "", "cnt": 3},
            {"tool_name": "Bash", "server_name": "", "cnt": 1},
        ],
    )
    stats = await store.get_tool_stats("default")
    assert stats["total_events"] == 4
    assert stats["success_rate"] == 0.75
    assert stats["top_tools"][0]["tool_name"] == "Read"
    assert stats["top_tools"][0]["count"] == 3
