"""Unit tests for the SurrealDB projects mixin (no live DB required)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from surreal_memory.core.project import Project
from surreal_memory.storage.surrealdb.projects import (
    SurrealDBProjectsMixin,
    _row_to_project,
    _to_surreal_id,
)
from surreal_memory.utils.timeutils import utcnow


class _FakeConn:
    """Records query/insert/delete calls; returns nothing useful."""

    def __init__(self, *, fail_query: bool = False, fail_delete: bool = False) -> None:
        self.queries: list[tuple[str, Any]] = []
        self.inserts: list[tuple[str, Any]] = []
        self.deletes: list[str] = []
        self._fail_query = fail_query
        self._fail_delete = fail_delete

    async def query(self, sql: str, data: Any = None) -> list[Any]:
        if self._fail_query:
            raise RuntimeError("upsert boom")
        self.queries.append((sql, data))
        return []

    async def insert(self, table: str, data: Any) -> None:
        self.inserts.append((table, data))

    async def delete(self, record_id: str) -> None:
        if self._fail_delete:
            raise RuntimeError("delete boom")
        self.deletes.append(record_id)


class _Store(SurrealDBProjectsMixin):
    """Minimal concrete mixin host with stubbed protocol methods."""

    def __init__(
        self, rows: list[dict[str, Any]] | None = None, conn: _FakeConn | None = None
    ) -> None:
        self._conn = conn or _FakeConn()
        self._rows = rows or []

    def _ensure_conn(self) -> Any:
        return self._conn

    def _get_brain_id(self) -> str:
        return "brain-1"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        return self._rows


def test_to_surreal_id_replaces_dashes() -> None:
    assert _to_surreal_id("a-b-c") == "a_b_c"


def test_row_to_project_parses_valid_data() -> None:
    proj = Project.create(name="alpha")
    parsed = _row_to_project({"data": proj.to_dict()})
    assert parsed is not None
    assert parsed.id == proj.id
    assert parsed.name == "alpha"


def test_row_to_project_returns_none_on_garbage() -> None:
    assert _row_to_project({"data": "not-a-dict"}) is None
    assert _row_to_project({}) is None


async def test_add_project_upserts_and_returns_id() -> None:
    store = _Store()
    proj = Project.create(name="alpha")
    returned = await store.add_project(proj)
    assert returned == proj.id
    assert len(store._conn.queries) == 1
    sql, data = store._conn.queries[0]
    assert sql.startswith("UPSERT project:")
    assert data["data"]["name"] == "alpha"
    assert data["data"]["brain_id"] == "brain-1"


async def test_add_project_falls_back_to_insert_on_query_failure() -> None:
    conn = _FakeConn(fail_query=True)
    store = _Store(conn=conn)
    proj = Project.create(name="alpha")
    returned = await store.add_project(proj)
    assert returned == proj.id
    assert len(conn.inserts) == 1
    table, data = conn.inserts[0]
    assert table == "project"
    assert data["id"] == _to_surreal_id(proj.id)


async def test_get_project_by_name_is_case_insensitive() -> None:
    proj = Project.create(name="Alpha")
    store = _Store(rows=[{"data": proj.to_dict()}])
    found = await store.get_project_by_name("alpha")
    assert found is not None
    assert found.name == "Alpha"


async def test_get_project_by_name_returns_none_when_absent() -> None:
    proj = Project.create(name="Alpha")
    store = _Store(rows=[{"data": proj.to_dict()}])
    assert await store.get_project_by_name("beta") is None


async def test_list_projects_sorts_by_priority_desc() -> None:
    low = Project.create(name="low", priority=1.0)
    high = Project.create(name="high", priority=9.0)
    store = _Store(rows=[{"data": low.to_dict()}, {"data": high.to_dict()}])
    projects = await store.list_projects()
    assert [p.name for p in projects] == ["high", "low"]


async def test_list_projects_active_only_excludes_finished() -> None:
    ongoing = Project.create(name="ongoing")
    finished = Project.create(name="finished", end_date=utcnow() - timedelta(days=1))
    store = _Store(rows=[{"data": ongoing.to_dict()}, {"data": finished.to_dict()}])
    projects = await store.list_projects(active_only=True)
    assert [p.name for p in projects] == ["ongoing"]


async def test_list_projects_tag_filter() -> None:
    tagged = Project.create(name="tagged", tags={"infra"})
    other = Project.create(name="other", tags={"docs"})
    store = _Store(rows=[{"data": tagged.to_dict()}, {"data": other.to_dict()}])
    projects = await store.list_projects(tags={"infra"})
    assert [p.name for p in projects] == ["tagged"]


async def test_delete_project_returns_true_on_success() -> None:
    store = _Store()
    assert await store.delete_project("some-id") is True
    assert store._conn.deletes == [f"project:{_to_surreal_id('some-id')}"]


async def test_delete_project_returns_false_on_failure() -> None:
    store = _Store(conn=_FakeConn(fail_delete=True))
    assert await store.delete_project("some-id") is False
