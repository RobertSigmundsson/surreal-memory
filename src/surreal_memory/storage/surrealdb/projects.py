"""SurrealDB project operations mixin.

Adds Project-entity CRUD to the SurrealDB backend. The SurrealDB-only refactor
(commit 1f6fe80) shipped `get_project_memories` (typed_memory filter) but no
Project entity, so `smem project create`, `smem remember --project NAME` and
`smem list --project NAME` failed with AttributeError. This restores parity with
the in-memory / SQLite backends.

A project is stored verbatim via `Project.to_dict()` under the `data` field, with
`brain_id`, `uid` (the original Project.id) and `name` denormalised alongside for
brain-scoped lookups. The SurrealDB record id is `project:<uid-with-underscores>`.
"""

from __future__ import annotations

import logging
from typing import Any

from surreal_memory.core.project import Project

logger = logging.getLogger(__name__)


def _to_surreal_id(record_id: str) -> str:
    return record_id.replace("-", "_")


def _row_to_project(row: dict[str, Any]) -> Project | None:
    """Convert a SurrealDB project record to a Project."""
    data = row.get("data")
    if isinstance(data, dict):
        try:
            return Project.from_dict(data)
        except Exception:
            logger.debug("Failed to parse project row via from_dict", exc_info=True)
    return None


class SurrealDBProjectsMixin:
    """Mixin providing Project CRUD for SurrealDBStorage."""

    # ------------------------------------------------------------------
    # Protocol stubs — satisfied by SurrealDBStorage at runtime
    # ------------------------------------------------------------------

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_project(self, project: Project) -> str:
        conn = self._ensure_conn()
        brain_id = self._get_brain_id()
        sid = _to_surreal_id(project.id)

        record_data: dict[str, Any] = {
            "brain_id": brain_id,
            "uid": project.id,
            "name": project.name,
            "data": project.to_dict(),
        }

        try:
            await conn.query(f"UPSERT project:{sid} CONTENT $data", {"data": record_data})
        except Exception:
            # Fallback: delete then insert
            try:
                await conn.delete(f"project:{sid}")
            except Exception:
                pass
            insert_data = dict(record_data)
            insert_data["id"] = sid
            await conn.insert("project", insert_data)

        return project.id

    async def get_project(self, project_id: str) -> Project | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM project WHERE brain_id = $brain_id AND uid = $uid LIMIT 1",
            brain_id=brain_id,
            uid=project_id,
        )
        return _row_to_project(rows[0]) if rows else None

    async def get_project_by_name(self, name: str) -> Project | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM project WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        name_lower = name.lower()
        for row in rows:
            project = _row_to_project(row)
            if project is not None and project.name.lower() == name_lower:
                return project
        return None

    async def list_projects(
        self,
        active_only: bool = False,
        tags: set[str] | None = None,
        limit: int = 100,
    ) -> list[Project]:
        brain_id = self._get_brain_id()
        limit = min(limit, 1000)
        rows = await self._query(
            "SELECT * FROM project WHERE brain_id = $brain_id LIMIT $limit",
            brain_id=brain_id,
            limit=limit,
        )

        results: list[Project] = []
        for row in rows:
            project = _row_to_project(row)
            if project is None:
                continue
            if active_only and not project.is_active:
                continue
            if tags is not None and not tags.intersection(project.tags):
                continue
            results.append(project)

        results.sort(key=lambda p: (p.priority, p.start_date), reverse=True)
        return results

    async def update_project(self, project: Project) -> None:
        # UPSERT semantics — same write path as add_project.
        await self.add_project(project)

    async def delete_project(self, project_id: str) -> bool:
        conn = self._ensure_conn()
        sid = _to_surreal_id(project_id)
        try:
            await conn.delete(f"project:{sid}")
        except Exception:
            logger.debug("Failed to delete project %s", project_id, exc_info=True)
            return False
        return True
