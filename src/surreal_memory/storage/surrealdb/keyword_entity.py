"""SurrealDB keyword document-frequency and entity-refs mixin."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _safe_id(text: str) -> str:
    """Make a SurrealDB-safe ID component from arbitrary text (alnum + underscore)."""
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_") or "x"
    return cleaned[:64]


class SurrealDBKeywordEntityMixin:
    """Mixin providing keyword DF and entity-ref CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    # ---------------- keyword document frequency ----------------

    async def get_keyword_df_batch(self, keywords: list[str]) -> dict[str, int]:
        """Get fiber_count for each keyword. Missing keywords are absent from the result."""
        if not keywords:
            return {}

        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT keyword, fiber_count FROM keyword_document_frequency"
            " WHERE brain_id = $brain_id AND keyword IN $keywords",
            brain_id=brain_id,
            keywords=list(set(keywords)),
        )
        return {str(r["keyword"]): int(r.get("fiber_count", 0)) for r in rows}

    async def increment_keyword_df(self, keywords: list[str]) -> None:
        """Increment fiber_count by 1 for each unique keyword (UPSERT semantics)."""
        if not keywords:
            return

        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        now = utcnow()
        bid_safe = _to_surreal_id(brain_id)
        unique_keywords = set(keywords)

        for kw in unique_keywords:
            sid = f"{bid_safe}_{_safe_id(kw)}"
            existing = await self._query(
                "SELECT fiber_count FROM keyword_document_frequency"
                " WHERE brain_id = $brain_id AND keyword = $keyword LIMIT 1",
                brain_id=brain_id,
                keyword=kw,
            )
            if existing:
                current = int(existing[0].get("fiber_count", 0))
                await conn.merge(
                    f"keyword_document_frequency:{sid}",
                    {"fiber_count": current + 1, "last_updated": now},
                )
            else:
                await conn.insert(
                    "keyword_document_frequency",
                    {
                        "id": sid,
                        "brain_id": brain_id,
                        "keyword": kw,
                        "fiber_count": 1,
                        "last_updated": now,
                    },
                )

    # ---------------- entity refs ----------------

    async def add_entity_ref(
        self,
        entity_text: str,
        fiber_id: str,
        created_at: Any | None = None,
    ) -> None:
        """Record an entity mention for a fiber. Idempotent on (brain, entity, fiber)."""
        brain_id = self._get_brain_id()

        existing = await self._query(
            "SELECT id FROM entity_refs"
            " WHERE brain_id = $brain_id"
            " AND entity_text = $entity_text"
            " AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            entity_text=entity_text,
            fiber_id=fiber_id,
        )
        if existing:
            return

        conn = self._ensure_conn()
        bid_safe = _to_surreal_id(brain_id)
        sid = f"{bid_safe}_{_safe_id(entity_text)}_{_to_surreal_id(fiber_id)}"
        await conn.insert(
            "entity_refs",
            {
                "id": sid,
                "brain_id": brain_id,
                "entity_text": entity_text,
                "fiber_id": fiber_id,
                "created_at": created_at if created_at is not None else utcnow(),
                "promoted": False,
            },
        )

    async def count_entity_refs(self, entity_text: str) -> int:
        """Count fibers mentioning this entity (unpromoted only)."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT count() AS cnt FROM entity_refs"
            " WHERE brain_id = $brain_id AND entity_text = $entity_text"
            " AND promoted = false GROUP ALL",
            brain_id=brain_id,
            entity_text=entity_text,
        )
        return int(rows[0]["cnt"]) if rows else 0

    async def get_entity_ref_fiber_ids(self, entity_text: str) -> list[str]:
        """Get fiber IDs that reference this entity (unpromoted only)."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT fiber_id FROM entity_refs"
            " WHERE brain_id = $brain_id AND entity_text = $entity_text"
            " AND promoted = false",
            brain_id=brain_id,
            entity_text=entity_text,
        )
        return [str(r.get("fiber_id", "")) for r in rows if r.get("fiber_id")]

    async def mark_entity_refs_promoted(self, entity_text: str) -> int:
        """Mark all unpromoted refs for an entity as promoted. Returns count updated."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM entity_refs"
            " WHERE brain_id = $brain_id AND entity_text = $entity_text"
            " AND promoted = false",
            brain_id=brain_id,
            entity_text=entity_text,
        )
        if not rows:
            return 0

        conn = self._ensure_conn()
        updated = 0
        for r in rows:
            rid = str(r.get("id", ""))
            if not rid:
                continue
            await conn.merge(rid, {"promoted": True})
            updated += 1
        return updated

    async def prune_old_entity_refs(self, max_age_days: int = 90) -> int:
        """Remove unpromoted entity refs older than max_age_days. Returns count deleted."""
        brain_id = self._get_brain_id()
        cutoff = utcnow() - timedelta(days=max_age_days)

        rows = await self._query(
            "SELECT id FROM entity_refs"
            " WHERE brain_id = $brain_id AND promoted = false AND created_at < $cutoff",
            brain_id=brain_id,
            cutoff=cutoff,
        )
        if not rows:
            return 0

        conn = self._ensure_conn()
        deleted = 0
        for r in rows:
            rid = str(r.get("id", ""))
            if not rid:
                continue
            await conn.delete(rid)
            deleted += 1
        return deleted
