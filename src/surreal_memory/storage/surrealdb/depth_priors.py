"""SurrealDB Bayesian depth priors mixin."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.engine.depth_prior import DepthPrior
from surreal_memory.engine.retrieval_types import DepthLevel
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _to_surreal_id(record_id: str) -> str:
    return record_id.replace("-", "_")


def _safe_text(text: str) -> str:
    """Make a SurrealDB-safe ID component from arbitrary text (max 48 chars)."""
    out = []
    for ch in text:
        if ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("_") or "x"
    return cleaned[:48]


def _parse_datetime(val: Any) -> datetime:
    if val is None:
        return utcnow()
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except (ValueError, AttributeError):
            pass
    return utcnow()


def _row_to_depth_prior(row: dict[str, Any]) -> DepthPrior:
    return DepthPrior(
        entity_text=str(row["entity_text"]),
        depth_level=DepthLevel(int(row.get("depth_level", 0))),
        alpha=float(row.get("alpha", 1.0)),
        beta=float(row.get("beta", 1.0)),
        total_queries=int(row.get("total_queries", 0)),
        last_updated=_parse_datetime(row.get("last_updated")),
        created_at=_parse_datetime(row.get("created_at")),
    )


class SurrealDBDepthPriorsMixin:
    """Mixin providing Bayesian depth prior CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_depth_priors_batch(
        self,
        entity_texts: list[str],
    ) -> dict[str, list[DepthPrior]]:
        """Batch-fetch depth priors for multiple entities.

        Returns dict mapping entity_text → list of DepthPrior (one per depth level).
        Entities with no stored priors are absent from the result.
        """
        if not entity_texts:
            return {}

        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM depth_priors"
            " WHERE brain_id = $brain_id AND entity_text IN $entity_texts",
            brain_id=brain_id,
            entity_texts=list(set(entity_texts)),
        )

        result: dict[str, list[DepthPrior]] = {}
        for r in rows:
            prior = _row_to_depth_prior(r)
            result.setdefault(prior.entity_text, []).append(prior)
        return result

    async def upsert_depth_prior(self, prior: DepthPrior) -> None:
        """Insert or update a depth prior for (brain_id, entity_text, depth_level)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        depth_val = int(prior.depth_level)
        sid = f"{_to_surreal_id(brain_id)}_{_safe_text(prior.entity_text)}_{depth_val}"

        existing = await self._query(
            "SELECT id FROM depth_priors"
            " WHERE brain_id = $brain_id AND entity_text = $entity_text"
            " AND depth_level = $depth_level LIMIT 1",
            brain_id=brain_id,
            entity_text=prior.entity_text,
            depth_level=depth_val,
        )

        data: dict[str, Any] = {
            "brain_id": brain_id,
            "entity_text": prior.entity_text,
            "depth_level": depth_val,
            "alpha": float(prior.alpha),
            "beta": float(prior.beta),
            "total_queries": int(prior.total_queries),
            "last_updated": prior.last_updated,
        }

        if existing:
            rid = str(existing[0].get("id", f"depth_priors:{sid}"))
            await conn.merge(rid, data)
        else:
            data["id"] = sid
            data["created_at"] = prior.created_at
            await conn.insert("depth_priors", data)

    async def get_stale_priors(self, older_than: datetime) -> list[DepthPrior]:
        """Return depth priors with last_updated before older_than."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM depth_priors WHERE brain_id = $brain_id AND last_updated < $older_than",
            brain_id=brain_id,
            older_than=older_than,
        )
        return [_row_to_depth_prior(r) for r in rows]

    async def delete_depth_priors(self, entity_text: str) -> int:
        """Delete all depth priors for an entity. Returns count deleted."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT id FROM depth_priors WHERE brain_id = $brain_id AND entity_text = $entity_text",
            brain_id=brain_id,
            entity_text=entity_text,
        )
        if not rows:
            return 0
        conn = self._ensure_conn()
        deleted = 0
        for r in rows:
            rid = str(r.get("id", ""))
            if rid:
                await conn.delete(rid)
                deleted += 1
        return deleted
