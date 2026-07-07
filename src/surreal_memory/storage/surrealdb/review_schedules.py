"""SurrealDB review schedules storage mixin (Leitner-box spaced repetition)."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.core.review_schedule import ReviewSchedule
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


def _to_surreal_id(record_id: str) -> str:
    return record_id.replace("-", "_")


def _parse_datetime(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo is not None else val
    if isinstance(val, str):
        try:
            parsed = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return parsed.replace(tzinfo=None) if parsed.tzinfo is not None else parsed
        except (ValueError, AttributeError):
            return None
    return None


def _row_to_schedule(row: dict[str, Any]) -> ReviewSchedule:
    """Convert a SurrealDB review_schedules record to a ReviewSchedule dataclass."""
    return ReviewSchedule(
        fiber_id=str(row["fiber_id"]),
        brain_id=str(row["brain_id"]),
        box=int(row.get("box", 1)),
        next_review=_parse_datetime(row.get("next_review")),
        last_reviewed=_parse_datetime(row.get("last_reviewed")),
        review_count=int(row.get("review_count", 0)),
        streak=int(row.get("streak", 0)),
        created_at=_parse_datetime(row.get("created_at")),
    )


class SurrealDBReviewSchedulesMixin:
    """Mixin providing review schedule CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def add_review_schedule(self, schedule: ReviewSchedule) -> str:
        """Insert or update a review schedule (upsert by fiber_id + brain_id)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        sid = f"{_to_surreal_id(brain_id)}_{_to_surreal_id(schedule.fiber_id)}"

        record_data: dict[str, Any] = {
            "fiber_id": schedule.fiber_id,
            "brain_id": brain_id,
            "box": int(schedule.box),
            "next_review": schedule.next_review,
            "last_reviewed": schedule.last_reviewed,
            "review_count": int(schedule.review_count),
            "streak": int(schedule.streak),
        }

        existing = await self._query(
            "SELECT id FROM review_schedules"
            " WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=schedule.fiber_id,
        )
        if existing:
            # Merge by the EXISTING record id, not the recomputed sid: a brain
            # rename updates the brain_id FIELD but not the record id, so the id
            # may still carry the old brain prefix. Recomputing the sid here would
            # target a non-existent record and the update would silently no-op.
            await conn.merge(existing[0]["id"], record_data)
        else:
            insert_data = dict(record_data)
            insert_data["id"] = sid
            insert_data["created_at"] = schedule.created_at or utcnow()
            await conn.insert("review_schedules", insert_data)

        return schedule.fiber_id

    async def get_review_schedule(self, fiber_id: str) -> ReviewSchedule | None:
        """Get a review schedule by fiber ID."""
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM review_schedules"
            " WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        if not rows:
            return None
        return _row_to_schedule(rows[0])

    async def get_due_reviews(self, limit: int = 20) -> list[ReviewSchedule]:
        """Get review schedules that are due (next_review <= now), ordered ASC."""
        brain_id = self._get_brain_id()
        safe_limit = min(limit, 100)
        now = utcnow()

        rows = await self._query(
            "SELECT * FROM review_schedules"
            " WHERE brain_id = $brain_id AND next_review IS NOT NONE AND next_review <= $now"
            " ORDER BY next_review ASC LIMIT $limit",
            brain_id=brain_id,
            now=now,
            limit=safe_limit,
        )
        return [_row_to_schedule(r) for r in rows]

    async def delete_review_schedule(self, fiber_id: str) -> bool:
        """Delete a review schedule. Returns True if deleted."""
        brain_id = self._get_brain_id()

        existing = await self._query(
            "SELECT id FROM review_schedules"
            " WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        if not existing:
            return False

        conn = self._ensure_conn()
        rid = str(existing[0].get("id", ""))
        if not rid:
            return False
        await conn.delete(rid)
        return True

    async def get_review_stats(self) -> dict[str, int]:
        """Return total/due plus per-box counts (boxes 1..5)."""
        brain_id = self._get_brain_id()
        now = utcnow()

        total_rows = await self._query(
            "SELECT count() AS cnt FROM review_schedules WHERE brain_id = $brain_id GROUP ALL",
            brain_id=brain_id,
        )
        due_rows = await self._query(
            "SELECT count() AS cnt FROM review_schedules"
            " WHERE brain_id = $brain_id AND next_review IS NOT NONE AND next_review <= $now"
            " GROUP ALL",
            brain_id=brain_id,
            now=now,
        )
        box_rows = await self._query(
            "SELECT box, count() AS cnt FROM review_schedules"
            " WHERE brain_id = $brain_id GROUP BY box",
            brain_id=brain_id,
        )

        stats: dict[str, int] = {
            "total": int(total_rows[0].get("cnt", 0)) if total_rows else 0,
            "due": int(due_rows[0].get("cnt", 0)) if due_rows else 0,
            "box_1": 0,
            "box_2": 0,
            "box_3": 0,
            "box_4": 0,
            "box_5": 0,
        }
        for r in box_rows:
            box = int(r.get("box", 0))
            if 1 <= box <= 5:
                stats[f"box_{box}"] = int(r.get("cnt", 0))
        return stats
