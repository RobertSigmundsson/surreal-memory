"""SurrealDB memory-maturation storage mixin (STM -> Working -> Episodic -> Semantic).

The maturation subsystem tracks each fiber's progress through the memory stages so
that ``_compute_consolidation_ratio`` (= semantic maturations / fibers) and the
``mature`` consolidation strategy work. Without this mixin the SurrealDB store fell
through to the no-op defaults in ``storage/base.py`` — ``save_maturation`` silently
dropped every write and ``find_maturations`` always returned ``[]`` — so the dashboard
reported "0 semantic" regardless of processing. Mirrors ``review_schedules.py``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from surreal_memory.engine.memory_stages import MaturationRecord, MemoryStage
from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.utils.timeutils import utcnow

logger = logging.getLogger(__name__)


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


def _row_to_maturation(row: dict[str, Any]) -> MaturationRecord:
    ts_raw = row.get("reinforcement_timestamps") or []
    timestamps = tuple(str(t) for t in ts_raw) if isinstance(ts_raw, (list, tuple)) else ()
    entered = _parse_datetime(row.get("stage_entered_at")) or utcnow()
    return MaturationRecord(
        fiber_id=str(row["fiber_id"]),
        brain_id=str(row["brain_id"]),
        stage=MemoryStage(str(row.get("stage", "stm"))),
        stage_entered_at=entered,
        rehearsal_count=int(row.get("rehearsal_count", 0)),
        reinforcement_timestamps=timestamps,
    )


class SurrealDBMaturationMixin:
    """Mixin providing maturation CRUD for SurrealDBStorage."""

    def _ensure_conn(self) -> Any:
        raise NotImplementedError

    def _get_brain_id(self) -> str:
        raise NotImplementedError

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def save_maturation(self, record: MaturationRecord) -> None:
        """Upsert a maturation record keyed by (brain_id, fiber_id)."""
        brain_id = self._get_brain_id()
        conn = self._ensure_conn()
        sid = f"{_to_surreal_id(brain_id)}_{_to_surreal_id(record.fiber_id)}"

        record_data: dict[str, Any] = {
            "fiber_id": record.fiber_id,
            "brain_id": brain_id,
            "stage": record.stage.value,
            "stage_entered_at": record.stage_entered_at,
            "rehearsal_count": int(record.rehearsal_count),
            "reinforcement_timestamps": list(record.reinforcement_timestamps),
        }

        existing = await self._query(
            "SELECT id FROM maturation WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=record.fiber_id,
        )
        try:
            if existing:
                # Merge by the existing record id (not the recomputed sid) so a
                # post-rename brain prefix mismatch can't silently no-op the write.
                await conn.merge(existing[0]["id"], record_data)
            else:
                insert_data = dict(record_data)
                insert_data["id"] = sid
                await conn.insert("maturation", insert_data)
        except Exception:
            # Fiber may have been deleted between read and write (parity with SQLite).
            logger.debug("Skipping maturation save for fiber %s", record.fiber_id)

    async def get_maturation(self, fiber_id: str) -> MaturationRecord | None:
        brain_id = self._get_brain_id()
        rows = await self._query(
            "SELECT * FROM maturation WHERE brain_id = $brain_id AND fiber_id = $fiber_id LIMIT 1",
            brain_id=brain_id,
            fiber_id=fiber_id,
        )
        return _row_to_maturation(rows[0]) if rows else None

    async def find_maturations(
        self,
        stage: MemoryStage | None = None,
        min_rehearsal_count: int = 0,
    ) -> list[MaturationRecord]:
        brain_id = self._get_brain_id()
        conditions = ["brain_id = $brain_id"]
        params: dict[str, Any] = {"brain_id": brain_id}

        if stage is not None:
            conditions.append("stage = $stage")
            params["stage"] = stage.value
        if min_rehearsal_count > 0:
            conditions.append("rehearsal_count >= $min_rc")
            params["min_rc"] = int(min_rehearsal_count)

        where = " AND ".join(conditions)
        rows = await self._query(
            f"SELECT * FROM maturation WHERE {where} LIMIT 5000",
            **params,
        )
        return [_row_to_maturation(r) for r in rows]

    async def cleanup_orphaned_maturations(self) -> int:
        """Delete maturation rows whose fiber no longer exists."""
        brain_id = self._get_brain_id()
        existing_rows = await self._query(
            "SELECT id, fiber_id FROM maturation WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        if not existing_rows:
            return 0
        fiber_rows = await self._query(
            "SELECT id FROM fiber WHERE brain_id = $brain_id",
            brain_id=brain_id,
        )
        live_fibers = {str(r["id"]).rsplit(":", 1)[-1] for r in fiber_rows}

        conn = self._ensure_conn()
        removed = 0
        for row in existing_rows:
            fid = str(row.get("fiber_id", ""))
            if _to_surreal_id(fid) not in live_fibers and fid not in live_fibers:
                rid = str(row.get("id", ""))
                if rid:
                    await conn.delete(rid)
                    removed += 1
        return removed

    async def backfill_maturations(self) -> dict[str, int]:
        """One-time: create maturation rows for existing fibers that lack one.

        Seeds each fiber's stage from its REAL age (``time_start``/``created_at``),
        not ``now``, so historical memories land at their age-appropriate stage
        immediately instead of restarting the maturation clock. Fibers that already
        have a maturation row are skipped, so this is idempotent and safe to re-run.

        Stage thresholds mirror ``engine.memory_stages`` (STM>30min Working,
        >4h Episodic, >=7d Semantic). The episodic->semantic spacing gate
        (distinct reinforcement days) is forward-looking for NEW memories; for this
        one-time backfill of memories that predate the maturation subsystem, age in
        the brain is the consolidation signal.
        """
        from surreal_memory.engine.memory_stages import (
            _EPISODIC_TO_SEMANTIC,
            _STM_TO_WORKING,
            _WORKING_TO_EPISODIC,
        )

        brain_id = self._get_brain_id()
        existing = {m.fiber_id for m in await self.find_maturations()}
        fibers = await self.get_fibers(limit=100000)  # type: ignore[attr-defined]
        now = utcnow()
        counts = {"stm": 0, "working": 0, "episodic": 0, "semantic": 0, "skipped": 0}

        for fiber in fibers:
            if fiber.id in existing:
                counts["skipped"] += 1
                continue
            base = fiber.time_start or fiber.created_at or now
            if base.tzinfo is not None:
                base = base.replace(tzinfo=None)
            age = now - base
            if age >= _EPISODIC_TO_SEMANTIC:
                stage, entered = MemoryStage.SEMANTIC, base + _EPISODIC_TO_SEMANTIC
            elif age >= _WORKING_TO_EPISODIC:
                stage, entered = MemoryStage.EPISODIC, base + _WORKING_TO_EPISODIC
            elif age >= _STM_TO_WORKING:
                stage, entered = MemoryStage.WORKING, base + _STM_TO_WORKING
            else:
                stage, entered = MemoryStage.SHORT_TERM, base
            await self.save_maturation(
                MaturationRecord(
                    fiber_id=fiber.id,
                    brain_id=brain_id,
                    stage=stage,
                    stage_entered_at=entered,
                )
            )
            counts[stage.value] += 1
        return counts
