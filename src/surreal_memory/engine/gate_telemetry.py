"""Write-gate telemetry — append engine-side gate decisions to the
`gate_decision` table (the same SCHEMALESS table the Hermes plugin writes to),
so scripts/gate_stats.py sees BOTH the plugin and the engine (smem
auto-capture / stop-hook) decisions in one report.

Fire-and-forget: logging must never break a write.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _lang_hint(s: str) -> str:
    """'cjk' if the text contains CJK / Kana / Hangul, else 'latin' (no lib)."""
    for ch in s:
        o = ord(ch)
        if (
            0x4E00 <= o <= 0x9FFF
            or 0x3400 <= o <= 0x4DBF
            or 0x3040 <= o <= 0x30FF
            or 0xAC00 <= o <= 0xD7A3
        ):
            return "cjk"
    return "latin"


async def log_gate_decision(
    storage: Any,
    *,
    intent: str,
    accepted: bool,
    reason: str,
    score: int | None,
    mode: str,
    content: str,
    agent_id: str = "",
) -> None:
    """Best-effort insert of one gate decision (both SHADOW and ENFORCE log)."""
    try:
        from surreal_memory.utils.timeutils import utcnow

        conn = storage._ensure_conn()
        s = (content or "").strip()
        await conn.insert(
            "gate_decision",
            {
                "ts": utcnow(),
                "intent": intent,
                "accepted": accepted,
                "score": score,
                "reason": reason,
                "preview": s[:80],
                "length": len(s),
                "lang_hint": _lang_hint(s),
                "agent_id": agent_id,
                "mode": mode,
                "source": "engine",
            },
        )
    except Exception:
        logger.debug("gate_decision log failed (non-fatal)", exc_info=True)
