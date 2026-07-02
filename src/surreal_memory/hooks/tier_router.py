"""Tier router for fresh auto-captures — a router, not a gate.

Everything is still saved; this only decides WHICH tier a fresh auto-capture
lands in, so low-value content does not crowd the default recall path.
Nothing is dropped: cold memories are retained, decay faster, and stay
reachable via explicit ``smem_recall(tier="cold")``.

Conservative policy (chosen with the owner, 2026-06-30):
- Session summaries (wall-of-text "Session activity:") → COLD. They are not
  atomic knowledge and are empirically inseparable by cheap signal (real
  prose vs chat); cold keeps them out of default recall while a real summary
  stays explicitly recoverable and noise summaries decay out.
- Auto-capture fragments → WARM by default. Only *obvious* noise — short AND
  non-specific AND syntactically incomplete (a mid-sentence regex fragment) —
  is routed to COLD. After the idempotency + near-dup guard the survivors are
  unique, so they stay visible by default (no hiding of valuable content).
- HOT is reserved for boundary/explicit memories, never auto-capture.

This deliberately does NOT use the G2 specificity classifier as an
accept/reject gate (its 67%/89% ceiling makes it unsafe as a gate). As a
router its mistakes are recoverable (wrong tier, never a lost neuron).
"""

from __future__ import annotations

import re

from surreal_memory.core.memory_types import MemoryTier

_PREFIX = re.compile(r"^(Error|Decision|TODO|Insight|Preference|Session activity):\s*", re.I)
_BACKTICK = re.compile(r"`[^`]+`")
_SNAKE = re.compile(r"\b\w+_\w+\b")
_IDENT = re.compile(r"\b\w+:\w+\b")
_VERSION = re.compile(r"\b[0-9a-f]{7,}\b|\bv\d+\b")

_SHORT_LEN = 50

# Trailing word that signals a cut-off fragment (PL + EN prepositions/conjunctions).
_TRAILING_STOP = frozenset(
    {
        "i",
        "oraz",
        "w",
        "do",
        "na",
        "z",
        "że",
        "nie",
        "po",
        "przy",
        "o",
        "a",
        "ale",
        "lub",
        "czy",
        "jako",
        "dla",
        "od",
        "za",
        "to",
        "the",
        "of",
        "and",
        "or",
        "in",
        "by",
        "is",
        "was",
        "for",
        "with",
        "at",
        "on",
    }
)


def _specificity(content: str) -> int:
    """Count concrete referents (backticked ids, snake_case, x:y, hashes/versions)."""
    return (
        len(_BACKTICK.findall(content))
        + len(_SNAKE.findall(content))
        + len(_IDENT.findall(content))
        + len(_VERSION.findall(content))
    )


def _incomplete(content: str) -> bool:
    """Heuristic: is this a cut-off, mid-sentence fragment?"""
    body = _PREFIX.sub("", content).strip()
    if len(body) < 3:
        return True
    if body.count("`") % 2 != 0 or body.count("(") != body.count(")"):
        return True
    if body[-1] in ",-—:":
        return True
    words = body.split()
    if words:
        last = re.sub(r"[^\wąćęłńóśźż]+$", "", words[-1].lower())
        if last in _TRAILING_STOP:
            return True
    return False


def _is_session_summary(content: str, memory_type: str | None, tags: object) -> bool:
    if isinstance(tags, (list, tuple, set, frozenset)) and "session_summary" in tags:
        return True
    return content.lstrip().lower().startswith("session activity")


def route_tier(
    content: str,
    *,
    memory_type: str | None = None,
    tags: object = None,
) -> str:
    """Tier for a fresh auto-capture. Router, not gate — content is always saved.

    Returns a ``MemoryTier`` value ("warm" or "cold"). Conservative: only
    session summaries and obvious short/non-specific/incomplete noise go cold.
    """
    if _is_session_summary(content, memory_type, tags):
        return MemoryTier.COLD
    if len(content.strip()) < _SHORT_LEN and _specificity(content) == 0 and _incomplete(content):
        return MemoryTier.COLD
    return MemoryTier.WARM
