"""Per-session idempotency + near-dup state for auto-capture hooks.

The Stop hook fires after *every* assistant turn and PreCompact fires on
each compaction; both re-read an overlapping transcript tail and run
``analyze_text_for_memories`` over it. Without memory of what was already
captured, identical (and near-identical) fragments — plus session
summaries, timestamps, paths — are re-encoded on every invocation, the
dominant source of memory poisoning (duplicate fragments accumulating
across turns).

This module provides two deterministic, dependency-free guards, keyed per
session:

1. **Exact** (idempotency): a set of normalized-content md5 hashes. A
   re-captured byte-identical fragment is skipped.
2. **Near-dup**: a parallel list of SimHashes. A fragment within
   ``NEAR_DUP_THRESHOLD`` Hamming bits of a previously captured one is
   skipped too — catching trivial variants (timestamp tick, whitespace,
   truncation) that exact hashing misses.

The threshold is calibrated on real pairs from the live brain (normalized
SimHash): trivial variants are small (whitespace/case = 0, punctuation
<=7), whereas *distinct* valuable fragments score >=23 (real-auto 28-40,
real-summary 23-38). A threshold of 7 leaves a 16-bit margin, so distinct
content is never collapsed, and it matches the engine's existing dedup
``simhash_threshold``. Larger single-fragment edits (truncation or an
appended timestamp on a short fragment, 8-18 bits) are deliberately
treated as *new* — they border on "different fragment", and the exact-key
guard still catches byte-identical re-captures. The "same template,
different content" trap (``completed X`` vs ``completed Y``) sits well
above the threshold, so different events are kept apart.

Design notes:
- Fail-open: any error → treat content as not-seen. Losing real content is
  worse than tolerating a duplicate, so errors never block a capture.
- Bounded: at most ``_MAX_HASHES_PER_SESSION`` entries per session and
  ``_MAX_SESSIONS`` sessions (FIFO trim) so the file cannot grow without
  bound.
- Atomic-ish write: temp file then ``replace`` to avoid a torn state file.
- Shared key: keyed on ``CLAUDE_SESSION_ID`` so Stop and PreCompact share
  one state and do not re-capture each other's writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path

from surreal_memory.utils.simhash import hamming_distance, simhash

logger = logging.getLogger(__name__)

# Bounded retention so the state file stays small.
_MAX_HASHES_PER_SESSION = 2000
_MAX_SESSIONS = 50

# SimHash Hamming threshold for near-duplicates. Calibrated on live pairs
# (normalized): trivial variants <=7, distinct-valuable >=23 (16-bit margin).
# Matches the engine's existing dedup simhash_threshold.
NEAR_DUP_THRESHOLD = 7

_WS = re.compile(r"\s+")


def _state_path() -> Path:
    """Location of the JSON state file."""
    data_dir = Path(os.environ.get("SURREAL_MEMORY_DIR", "")) or (Path.home() / ".surrealmemory")
    return data_dir / "capture_state.json"


def session_key(transcript_path: str | None = None) -> str:
    """Stable per-session key.

    Prefers ``CLAUDE_SESSION_ID`` (set by Claude Code); falls back to a hash
    of the transcript path, then a constant.
    """
    sid = os.environ.get("CLAUDE_SESSION_ID", "").strip()
    if sid:
        return sid
    if transcript_path:
        return "tp_" + hashlib.md5(transcript_path.encode("utf-8")).hexdigest()[:16]
    return "default"


def _normalize(content: str) -> str:
    return _WS.sub(" ", content.strip().lower())


def content_key(content: str) -> str:
    """Deterministic exact key: whitespace-normalized, lowercased md5."""
    return hashlib.md5(_normalize(content).encode("utf-8")).hexdigest()


def content_simhash(content: str) -> int:
    """SimHash of normalized content (same normalization as content_key)."""
    return simhash(_normalize(content))


def _load_all() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("capture_state: load failed (fail-open, treat as empty)", exc_info=True)
        return {}


def load_seen(skey: str) -> tuple[set[str], list[int]]:
    """Return (exact content-keys, SimHashes) already captured this session."""
    entry = _load_all().get(skey, {})
    if not isinstance(entry, dict):
        return set(), []
    keys = entry.get("hashes", [])
    sims = entry.get("simhashes", [])
    key_set = set(keys) if isinstance(keys, list) else set()
    sim_list = [s for s in sims if isinstance(s, int)] if isinstance(sims, list) else []
    return key_set, sim_list


def is_duplicate(
    content: str,
    seen_keys: set[str],
    seen_simhashes: list[int],
    threshold: int = NEAR_DUP_THRESHOLD,
) -> bool:
    """True if content is an exact or near-duplicate of something seen this session.

    Exact: normalized md5 already present. Near-dup: SimHash within
    ``threshold`` Hamming bits of a previously captured fragment.
    """
    if content_key(content) in seen_keys:
        return True
    if seen_simhashes:
        sh = content_simhash(content)
        if any(hamming_distance(sh, s) <= threshold for s in seen_simhashes):
            return True
    return False


def mark_seen(skey: str, contents: list[str]) -> None:
    """Persist newly captured contents (exact key + SimHash) for this session.

    Accepts raw content strings (computes both keys). Bounded, atomic-ish,
    and never raises — idempotency bookkeeping must not break a capture path.
    """
    if not contents:
        return
    try:
        ts = ""
        try:
            from surreal_memory.utils.timeutils import utcnow

            ts = utcnow().isoformat()
        except Exception:
            ts = ""

        data = _load_all()
        entry = data.get(skey, {}) if isinstance(data.get(skey), dict) else {}
        keys = entry.get("hashes", [])
        sims = entry.get("simhashes", [])
        if not isinstance(keys, list):
            keys = []
        if not isinstance(sims, list):
            sims = []

        seen_k = set(keys)
        for c in contents:
            k = content_key(c)
            if k in seen_k:
                continue
            seen_k.add(k)
            keys.append(k)
            sims.append(content_simhash(c))

        # Bound both lists (FIFO; kept parallel by trimming the same tail count).
        if len(keys) > _MAX_HASHES_PER_SESSION:
            keys = keys[-_MAX_HASHES_PER_SESSION:]
        if len(sims) > _MAX_HASHES_PER_SESSION:
            sims = sims[-_MAX_HASHES_PER_SESSION:]
        data[skey] = {"hashes": keys, "simhashes": sims, "ts": ts}

        # Bound the number of retained sessions (drop oldest by timestamp).
        if len(data) > _MAX_SESSIONS:
            ordered = sorted(
                data.items(),
                key=lambda kv: kv[1].get("ts", "") if isinstance(kv[1], dict) else "",
            )
            data = dict(ordered[-_MAX_SESSIONS:])

        path = _state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("capture_state: mark_seen failed (non-fatal)", exc_info=True)
