"""Reasoning-strategy injection: pick the target model, build the prompt block.

Pure, testable logic used by the SessionStart hook (and, from run 007 Faza 5b,
the UserPromptSubmit hook):

- ``resolve_active_model(hook_input)`` — SessionStart payloads carry no ``model``
  field, so resolve it via a fallback chain (payload → transcript tail →
  env → ~/.claude/settings.json → "default").
- ``build_injection_context(storage, model, config)`` — map the active model to a
  source model via ``injection_map`` (glob, first-match, "default" fallback),
  pull that source's ReasoningBank pattern fibers, and render a compact markdown
  block ("## Reasoning strategies (learned from <source>)").
- session-scoped idempotency markers so SessionStart + UserPromptSubmit inject at
  most once per session.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.engine.reasoning_miner import normalize_model

if TYPE_CHECKING:
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)

_SYNTHETIC_MODEL = "<synthetic>"
_MAX_PER_CATEGORY = 2
_MARKER_MAX_AGE_S = 7 * 86400  # prune injection markers older than 7 days
_TRANSCRIPT_TAIL_LINES = 300
# Ceiling for the pattern-fiber fetch. Patterns are idempotent by
# _reasoning_signature (reasoning_distiller._materialize_pattern), so the
# population is bounded by distinct patterns and stays well under this; matches
# the distiller's own fetch limit. If it is ever hit, we warn rather than let a
# source model's patterns silently fall outside the window (the post-LIMIT
# metadata-filter failure mode documented in storage.find_fibers).
_PATTERN_FETCH_LIMIT = 20_000

# Claude Code short model aliases -> canonical ids used across the reasoning
# pipeline. Full ids pass through ``normalize_model`` unchanged.
_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "opusplan": "claude-opus-5",
    "haiku": "claude-haiku-4-5",
    "fable": "claude-fable-5",
}


# ── Model resolution ─────────────────────────────────────────────────────────


def _model_from_transcript_tail(transcript_path: str) -> str:
    """Return the last assistant turn's model from a JSONL transcript tail.

    ``transcript_path`` comes from untrusted hook stdin, so only transcripts
    under ~/.claude are read (mirrors the pre_compact / stop path-allowlist
    guard). Any stat/read error degrades to "" so the fallback chain continues.
    """
    if not transcript_path:
        return ""
    try:
        resolved = Path(transcript_path).resolve()
        if not resolved.is_relative_to((Path.home() / ".claude").resolve()):
            return ""
        if not resolved.is_file():
            return ""
        with resolved.open(encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    for raw in reversed(lines[-_TRANSCRIPT_TAIL_LINES:]):
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        model = str(message.get("model") or "")
        if model and model != _SYNTHETIC_MODEL:
            return normalize_model(model)
    return ""


def _model_from_settings() -> str:
    """Return the model from ~/.claude/settings.json (alias-expanded)."""
    try:
        path = Path.home() / ".claude" / "settings.json"
        if not path.exists():
            return ""
        data = json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, json.JSONDecodeError):
        return ""
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(model, str) or not model.strip():
        return ""
    model = model.strip()
    return _MODEL_ALIASES.get(model.lower(), normalize_model(model))


def resolve_active_model(hook_input: dict[str, Any]) -> str:
    """Resolve the active model for injection via a fallback chain.

    Order: (1) ``hook_input["model"]`` (future-proof), (2) last assistant
    ``message.model`` from the transcript tail, (3) env
    ``SMEM_REASONING_TARGET_MODEL`` / ``ANTHROPIC_MODEL``, (4) ~/.claude/
    settings.json ``model`` (alias-expanded), (5) ``"default"``. Results are
    date-suffix-normalized so they agree with mining/distillation model names.
    """
    payload_model = hook_input.get("model")
    if isinstance(payload_model, str) and payload_model.strip():
        return _MODEL_ALIASES.get(payload_model.strip().lower(), normalize_model(payload_model))

    from_transcript = _model_from_transcript_tail(str(hook_input.get("transcript_path") or ""))
    if from_transcript:
        return from_transcript

    for env_var in ("SMEM_REASONING_TARGET_MODEL", "ANTHROPIC_MODEL"):
        value = os.environ.get(env_var)
        if value and value.strip():
            return _MODEL_ALIASES.get(value.strip().lower(), normalize_model(value))

    from_settings = _model_from_settings()
    if from_settings:
        return from_settings

    return "default"


# ── Injection context ────────────────────────────────────────────────────────


def _split_sources(source: str) -> tuple[str, ...]:
    """Split a map VALUE into source models, preserving order, deduped.

    Multi-source doctrine (Robert, 2026-07-27: every model learns from the
    models STRONGER than itself): a value may carry several sources separated
    by "," (config.toml) or "|" (the env-var format uses "," as the PAIR
    separator, so values there must use "|"). Single-source values pass
    through unchanged — full back-compat.
    """
    seen: list[str] = []
    for part in source.replace("|", ",").split(","):
        name = part.strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


def _resolve_source_models(
    model: str, injection_map: tuple[tuple[str, str], ...]
) -> tuple[str, ...]:
    """Map the active model to its source models via injection_map (glob
    first-match, then the literal ``default`` key)."""
    for target, source in injection_map:
        if target != "default" and fnmatch(model, target):
            return _split_sources(source)
    for target, source in injection_map:
        if target == "default":
            return _split_sources(source)
    return ()


_MOVES_PREFIX = "Moves: "


def _mark_unverified_order(metadata: dict[str, Any], strategy: str) -> str:
    """Stop a pre-measurement pattern from claiming an order nobody measured.

    Patterns distilled before ``segment_moves`` walked the text carry a chain
    derived from a dictionary iteration, and it was rendered with arrows either
    way. On the live corpus that rendered order disagreed with the text in 67.5%
    of the traces that had one — so these lines teach a sequence that may never
    have happened. Only a pattern whose chain was actually walked out of the
    text carries ``_reasoning_chain_source == "measured"``; a chain recovered by
    parsing an old rendered line is stamped ``"legacy-parsed"`` and is NOT
    evidence of order — the old renderer used arrows for the frequency top-3
    too, so a parse cannot tell the two apart. An absent stamp downgrades as
    well: unknown provenance is not proof.

    The bank is not rewritten (its history is what it is). The claim is
    downgraded where it is read, and disappears by itself as those patterns are
    merged into measured ones.
    """
    if metadata.get("_reasoning_chain_source") == "measured":
        return strategy
    head, sep, rest = strategy.partition("\n")
    if not head.startswith(_MOVES_PREFIX) or "->" not in head:
        return strategy
    moves = [p.strip() for p in head[len(_MOVES_PREFIX) :].split("->") if p.strip()]
    if not moves:
        return strategy
    return f"Moves (order unverified): {', '.join(moves)}{sep}{rest}"


async def build_injection_context(
    storage: NeuralStorage,
    model: str,
    config: UnifiedConfig,
) -> str:
    """Render the reasoning-strategies markdown block for *model*, or "".

    Empty when injection is disabled, no injection_map entry matches, or the
    mapped source model has no pattern fibers. ``storage`` must be on the target
    brain (find_fibers uses the current brain).
    """
    rt = config.reasoning_training
    if not rt.injection_enabled:
        return ""
    sources = _resolve_source_models(model, rt.injection_map)
    if not sources:
        return ""

    fibers = await storage.find_fibers(
        metadata_key="_reasoning_pattern", limit=_PATTERN_FETCH_LIMIT
    )
    if len(fibers) >= _PATTERN_FETCH_LIMIT:
        # Truncation would silently drop a model's patterns; surface it instead.
        logger.warning(
            "reasoning pattern-fiber fetch hit the %d-row ceiling; some %s "
            "patterns may be missing from injection",
            _PATTERN_FETCH_LIMIT,
            sources,
        )

    def _rank(f: Any) -> tuple[float, str, str]:
        # Negative rank first so a plain ascending sort is best-first, then
        # title and id as tie-breaks: find_fibers does not promise a stable row
        # order, so without them two runs over the same bank could inject
        # different strategies and neither would be wrong.
        md = f.metadata
        score = float(md.get("_reasoning_confidence", 0.0)) * float(
            md.get("_reasoning_frequency", 0.0)
        )
        return (-score, str(md.get("_reasoning_title", "")), str(getattr(f, "id", "")))

    # One block per source, each with its own patterns/char budget — a model
    # mapped to several sources (learn-from-the-stronger doctrine) gets every
    # bank fully, not the first bank crowding out the rest. A source with no
    # patterns contributes nothing (no empty headers).
    blocks: list[str] = []
    # Titles already delivered by an earlier block of THIS payload. Two source
    # models converge on the same strategy often enough that a sonnet session
    # was measurably served "debugging: restate-goal, verify, backtrack" twice
    # in one prompt — once per block. Each source still gets its own block and
    # its own budget (multi-source doctrine); it just does not spend a slot on
    # a strategy the session is already holding.
    delivered: set[str] = set()
    for source in sources:
        candidates = [
            f
            for f in fibers
            if f.metadata.get("_source_model") == source
            and str(f.metadata.get("_reasoning_title", "")).strip() not in delivered
        ]
        if not candidates:
            continue
        candidates.sort(key=_rank)

        # Take the best of each category in turn, rather than the globally best
        # five. confidence is a SHARE of a category (size / traces_in_category),
        # so confidence x frequency reduces to size^2 / traces_in_category and
        # systematically rewards whichever category is rarest — measured
        # 2026-08-26, the top 8 for BOTH opus-5 and fable-5 were 8/8
        # "verification". Only _MAX_PER_CATEGORY stopped the block from being
        # five of the same thing; it stays here as a safety belt, but the
        # diversity now comes from the selection itself.
        by_category: dict[str, list[Any]] = {}
        for f in candidates:
            by_category.setdefault(str(f.metadata.get("_reasoning_category", "")), []).append(f)

        # dict preserves insertion order, and candidates are already best-first,
        # so categories are visited in order of their strongest candidate.
        chosen: list[Any] = []
        depth = 0
        while len(chosen) < rt.injection_max_patterns and depth < _MAX_PER_CATEGORY:
            took_one = False
            for items in by_category.values():
                if depth >= len(items):
                    continue
                chosen.append(items[depth])
                took_one = True
                if len(chosen) >= rt.injection_max_patterns:
                    break
            if not took_one:
                break
            depth += 1
        if not chosen:
            continue

        header = f"## Reasoning strategies (learned from {source})"
        parts = [header, ""]
        total = len(header) + 1
        for i, f in enumerate(chosen, start=1):
            md = f.metadata
            title = str(md.get("_reasoning_title", "")).strip()
            body = str(
                md.get("_reasoning_strategy") or md.get("_reasoning_description", "")
            ).strip()
            body = _mark_unverified_order(md, body)
            body = " ".join(body.split())  # collapse whitespace/newlines to one line
            entry = f"{i}. **{title}** — {body}" if body else f"{i}. **{title}**"
            # Always include the first entry; later ones respect the char budget.
            if i > 1 and total + len(entry) + 1 > rt.injection_max_chars:
                break
            parts.append(entry)
            total += len(entry) + 1
            delivered.add(title)
        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


# ── Hook orchestration (shared by SessionStart + UserPromptSubmit) ────────────


async def get_reasoning_context(hook_input: dict[str, Any]) -> str:
    """Resolve the active model, build its reasoning block, mark the session.

    Shared by the SessionStart, UserPromptSubmit and SubagentStart hooks. Opt-in
    via reasoning_training.injection_enabled. Main-session hooks (SessionStart /
    UserPromptSubmit) inject at most once per session via the marker
    (already_injected/mark_injected) — whichever fires first wins. Two exemptions
    from the marker:
    - SubagentStart: fires once per spawned subagent and EVERY subagent must
      receive the strategies (payload carries the PARENT session_id, so the
      per-session marker would silently starve all subagents after the first).
    - SessionStart with source == "compact": compaction just flattened the old
      context, so the previously injected block is GONE from the conversation —
      honoring the marker would leave the session strategy-less until it ends;
      re-inject instead (startup/resume/clear keep the once-per-session rule:
      their history, including the block, is still present).
    Storage is opened on the current brain and always closed. Returns "" when
    injection is disabled, already done this session (non-exempt events only),
    or nothing matched.
    """
    from surreal_memory.unified_config import get_config, get_shared_storage

    config = get_config()
    if not config.reasoning_training.injection_enabled:
        return ""
    event = str(hook_input.get("hook_event_name") or "")
    per_session = not (
        event == "SubagentStart"
        or (event == "SessionStart" and str(hook_input.get("source") or "") == "compact")
    )
    session_id = str(hook_input.get("session_id") or "")
    if per_session and already_injected(session_id):
        return ""

    model = resolve_active_model(hook_input)
    storage = await get_shared_storage(config.current_brain)
    try:
        block = await build_injection_context(storage, model, config)
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("reasoning storage.close() failed (non-fatal)", exc_info=True)

    if block and per_session:
        mark_injected(session_id)
    return block


# ── Session idempotency markers ──────────────────────────────────────────────


def _marker_dir() -> Path:
    # Honor SURREAL_MEMORY_DIR (matches hooks/post_tool_use._get_data_dir) so
    # markers sit alongside the rest of the data dir and tests can redirect them.
    custom = os.environ.get("SURREAL_MEMORY_DIR", "")
    base = Path(custom) if custom else (Path.home() / ".surrealmemory")
    return base / "reasoning_injected"


def _safe_session(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:128]


def already_injected(session_id: str) -> bool:
    """True if this session was already injected (marker present)."""
    if not session_id:
        return False
    return (_marker_dir() / _safe_session(session_id)).exists()


def mark_injected(session_id: str) -> None:
    """Record that this session has been injected; prune stale markers."""
    if not session_id:
        return
    directory = _marker_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _safe_session(session_id)).write_text("", encoding="utf-8")
    except OSError:
        logger.debug("reasoning injection marker write failed", exc_info=True)
        return
    _cleanup_markers(directory)


def _cleanup_markers(directory: Path) -> None:
    cutoff = time.time() - _MARKER_MAX_AGE_S
    try:
        for marker in directory.iterdir():
            try:
                if marker.is_file() and marker.stat().st_mtime < cutoff:
                    marker.unlink(missing_ok=True)
            except OSError:
                continue
    except OSError:
        logger.debug("reasoning injection marker cleanup failed", exc_info=True)
