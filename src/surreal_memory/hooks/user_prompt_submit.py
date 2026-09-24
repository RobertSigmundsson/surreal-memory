"""UserPromptSubmit hook: recall relevant memory + inject reasoning strategies.

TWO blocks, on different schedules and for different reasons:

* **Memory recall (per turn, opt-in).** SessionStart can only LIST the newest
  memories for the project — it runs before the user has said anything, so it
  has nothing to search with. This hook is the only one that holds the actual
  question, so it is the only place a topic-keyed query is possible. Without it
  the same newest memories are redelivered every session and everything older
  never surfaces (measured on this brain: 51% of neurons never accessed, recall
  confidence 20%).
* **Reasoning strategies (once per session).** Unchanged behaviour, below.

SessionStart runs before any assistant turn exists, so the active model often
can't be resolved yet. From the second prompt on, the model is resolvable from
the transcript tail, so this hook injects model-appropriate reasoning strategies
that SessionStart may have missed. It shares the once-per-session marker with
SessionStart (whichever fires first wins), so the two never double-inject.

Opt-in via reasoning_training.injection_enabled.

Claude Code injects a UserPromptSubmit hook's context ONLY via the
``hookSpecificOutput.additionalContext`` JSON field on stdout (exit 0). Plain
stdout is echoed to the transcript but is NOT added to the model's context, so
the block is emitted inside that JSON envelope.

Usage as Claude Code hook:
    Reads JSON from stdin (session_id, transcript_path, cwd, prompt).
    Emits the reasoning block as hookSpecificOutput JSON on stdout (or nothing).
    Status messages go to stderr. Always exits 0 — never blocks the prompt.

Usage standalone:
    echo '{}' | python -m surreal_memory.hooks.user_prompt_submit
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Rough chars-per-token used to turn the configured token ceiling into a hard
# character cut. Deliberately crude: it only has to bound the injection, and a
# cheap over-estimate is better than a tokenizer import on every prompt.
_CHARS_PER_TOKEN = 4


# Durable record of every prompt-recall skip for system content. Hook stderr is
# not persisted by Claude Code (measured 2026-09-24: 0 occurrences in transcripts),
# so a filter reporting only there would be a filter without a trace.
_SKIP_LOG = "prompt_recall_pominiete.jsonl"


def _data_dir() -> Path:
    custom = os.environ.get("SURREAL_MEMORY_DIR", "")
    return Path(custom) if custom else (Path.home() / ".surrealmemory")


def _record_skip(prefix: str, length: int, session: str) -> None:
    """Append one line per skip — prefix, length, session; NEVER the prompt text
    (bash-mode stdout can carry secrets). A failed write degrades visibly on stderr;
    the skip itself still holds and the prompt is never blocked."""
    rekord = {
        "ts": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "powod": "prefiks",
        "prefiks": prefix,
        "dlugosc": length,
        "sesja": session,
    }
    try:
        with open(_data_dir() / _SKIP_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rekord, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(  # noqa: T201
            f"[Surreal-Memory] licznik pominięć: zapis nieudany ({type(exc).__name__})",
            file=sys.stderr,
        )
        return
    print(  # noqa: T201
        f"[Surreal-Memory] prompt recall pominięty: treść systemowa ({prefix}) -> {_SKIP_LOG}",
        file=sys.stderr,
    )


# Every recall from a host path leaves a retrieval_trace (tor ``cli``), the hook
# included: it runs on every longer prompt, so without a trace the most frequent
# host recall was invisible to recall telemetry (V-GATE r1 F-3, 2026-09-24).
# The hook's agent_id is kept apart from an explicit ``smem recall`` in the same
# session (``claude-code:<entrypoint>``) so the two can be counted separately.
HOOK_AGENT_PREFIX = "claude-code-hook:"
HOOK_AGENT_DEFAULT = "cli-hook"
HOOK_AGENT_SUFFIX = ":hook"
_TRACE_ERROR_LOG = "prompt_recall_slad_bledy.jsonl"


def resolve_hook_identity(
    hook_input: dict[str, Any], env: dict[str, str] | Any
) -> tuple[str, str | None]:
    """``agent_id`` and ``session_id`` of one hook recall.

    agent_id: ``SMEM_AGENT_ID`` + ``:hook`` -> ``claude-code-hook:<CLAUDE_CODE_ENTRYPOINT>``
    -> ``cli-hook``. session_id: the hook input's ``session_id`` -> ``CLAUDE_CODE_SESSION_ID``.

    Raises:
        IdentityError: an id does not match the trace patterns (no silent substitution).
    """
    from surreal_memory.engine import recall_api
    from surreal_memory.engine.cli_recall_api import IdentityError

    raw_agent = env.get("SMEM_AGENT_ID") or ""
    entrypoint = env.get("CLAUDE_CODE_ENTRYPOINT") or ""
    if raw_agent:
        agent_id, variable = raw_agent + HOOK_AGENT_SUFFIX, "SMEM_AGENT_ID"
    elif entrypoint:
        agent_id, variable = HOOK_AGENT_PREFIX + entrypoint, "CLAUDE_CODE_ENTRYPOINT"
    else:
        agent_id, variable = HOOK_AGENT_DEFAULT, "default"
    if not recall_api.AGENT_ID_PATTERN.match(agent_id):
        raise IdentityError(variable, "niepoprawny-agent_id")
    session_id = (
        str(hook_input.get("session_id") or "") or env.get("CLAUDE_CODE_SESSION_ID") or None
    )
    if session_id is not None and not recall_api.SESSION_ID_PATTERN.match(session_id):
        raise IdentityError("session_id", "niepoprawny-session_id")
    return agent_id, session_id


def _record_trace_error(line: str, session: str) -> None:
    """Hook stderr is not persisted — a trace that failed is also written to a file
    (status and reason only, never the prompt). Never raises."""
    rekord = {
        "ts": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "blad": line,
        "sesja": session,
    }
    try:
        with open(_data_dir() / _TRACE_ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rekord, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(  # noqa: T201
            f"[Surreal-Memory] dziennik błędów śladu: zapis nieudany ({type(exc).__name__})",
            file=sys.stderr,
        )


async def _persist_hook_trace(
    storage: Any,
    result: Any,
    *,
    brain: Any,
    prompt: str,
    max_tokens: int,
    hook_input: dict[str, Any],
    config: Any,
) -> None:
    """One retrieval_trace (tor ``cli``) for this hook recall. Never raises, never
    blocks the prompt, never mutates ``result``; a failure is visible on stderr and in
    ``prompt_recall_slad_bledy.jsonl``."""
    from surreal_memory.engine import recall_api
    from surreal_memory.engine.cli_recall_api import persist_identified_trace

    try:
        depth = int(result.depth_used.value)
    except (AttributeError, TypeError, ValueError):
        depth = 1
    outcome = await persist_identified_trace(
        storage,
        result,
        brain=brain,
        query=prompt,
        depth=depth,
        max_tokens=max_tokens,
        min_confidence=0.0,
        flag=None,
        tor=recall_api.TOR_CLI,
        identity=lambda: resolve_hook_identity(hook_input, os.environ),
        config=config,
    )
    line = outcome.stderr_line()
    if line:
        print(line, file=sys.stderr)  # noqa: T201
        _record_trace_error(line, str(hook_input.get("session_id") or ""))


def read_hook_input() -> dict[str, Any]:
    """Read Claude Code hook JSON from stdin (empty/malformed -> {})."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


async def get_prompt_recall(hook_input: dict[str, Any]) -> str:
    """Recall memories relevant to THIS prompt, or "" when there is nothing to do.

    Bounded on purpose — it runs on every turn:
    * a prompt shorter than ``min_prompt_chars`` carries no query worth making
      ("ok", "tak", "dalej"), so it is skipped rather than answered with noise;
    * the retrieval is capped by ``max_tokens`` so memory cannot crowd out the
      conversation it is supposed to help;
    * the whole thing runs under ``timeout_seconds`` — memory that makes the
      user wait is worse than memory that stays quiet.

    Any failure degrades to "": the prompt must never be blocked by recall.
    """
    from surreal_memory.engine.retrieval import ReflexPipeline
    from surreal_memory.unified_config import (
        DEFAULT_SYSTEM_PREFIXES,
        get_config,
        get_shared_storage,
    )

    config = get_config()
    cfg = config.prompt_recall
    if not cfg.enabled:
        return ""
    prompt = str(hook_input.get("prompt") or "").strip()
    # Claude Code system content (task notifications, bash-mode input) is not a
    # question: no recall, no Jev call, no storage connection. Checked before the
    # length gate so the recorded reason is always the specific one.
    # A config object without the field (older or duck-typed) keeps the filter ON.
    for prefix in getattr(cfg, "system_prefixes", DEFAULT_SYSTEM_PREFIXES):
        if prefix and prompt.startswith(prefix):
            _record_skip(prefix, len(prompt), str(hook_input.get("session_id") or ""))
            return ""
    if len(prompt) < cfg.min_prompt_chars:
        return ""

    storage = await get_shared_storage(config.current_brain)
    try:
        brain_id = storage.brain_id or config.current_brain
        brain = await storage.get_brain(brain_id)
        if brain is None:
            return ""
        pipeline = ReflexPipeline(storage, brain.config)
        result = await pipeline.query(
            query=prompt,
            max_tokens=cfg.max_tokens,
            session_id=str(hook_input.get("session_id") or "ups"),
        )
        await _persist_hook_trace(
            storage,
            result,
            brain=brain,
            prompt=prompt,
            max_tokens=cfg.max_tokens,
            hook_input=hook_input,
            config=config,
        )
        context = (result.context or "").strip()
        if not context:
            return ""
        # The pipeline formats its own "## Relevant Memories" heading. Keeping it
        # under ours put two headings on every single prompt, which is noise the
        # user pays for each turn — drop the inner one and keep the heading that
        # says WHY this block is here.
        first, sep, rest = context.partition("\n")
        if first.strip().lower().startswith("## relevant memories"):
            context = rest.strip() if sep else ""
            if not context:
                return ""
        # ReflexPipeline treats max_tokens as a TARGET, not a ceiling: measured on
        # this brain it overshoots by ~70% consistently (100 -> 174, 600 -> 1009,
        # 4000 -> 2433 estimated tokens). A config field named max_tokens that does
        # not cap is a promise the code does not keep, and this runs every turn —
        # so enforce the ceiling here and say when it bit.
        limit = max(1, cfg.max_tokens) * _CHARS_PER_TOKEN
        if len(context) > limit:
            context = context[:limit].rstrip() + f"\n\n[…przycięte do {cfg.max_tokens} tokenów]"
        return f"## Relevant memory (recalled for this prompt)\n\n{context}"
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("storage.close() failed (non-fatal)", exc_info=True)


async def _recall_within_timeout(hook_input: dict[str, Any], seconds: float) -> str:
    """Run the recall under a wall-clock cap; a slow brain yields nothing, not a stall."""
    try:
        return await asyncio.wait_for(get_prompt_recall(hook_input), timeout=seconds)
    except TimeoutError:
        print(  # noqa: T201
            f"[Surreal-Memory] prompt recall exceeded {seconds}s — skipped", file=sys.stderr
        )
        return ""


def main() -> None:
    """Entry point for the UserPromptSubmit hook."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    hook_input = read_hook_input()

    from surreal_memory.engine.reasoning_injection import get_reasoning_context

    sections: list[str] = []

    # Memory relevant to THIS prompt (per turn, opt-in).
    try:
        from surreal_memory.unified_config import get_config

        timeout = get_config().prompt_recall.timeout_seconds
        recalled = asyncio.run(_recall_within_timeout(hook_input, timeout))
    except Exception:
        recalled = ""
        print("[Surreal-Memory] UserPromptSubmit memory recall failed", file=sys.stderr)  # noqa: T201
    if recalled:
        sections.append(recalled)

    # Reasoning strategies (once per session; marker shared with SessionStart).
    try:
        strategies = asyncio.run(get_reasoning_context(hook_input))
    except Exception:
        strategies = ""
        # Never block the prompt — degrade to no injection.
        print("[Surreal-Memory] UserPromptSubmit reasoning injection failed", file=sys.stderr)  # noqa: T201
    if strategies:
        sections.append(strategies)

    block = "\n\n".join(sections)
    if block:
        # Context reaches the model ONLY through hookSpecificOutput.additionalContext
        # (plain stdout is transcript-only for this event).
        print(  # noqa: T201
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "UserPromptSubmit",
                        "additionalContext": block,
                    }
                }
            )
        )
    else:
        print("[Surreal-Memory] Nothing to inject (no recall, no strategies)", file=sys.stderr)  # noqa: T201

    sys.exit(0)


if __name__ == "__main__":
    main()
