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
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


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
    from surreal_memory.unified_config import get_config, get_shared_storage

    config = get_config()
    cfg = config.prompt_recall
    if not cfg.enabled:
        return ""
    prompt = str(hook_input.get("prompt") or "").strip()
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
        context = (result.context or "").strip()
        if not context:
            return ""
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
