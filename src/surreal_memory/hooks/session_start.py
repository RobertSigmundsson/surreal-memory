"""SessionStart hook: inject recent memories at Claude Code session start.

Called by Claude Code when a new session starts.
Reads recent memories from the brain and outputs them as context,
making prior knowledge available from the very first turn.

Usage as Claude Code hook:
    Reads JSON from stdin (may be empty for SessionStart).
    Outputs hookSpecificOutput JSON (hookEventName + additionalContext) to stdout.
    Status messages go to stderr.

Usage standalone:
    echo '{}' | python -m surreal_memory.hooks.session_start
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

CONTEXT_LIMIT = 10
# Per-memory bullet length cap for the injected context block.
BULLET_MAX_CHARS = 300


async def get_recent_memories(project_name: str | None) -> str:
    """Fetch recent memories scoped to the current project, as markdown.

    Memories are filtered to the project the agent is working in (the shared
    brain holds every project's memories). When the project has no memories
    yet — or no project could be resolved — returns an empty string rather
    than leaking other projects' context.
    """
    from surreal_memory.engine.superseded_filter import is_excluded_by_validity
    from surreal_memory.hooks import liczniki
    from surreal_memory.unified_config import get_config, get_shared_storage

    config = get_config()
    storage = await get_shared_storage(config.current_brain)
    try:
        if not project_name:
            liczniki.zwieksz({"sessionstart_wywolania": 1})
            return ""

        # The repo name doubles as the project id (opaque scope label).
        typed = await storage.get_project_memories(project_name)
        if not typed:
            liczniki.zwieksz({"sessionstart_wywolania": 1})
            return ""

        # Superseded facts (typed_memory.valid_until set) are dropped from the list — the same predicate as recall
        # (`is_excluded_by_validity`, same escape hatch). The filter runs AFTER the newest-N cut, so the list is a
        # strict subset of what it was: no older memory takes the place of a dropped one (that would change what
        # the user reads, not just remove stale entries).
        lines: list[str] = []
        odfiltrowane = 0
        for tm in typed[:CONTEXT_LIMIT]:
            if is_excluded_by_validity(tm, valid_at=None, include_superseded=False):
                odfiltrowane += 1
                continue
            fiber = await storage.get_fiber(tm.fiber_id)
            if fiber is None:
                continue
            text = fiber.summary or fiber.essence
            if text and text.strip():
                snippet = text.strip()
                if len(snippet) > BULLET_MAX_CHARS:
                    snippet = snippet[:BULLET_MAX_CHARS].rstrip() + "…"
                lines.append(f"- {snippet}")

        # No retrieval_trace here (a list, not a recall — kept out of tor `cli`); a daily counter instead, so the
        # filter's work on prod is visible without a synthetic trace row.
        liczniki.zwieksz(
            {
                "sessionstart_wywolania": 1,
                "sessionstart_pokazane": len(lines),
                "sessionstart_odfiltrowane_nieaktualne": odfiltrowane,
            }
        )
        return "\n".join(lines)
    finally:
        try:
            await storage.close()
        except Exception:
            logger.debug("storage.close() failed (non-fatal)", exc_info=True)


def read_hook_input() -> dict[str, Any]:
    """Read Claude Code hook JSON from stdin."""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return {}
        result: dict[str, Any] = json.loads(raw)
        return result
    except (json.JSONDecodeError, OSError):
        return {}


async def get_reasoning_injection(hook_input: dict[str, Any]) -> str:
    """Build the reasoning-strategies context block for this session (or "").

    Thin delegate to the shared engine orchestrator, which SessionStart and the
    UserPromptSubmit hook both call. Opt-in via reasoning_training.injection_enabled;
    injects at most once per session via a marker shared between the two hooks.
    """
    from surreal_memory.engine.reasoning_injection import get_reasoning_context

    return await get_reasoning_context(hook_input)


def main() -> None:
    """Entry point for SessionStart hook."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Consume stdin so Claude Code doesn't block; also carries `cwd`.
    hook_input = read_hook_input()

    from surreal_memory.hooks.project_context import derive_project_name

    project_name = derive_project_name(hook_input)
    sections: list[str] = []

    # Recent-memories block (scoped to the resolved project).
    if project_name:
        try:
            memories = asyncio.run(get_recent_memories(project_name))
        except Exception:
            memories = ""
            print("[Surreal-Memory] Session start context load failed", file=sys.stderr)  # noqa: T201
        if memories:
            sections.append(f"## Recent Memories — project: {project_name}\n\n{memories}")

    # Reasoning-strategies block (opt-in; model-based, not project-scoped).
    try:
        reasoning = asyncio.run(get_reasoning_injection(hook_input))
    except Exception:
        reasoning = ""
        print("[Surreal-Memory] Session start reasoning injection failed", file=sys.stderr)  # noqa: T201
    if reasoning:
        sections.append(reasoning)

    if not sections:
        print("[Surreal-Memory] No session-start context to inject", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    # Claude Code adds a SessionStart hook's context to the model ONLY via the
    # hookSpecificOutput.additionalContext JSON field (a top-level {"type":"context"}
    # is parsed as JSON but carries no recognized field, so nothing is injected).
    # Echo back the event we were actually invoked for instead of hardcoding
    # SessionStart: this entry point is also wired to SubagentStart so subagents
    # receive memories + reasoning strategies, and a hookEventName that disagrees
    # with the live event makes Claude Code drop additionalContext — the hook
    # would appear to run while injecting nothing.
    event_name = str(hook_input.get("hook_event_name") or "SessionStart")
    print(  # noqa: T201
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": event_name,
                    "additionalContext": "\n\n".join(sections),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
