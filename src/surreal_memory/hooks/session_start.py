"""SessionStart hook: inject recent memories at Claude Code session start.

Called by Claude Code when a new session starts.
Reads recent memories from the brain and outputs them as context,
making prior knowledge available from the very first turn.

Usage as Claude Code hook:
    Reads JSON from stdin (may be empty for SessionStart).
    Outputs {"type": "context", "content": "<markdown>"} to stdout.
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


async def get_recent_memories() -> str:
    """Fetch recent memories from the active brain and format as markdown."""
    from surreal_memory.unified_config import get_config, get_shared_storage

    config = get_config()
    storage = await get_shared_storage(config.current_brain)
    try:
        fibers = await storage.get_fibers(limit=CONTEXT_LIMIT)
        if not fibers:
            return ""

        lines: list[str] = []
        for fiber in fibers:
            text = fiber.summary or fiber.essence
            if text and text.strip():
                lines.append(f"- {text.strip()}")

        return "\n".join(lines)
    finally:
        await storage.close()


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


def main() -> None:
    """Entry point for SessionStart hook."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Consume stdin so Claude Code doesn't block
    read_hook_input()

    try:
        context_text = asyncio.run(get_recent_memories())
    except Exception:
        print("[Surreal-Memory] Session start context load failed", file=sys.stderr)  # noqa: T201
        sys.exit(0)  # Never block Claude Code

    if not context_text:
        print("[Surreal-Memory] No memories to inject at session start", file=sys.stderr)  # noqa: T201
        sys.exit(0)

    count = context_text.count("\n") + 1
    print(f"[Surreal-Memory] Injecting {count} memories at session start", file=sys.stderr)  # noqa: T201

    response = {
        "type": "context",
        "content": f"## Recent Memories\n\n{context_text}",
    }
    print(json.dumps(response))  # noqa: T201


if __name__ == "__main__":
    main()
