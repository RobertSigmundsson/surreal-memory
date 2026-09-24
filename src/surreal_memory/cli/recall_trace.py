"""Retrieval trace for CLI recall (tor ``"cli"``).

``smem recall`` / ``smem q`` keep their own engine call (``ReflexPipeline.query`` with the same
arguments as before) and AFTER it write one ``retrieval_trace`` through
``recall_api.persist_trace`` — the same builder the MCP tool and the recall-http shim use — so the
recall result is unchanged and the trace describes exactly what the caller printed.

Identity comes from the environment of the calling process (Claude Code sets it for every Bash
call): ``SMEM_AGENT_ID`` → ``claude-code:<CLAUDE_CODE_ENTRYPOINT>`` → ``"cli"``; the session is
``CLAUDE_CODE_SESSION_ID``. The first SET source is binding: an invalid value writes no trace and
says so (never a silent fall-through to another identity).

Whether a trace is written: ``--no-trace`` → never; ``--trace`` → always; no flag → the ``[trace]``
section of the unified config (``enabled`` + ``sample_rate``), like MCP. The write is synchronous,
so a failure is visible: one ``SMEM-SLAD-BLAD`` line on stderr and ``trace_status``/``trace_error``
in ``--json`` — the recall itself still succeeds (exit code unchanged).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from surreal_memory.engine import recall_api
from surreal_memory.engine.cli_recall_api import (
    MARKER,
    IdentityError,
    TraceOutcome,
    persist_identified_trace,
    trace_wanted,
)

ENV_AGENT_ID: Final = "SMEM_AGENT_ID"
ENV_ENTRYPOINT: Final = "CLAUDE_CODE_ENTRYPOINT"
ENV_SESSION_ID: Final = "CLAUDE_CODE_SESSION_ID"
AGENT_PREFIX_CLAUDE_CODE: Final = "claude-code:"
AGENT_DEFAULT: Final = "cli"

AgentSource = Literal["SMEM_AGENT_ID", "CLAUDE_CODE_ENTRYPOINT", "default"]

# The trace machinery lives in engine.cli_recall_api (shared with the recall-http shim); these
# are the CLI names for it.
CliIdentityError = IdentityError
CliTraceOutcome = TraceOutcome
cli_trace_wanted = trace_wanted
__all__ = [
    "MARKER",
    "CliIdentity",
    "CliIdentityError",
    "CliTraceOutcome",
    "cli_trace_wanted",
    "persist_cli_trace",
    "resolve_cli_identity",
]


@dataclass(frozen=True)
class CliIdentity:
    agent_id: str
    session_id: str | None
    agent_source: AgentSource


def _reason(value: str, limit: int) -> str:
    return f"za-dlugi({len(value)})" if len(value) > limit else "niedozwolony-znak"


def resolve_cli_identity(env: Mapping[str, str]) -> CliIdentity:
    """Resolve ``agent_id`` and ``session_id`` from the environment.

    Raises:
        CliIdentityError: the first set identity source (or the session) is not a valid id.
    """
    raw_agent = env.get(ENV_AGENT_ID) or ""
    entrypoint = env.get(ENV_ENTRYPOINT) or ""
    source: AgentSource
    if raw_agent:
        agent_id, source, variable = raw_agent, "SMEM_AGENT_ID", ENV_AGENT_ID
    elif entrypoint:
        agent_id = AGENT_PREFIX_CLAUDE_CODE + entrypoint
        source, variable = "CLAUDE_CODE_ENTRYPOINT", ENV_ENTRYPOINT
    else:
        agent_id, source, variable = AGENT_DEFAULT, "default", ""
    if not recall_api.AGENT_ID_PATTERN.match(agent_id):
        raise CliIdentityError(variable, _reason(agent_id, recall_api.AGENT_ID_MAX))

    session_id = env.get(ENV_SESSION_ID) or None
    if session_id is not None and not recall_api.SESSION_ID_PATTERN.match(session_id):
        raise CliIdentityError(ENV_SESSION_ID, _reason(session_id, 128))
    return CliIdentity(agent_id=agent_id, session_id=session_id, agent_source=source)


async def persist_cli_trace(
    storage: Any,
    result: Any,
    *,
    brain: Any,
    query: str,
    depth: int,
    max_tokens: int,
    min_confidence: float,
    flag: bool | None,
    env: Mapping[str, str] | None = None,
    config: Any | None = None,
) -> CliTraceOutcome:
    """Write the trace of one CLI recall (tor ``cli``). Never raises; never touches the pipeline."""

    def _identity() -> tuple[str, str | None]:
        ident = resolve_cli_identity(os.environ if env is None else env)
        return ident.agent_id, ident.session_id

    return await persist_identified_trace(
        storage,
        result,
        brain=brain,
        query=query,
        depth=depth,
        max_tokens=max_tokens,
        min_confidence=min_confidence,
        flag=flag,
        tor=recall_api.TOR_CLI,
        identity=_identity,
        config=config,
    )
