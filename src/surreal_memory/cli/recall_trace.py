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

import logging
import os
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from surreal_memory.engine import recall_api

logger = logging.getLogger(__name__)

ENV_AGENT_ID: Final = "SMEM_AGENT_ID"
ENV_ENTRYPOINT: Final = "CLAUDE_CODE_ENTRYPOINT"
ENV_SESSION_ID: Final = "CLAUDE_CODE_SESSION_ID"
AGENT_PREFIX_CLAUDE_CODE: Final = "claude-code:"
AGENT_DEFAULT: Final = "cli"
MARKER: Final = "SMEM-SLAD-BLAD"

CliTraceStatus = Literal["sync", "sync_error", "off", "disabled", "identity_error"]
AgentSource = Literal["SMEM_AGENT_ID", "CLAUDE_CODE_ENTRYPOINT", "default"]


class CliIdentityError(ValueError):
    """Invalid identity in the environment. Carries the variable NAME and the reason, never the value."""

    def __init__(self, variable: str, reason: str) -> None:
        super().__init__(f"{variable}:{reason}")
        self.variable = variable
        self.reason = reason


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


def cli_trace_wanted(
    trace_cfg: Any, flag: bool | None, *, draw: Callable[[], float] = random.random
) -> bool:
    """``True``/``False`` flag wins; no flag = ``[trace]`` enabled + sample_rate (as persist_trace)."""
    if flag is not None:
        return flag
    if not trace_cfg.enabled:
        return False
    return not (trace_cfg.sample_rate < 1.0 and draw() >= trace_cfg.sample_rate)


@dataclass(frozen=True)
class CliTraceOutcome:
    status: CliTraceStatus
    trace_id: str | None = None
    error: str | None = None  # no env values, no query text

    def json_fields(self) -> dict[str, Any]:
        out: dict[str, Any] = {"trace_status": self.status}
        if self.trace_id is not None:
            out["trace_id"] = self.trace_id
        if self.error is not None:
            out["trace_error"] = self.error
        return out

    def stderr_line(self) -> str | None:
        if self.status in ("sync", "off", "disabled"):
            return None
        return f"{MARKER} tor={recall_api.TOR_CLI} status={self.status} powod={self.error}"


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
    """Write the trace of one CLI recall. Never raises; never touches the pipeline."""
    if flag is False:
        return CliTraceOutcome("disabled")
    try:
        if config is None:
            # The UNIFIED config — the CLI's own CLIConfig has no [trace] section.
            from surreal_memory.unified_config import get_config

            config = get_config()
        trace_cfg = getattr(config, "trace", None)
        if trace_cfg is None:
            logger.warning("CLI recall trace: config has no [trace] section")
            return CliTraceOutcome("sync_error", error="brak-sekcji-trace")
        if not cli_trace_wanted(trace_cfg, flag):
            return CliTraceOutcome("off")
        try:
            ident = resolve_cli_identity(os.environ if env is None else env)
        except CliIdentityError as exc:
            logger.warning("CLI recall trace: invalid identity in %s", exc.variable)
            return CliTraceOutcome("identity_error", error=str(exc))

        # The pipeline got the original query; only the trace copy is made encodable.
        q = query.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        args: dict[str, Any] = {
            "query": q,
            "depth": depth,
            "max_tokens": max_tokens,
            "session_id": ident.session_id,
            "trace": True,
        }
        if min_confidence > 0.0:
            args["min_confidence"] = min_confidence
        sink: dict[str, Any] = {}
        st = await recall_api.persist_trace(
            sink,
            result,
            query=q,
            args=args,
            brain=brain,
            mode="associative",
            storage=storage,
            config=config,
            tor=recall_api.TOR_CLI,
            agent_id=ident.agent_id,
            trace_tasks=None,
        )
        trace_id = sink.get("trace_id")
        if st == "sync" and trace_id:
            return CliTraceOutcome("sync", trace_id=str(trace_id))
        # Per-call persist returns "sync" or "sync_error"; "off" here means it raised internally.
        logger.warning("CLI recall trace not persisted: status=%s", st)
        return CliTraceOutcome(
            "sync_error", error=str(sink.get("trace_error") or f"persist_trace-status={st}")
        )
    except Exception as exc:
        logger.warning("CLI recall trace failed: %s", type(exc).__name__)
        return CliTraceOutcome("sync_error", error=f"wyjatek-{type(exc).__name__}")
