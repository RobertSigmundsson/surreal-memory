"""``smem recall-http`` — the minimal recall shim for non-MCP callers (hermes pods)."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import typer

KEY_ENV = "SMEM_RECALL_HTTP_KEY"


def configure_logging() -> logging.Logger:
    """One INFO line per recall (tor, agent, query sha8, ms, path, trace) to stderr/journal.

    The root logger stays at WARNING under uvicorn, so without this the per-request line —
    the only record of shim-side latency per caller path — would never be emitted.
    """
    log = logging.getLogger("surreal_memory.recall_http")
    if not any(getattr(h, "_recall_http", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler._recall_http = True  # type: ignore[attr-defined]
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


def recall_http(
    host: Annotated[str, typer.Option("--host", help="Host to bind to")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port to bind to")] = 18400,
    max_concurrency: Annotated[
        int, typer.Option("--max-concurrency", help="Recalls executed at once")
    ] = 4,
    queue_timeout_s: Annotated[
        float, typer.Option("--queue-timeout", help="Seconds to wait for a slot, then 503")
    ] = 2.0,
    trace: Annotated[
        str,
        typer.Option("--trace", help="force = every recall writes a trace; config = obey [trace]"),
    ] = "force",
    reconsolidate: Annotated[
        bool,
        typer.Option("--reconsolidate/--no-reconsolidate", help="Recall reconsolidates top hits"),
    ] = True,
) -> None:
    """Run the recall-only HTTP shim (GET /health, POST /v1/recall; bearer auth; no daemons).

    The bearer secret is read from the SMEM_RECALL_HTTP_KEY environment variable; a missing
    or short key is a hard error (exit 78), never an unauthenticated server.
    """
    key = os.environ.get(KEY_ENV, "")
    if len(key) < 32:
        typer.echo(f"ERROR: {KEY_ENV} missing or shorter than 32 characters", err=True)
        raise typer.Exit(78)
    if trace not in ("force", "config"):
        typer.echo("ERROR: --trace must be 'force' or 'config'", err=True)
        raise typer.Exit(2)
    try:
        import uvicorn
    except ImportError:
        typer.echo(
            "Error: uvicorn not installed. Run: pip install surreal-memory[server]", err=True
        )
        raise typer.Exit(1)

    from surreal_memory.recall_http import create_app

    configure_logging()

    app = create_app(
        key=key,
        max_concurrency=max_concurrency,
        queue_timeout_s=queue_timeout_s,
        trace_mode="force" if trace == "force" else "config",
        reconsolidate=reconsolidate,
    )
    del key
    typer.echo(f"smem recall-http on http://{host}:{port} (routes: /health, /v1/recall)")
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        access_log=False,
        server_header=False,
        log_level="info",
    )


def register(app: typer.Typer) -> None:
    """Register the recall-http command."""
    app.command(name="recall-http")(recall_http)
