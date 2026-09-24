"""``smem recall-http`` — the minimal recall shim for non-MCP callers (hermes pods)."""

from __future__ import annotations

import logging
import os
from typing import Annotated

import typer

KEY_ENV = "SMEM_RECALL_HTTP_KEY"
REMEMBER_KEY_ENV = "SMEM_REMEMBER_HTTP_KEY"


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
    skutki: Annotated[
        str,
        typer.Option(
            "--skutki",
            help="inline = side effects before the answer (today); odroczone = answer first, "
            "side effects + trace after it, awaited before the next recall",
        ),
    ] = "inline",
    bariera_s: Annotated[
        float, typer.Option("--bariera-s", help="Max wait for pending side effects before a recall")
    ] = 10.0,
    zapis: Annotated[
        bool,
        typer.Option(
            "--zapis/--bez-zapisu",
            help="Enable POST /v1/remember with the write key from SMEM_REMEMBER_HTTP_KEY "
            "(off = the route answers 403 zapis_wylaczony)",
        ),
    ] = False,
) -> None:
    """Run the HTTP shim (GET /health, POST /v1/recall, /v1/recall-cli, /v1/remember; bearer
    auth with disjoint read/write scopes; no daemons).

    The read bearer comes from SMEM_RECALL_HTTP_KEY; a missing or short key is a hard error
    (exit 78), never an unauthenticated server. Writing is enabled only by --zapis, and then the
    write bearer SMEM_REMEMBER_HTTP_KEY must be present, >= 32 characters and differ from the
    read key (else exit 78) — a write key merely present in the environment never enables writes.
    """
    key = os.environ.get(KEY_ENV, "")
    if len(key) < 32:
        typer.echo(f"ERROR: {KEY_ENV} missing or shorter than 32 characters", err=True)
        raise typer.Exit(78)
    remember_key: str | None = None
    if zapis:
        remember_key = os.environ.get(REMEMBER_KEY_ENV, "")
        if len(remember_key) < 32:
            typer.echo(f"ERROR: {REMEMBER_KEY_ENV} missing or shorter than 32 characters", err=True)
            raise typer.Exit(78)
        if remember_key == key:
            typer.echo(f"ERROR: {REMEMBER_KEY_ENV} must differ from {KEY_ENV}", err=True)
            raise typer.Exit(78)
    if skutki not in ("inline", "odroczone"):
        typer.echo("ERROR: --skutki must be 'inline' or 'odroczone'", err=True)
        raise typer.Exit(2)
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

    log = configure_logging()
    if remember_key is None:
        log.warning("recall-http: zapis wylaczony (brak --zapis)")
        if os.environ.get(REMEMBER_KEY_ENV):
            log.warning("recall-http: klucz zapisu obecny, ale zapis wylaczony — brak --zapis")

    app = create_app(
        key=key,
        remember_key=remember_key,
        max_concurrency=max_concurrency,
        queue_timeout_s=queue_timeout_s,
        trace_mode="force" if trace == "force" else "config",
        reconsolidate=reconsolidate,
        skutki="odroczone" if skutki == "odroczone" else "inline",
        bariera_s=bariera_s,
    )
    del key, remember_key
    typer.echo(
        f"smem recall-http on http://{host}:{port} "
        f"(routes: /health, /v1/recall, /v1/recall-cli, /v1/remember; zapis={'on' if zapis else 'off'})"
    )
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
