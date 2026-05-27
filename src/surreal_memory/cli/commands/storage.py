"""CLI commands for managing the SurrealDB storage backend."""

from __future__ import annotations

import logging

import typer

logger = logging.getLogger(__name__)

storage_app = typer.Typer(
    name="storage",
    help="Manage the SurrealDB storage backend",
    no_args_is_help=True,
)


@storage_app.command("status")
def storage_status() -> None:
    """Show SurrealDB connection status and active brain.

    Reports the configured SurrealDB URL / namespace / database, whether
    the connection succeeds, and which brain the current process targets.

    Examples:
        smem storage status
    """
    import asyncio
    import os

    from surreal_memory.unified_config import get_config

    cfg = get_config(reload=True)
    brain_name = cfg.current_brain

    typer.secho("Storage Status", bold=True)
    typer.echo(f"  Brain:            {brain_name}")
    typer.echo(f"  Configured backend: {cfg.storage_backend}")
    typer.echo()

    typer.secho("SurrealDB Endpoint", bold=True)
    typer.echo(f"  URL:        {os.getenv('SURREALDB_URL', 'http://localhost:8001')}")
    typer.echo(f"  Namespace:  {os.getenv('SURREALDB_NS', 'surreal_memory')}")
    typer.echo(f"  Database:   {os.getenv('SURREALDB_DB', 'default')}")
    typer.echo()

    # Connection probe
    async def _probe() -> tuple[bool, str]:
        try:
            from surreal_memory.storage.surrealdb import SurrealDBStorage

            storage = SurrealDBStorage()
            await storage.initialize()
            await storage.close()
        except Exception as exc:
            return False, str(exc)
        return True, ""

    ok, err = asyncio.run(_probe())
    if ok:
        typer.secho("SurrealDB: reachable", fg=typer.colors.GREEN)
    else:
        typer.secho("SurrealDB: unreachable", fg=typer.colors.RED)
        if err:
            typer.echo(f"  {err}")
        typer.echo()
        typer.echo("Hint: `docker compose -f docker-compose.surrealdb.yml up -d`")


def register(app: typer.Typer) -> None:
    """Register storage commands with the main app."""
    app.add_typer(storage_app, name="storage")
