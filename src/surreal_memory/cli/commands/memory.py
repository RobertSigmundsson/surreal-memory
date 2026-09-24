"""Core memory commands: remember, todo, recall, context."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from surreal_memory.cli.storage import PersistentStorage

import typer

from surreal_memory.cli._helpers import get_config, get_storage, output_result, run_async
from surreal_memory.cli.recall_trace import CliTraceOutcome, persist_cli_trace
from surreal_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
)
from surreal_memory.engine import remember_api
from surreal_memory.engine.cli_recall_api import recall_like_cli
from surreal_memory.engine.dedup.factory import build_dedup_pipeline
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.safety.freshness import (
    FreshnessLevel,
    analyze_freshness,
    evaluate_freshness,
    format_age,
    get_freshness_indicator,
)
from surreal_memory.safety.sensitive import (
    format_sensitive_warning,
)
from surreal_memory.utils.timeutils import utcnow


async def _resolve_project_id(storage: PersistentStorage, project: str | None) -> str | None:
    """Look up project by name, return ID or None."""
    if not project:
        return None
    proj = await storage.get_project_by_name(project)
    if not proj:
        return None
    return proj.id


def remember(
    content: Annotated[str, typer.Argument(help="Content to remember")] = "",
    tags: Annotated[
        list[str] | None, typer.Option("--tag", "-t", help="Tags for the memory")
    ] = None,
    memory_type: Annotated[
        str | None,
        typer.Option(
            "--type",
            "-T",
            help="Memory type: fact, decision, preference, todo, insight, context, instruction, error, workflow, reference (auto-detected if not specified)",
        ),
    ] = None,
    priority: Annotated[
        int | None,
        typer.Option("--priority", "-p", help="Priority 0-10 (0=lowest, 5=normal, 10=critical)"),
    ] = None,
    expires: Annotated[
        int | None,
        typer.Option("--expires", "-e", help="Days until this memory expires"),
    ] = None,
    project: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Associate with a project (by name)"),
    ] = None,
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage for this command")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Store even if sensitive content detected")
    ] = False,
    redact: Annotated[
        bool, typer.Option("--redact", "-r", help="Auto-redact sensitive content before storing")
    ] = False,
    timestamp: Annotated[
        str | None,
        typer.Option(
            "--timestamp",
            "--at",
            help="ISO datetime of original event (e.g. '2026-03-02T08:00:00'). Defaults to now.",
        ),
    ] = None,
    ephemeral: Annotated[
        bool,
        typer.Option(
            "--ephemeral", help="Session-scoped memory: auto-expires after 24h, never synced"
        ),
    ] = False,
    stdin: Annotated[
        bool,
        typer.Option("--stdin", help="Read content from stdin (safe for shell-special characters)"),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Store a new memory (type auto-detected if not specified).

    Examples:
        smem remember "Fixed auth bug by adding null check"
        smem remember "We decided to use PostgreSQL" --type decision
        smem remember "Need to refactor auth module" --type todo --priority 7
        smem remember "API_KEY=xxx" --redact
        smem remember "Meeting at 8am" --timestamp "2026-03-02T08:00:00"
        smem remember "Debug note" --ephemeral
        echo "content with backticks" | smem remember --stdin --type context
    """
    import sys

    if stdin:
        content = sys.stdin.read().strip()
    if not content:
        typer.secho(
            "Error: content is required (pass as argument or use --stdin).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    try:
        chk = remember_api.check_content(content, force=force, redact=redact)
    except remember_api.SensitiveContentError as exc:
        typer.echo(format_sensitive_warning(list(exc.matches)))
        raise typer.Exit(1)
    if chk.redacted:
        typer.secho(f"Redacted {len(chk.matches)} sensitive item(s)", fg=typer.colors.YELLOW)
    store_content = chk.content
    try:
        mem_type = remember_api.resolve_memory_type(memory_type, store_content)
    except remember_api.InvalidMemoryTypeError as exc:
        typer.secho(
            f"Invalid memory type. Valid types: {', '.join(exc.valid)}", fg=typer.colors.RED
        )
        raise typer.Exit(1)
    expiry_days = remember_api.resolve_expiry_days(mem_type, expires, ephemeral=ephemeral)
    mem_priority, priority_was_explicit = remember_api.resolve_priority(priority)

    # Parse --timestamp for original event time
    event_timestamp: datetime | None = None
    if timestamp:
        try:
            event_timestamp = datetime.fromisoformat(timestamp)
            if event_timestamp.tzinfo is not None:
                event_timestamp = event_timestamp.replace(tzinfo=None)
        except (ValueError, TypeError):
            typer.secho(
                f"Invalid timestamp format: {timestamp}. Use ISO format (e.g. '2026-03-02T08:00:00').",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

    async def _remember() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config, force_shared=shared)

        brain_id: str = (
            storage.brain_id or "" if hasattr(storage, "brain_id") else config.current_brain
        )
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}

        project_id = await _resolve_project_id(storage, project)
        if project and project_id is None:
            return {
                "error": f"Project '{project}' not found. Create it with: smem project create \"{project}\""
            }

        stored = await remember_api.encode_and_store(
            storage,
            brain.config,
            store_content,
            tags=set(tags) if tags else None,
            mem_type=mem_type,
            mem_priority=mem_priority,
            expiry_days=expiry_days,
            project_id=project_id,
            event_timestamp=event_timestamp,
            ephemeral=ephemeral,
            priority_was_explicit=priority_was_explicit,
            attribution=remember_api.CLI_ATTRIBUTION,
        )
        return remember_api.response_dict(
            stored,
            content=store_content,
            mem_type=mem_type,
            mem_priority=mem_priority,
            ephemeral=ephemeral,
            project=project if project_id else None,
            forced_matches=len(chk.matches) if force else 0,
        )

    result = run_async(_remember())
    output_result(result, json_output)


def todo(
    task: Annotated[str, typer.Argument(help="Task to remember")],
    priority: Annotated[
        int,
        typer.Option(
            "--priority", "-p", help="Priority 0-10 (default: 5=normal, 7=high, 10=critical)"
        ),
    ] = 5,
    project: Annotated[
        str | None,
        typer.Option("--project", "-P", help="Associate with a project"),
    ] = None,
    expires: Annotated[
        int | None,
        typer.Option("--expires", "-e", help="Days until expiry (default: 30)"),
    ] = None,
    tags: Annotated[
        list[str] | None,
        typer.Option("--tag", "-t", help="Tags for the task"),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Quick shortcut to add a TODO memory.

    Examples:
        smem todo "Fix the login bug"
        smem todo "Review PR #123" --priority 7
    """
    # Determine expiry (default 30 days for todos)
    expiry_days = expires if expires is not None else 30
    mem_priority = Priority.from_int(priority)

    async def _todo() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config)

        brain = await storage.get_brain(storage.brain_id or "")
        if not brain:
            return {"error": "No brain configured"}

        # Look up project if specified
        project_id = None
        if project:
            proj = await storage.get_project_by_name(project)
            if not proj:
                return {
                    "error": f"Project '{project}' not found. Create it with: smem project create \"{project}\""
                }
            project_id = proj.id

        encoder = MemoryEncoder(storage, brain.config, dedup_pipeline=build_dedup_pipeline(storage))
        storage.disable_auto_save()

        result = await encoder.encode(
            content=task,
            timestamp=utcnow(),
            tags=set(tags) if tags else None,
        )

        # Create TODO typed memory
        typed_mem = TypedMemory.create(
            fiber_id=result.fiber.id,
            memory_type=MemoryType.TODO,
            priority=mem_priority,
            source="user_input",
            expires_in_days=expiry_days,
            tags=set(tags) if tags else None,
            project_id=project_id,
        )
        await storage.add_typed_memory(typed_mem)
        await storage.batch_save()

        response = {
            "message": f"TODO: {task[:50]}{'...' if len(task) > 50 else ''}",
            "fiber_id": result.fiber.id,
            "memory_type": "todo",
            "priority": mem_priority.name.lower(),
            "expires_in_days": typed_mem.days_until_expiry,
        }

        if project_id:
            response["project"] = project

        return response

    result = run_async(_todo())
    output_result(result, json_output)


def recall(
    query: Annotated[str, typer.Argument(help="Query to search memories")],
    depth: Annotated[
        int | None,
        typer.Option("--depth", "-d", help="Search depth (0=instant, 1=context, 2=habit, 3=deep)"),
    ] = None,
    max_tokens: Annotated[
        int, typer.Option("--max-tokens", "-m", help="Max tokens in response")
    ] = 500,
    min_confidence: Annotated[
        float, typer.Option("--min-confidence", "-c", help="Minimum confidence threshold (0.0-1.0)")
    ] = 0.0,
    shared: Annotated[
        bool, typer.Option("--shared", "-S", help="Use shared/remote storage for this command")
    ] = False,
    show_age: Annotated[
        bool, typer.Option("--show-age", "-a", help="Show memory ages in results")
    ] = True,
    show_routing: Annotated[
        bool, typer.Option("--show-routing", "-R", help="Show query routing info")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    trace: Annotated[
        bool | None,
        typer.Option(
            "--trace/--no-trace",
            help="Retrieval trace: default obeys [trace] in config; --trace forces one; "
            "--no-trace writes none",
        ),
    ] = None,
) -> None:
    """Query memories with intelligent routing (query type auto-detected).

    Examples:
        smem recall "What did I do with auth?"
        smem recall "meetings with Alice" --depth 2
        smem recall "Why did the build fail?" --show-routing
        smem recall "project status" --min-confidence 0.5
    """

    async def _recall() -> tuple[dict[str, Any], CliTraceOutcome | None]:
        config = get_config()
        storage = await get_storage(config, force_shared=shared)

        brain_id: str = (
            storage.brain_id or "" if hasattr(storage, "brain_id") else config.current_brain
        )
        brain = await storage.get_brain(brain_id)
        if not brain:
            return {"error": "No brain configured"}, None

        async def _po_zapytaniu(res: Any, depth_value: int) -> CliTraceOutcome:
            # persist_cli_trace is looked up in this module at call time (tests patch it here).
            return await persist_cli_trace(
                storage,
                res,
                brain=brain,
                query=query,
                depth=depth_value,
                max_tokens=max_tokens,
                min_confidence=min_confidence,
                flag=trace,
            )

        return await recall_like_cli(
            storage,
            brain,
            query=query,
            depth=depth,
            max_tokens=max_tokens,
            min_confidence=min_confidence,
            show_routing=show_routing,
            show_age=show_age,
            po_zapytaniu=_po_zapytaniu,
        )

    result, slad = run_async(_recall())
    output_result(result, json_output)
    line = slad.stderr_line() if slad is not None else None
    if line:
        typer.echo(line, err=True)


def context(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Number of recent memories")] = 10,
    fresh_only: Annotated[
        bool, typer.Option("--fresh-only", help="Only include memories < 30 days old")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Get recent context (for injecting into AI conversations).

    Examples:
        smem context
        smem context --limit 5 --json
        smem context --fresh-only
    """

    async def _context() -> dict[str, Any]:
        config = get_config()
        storage = await get_storage(config)

        # Get recent fibers
        fibers = await storage.get_fibers(limit=limit * 2 if fresh_only else limit)

        if not fibers:
            return {"context": "No memories stored yet.", "count": 0}

        # Filter by freshness if requested
        now = utcnow()
        if fresh_only:
            fresh_fibers = []
            for fiber in fibers:
                freshness = evaluate_freshness(fiber.created_at, now)
                if freshness.level in (FreshnessLevel.FRESH, FreshnessLevel.RECENT):
                    fresh_fibers.append(fiber)
            fibers = fresh_fibers[:limit]

        # Build context string with age indicators
        context_parts = []
        fiber_data = []

        for fiber in fibers:
            freshness = evaluate_freshness(fiber.created_at, now)
            indicator = get_freshness_indicator(freshness.level)
            age_str = format_age(freshness.age_days)

            fiber_content = fiber.summary
            if not fiber_content and fiber.anchor_neuron_id:
                anchor = await storage.get_neuron(fiber.anchor_neuron_id)
                if anchor:
                    fiber_content = anchor.content

            if fiber_content:
                context_parts.append(f"{indicator} [{age_str}] {fiber_content}")
                fiber_data.append(
                    {
                        "id": fiber.id,
                        "summary": fiber_content,
                        "created_at": fiber.created_at.isoformat(),
                        "age": age_str,
                        "freshness": freshness.level.value,
                    }
                )

        context_str = "\n".join(context_parts) if context_parts else "No context available."

        # Analyze overall freshness
        created_dates = [f.created_at for f in fibers]
        freshness_report = analyze_freshness(created_dates, now)

        return {
            "context": context_str,
            "count": len(fiber_data),
            "fibers": fiber_data,
            "freshness_summary": {
                "fresh": freshness_report.fresh,
                "recent": freshness_report.recent,
                "aging": freshness_report.aging,
                "stale": freshness_report.stale,
                "ancient": freshness_report.ancient,
            },
        }

    result = run_async(_context())
    output_result(result, json_output)


def register(app: typer.Typer) -> None:
    """Register memory commands on the app."""
    app.command()(remember)
    app.command()(todo)
    app.command()(recall)
    app.command()(context)
