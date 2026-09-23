"""MCP handler mixin for recall and context tools."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from surreal_memory.engine import recall_api
from surreal_memory.engine.hooks import HookEvent
from surreal_memory.mcp.constants import MAX_HOT_CONTEXT_MEMORIES
from surreal_memory.utils.timeutils import utcnow

if TYPE_CHECKING:
    from surreal_memory.engine.hooks import HookRegistry
    from surreal_memory.mcp.maintenance_handler import HealthPulse
    from surreal_memory.storage.base import NeuralStorage
    from surreal_memory.unified_config import UnifiedConfig

logger = logging.getLogger(__name__)


class _McpRecallExtras:
    """MCP-only recall side effects, invoked by ``recall_api.recall`` at their original positions.

    Each hook resolves the handler method at call time, so ``patch.object(server, ...)`` in
    tests keeps intercepting exactly as before the recall body moved to the engine.
    """

    __slots__ = ("_h",)

    def __init__(self, handler: RecallHandler) -> None:
        self._h = handler

    async def active_session(self, storage: NeuralStorage) -> dict[str, Any] | None:
        return await self._h._get_active_session(storage)

    def surface_depth(self, query: str) -> tuple[dict[str, Any] | None, int | None]:
        return self._h._check_surface_depth(query)

    async def after_query(self, query: str) -> None:
        # Passive auto-capture on long queries
        if self._h.config.auto.enabled and len(query) >= 50:
            await self._h._passive_capture(query)

        self._h._fire_eternal_trigger(query)

    async def decorate(
        self,
        response: dict[str, Any],
        result: Any,
        query: str,
        brain: Any,
        storage: NeuralStorage,
    ) -> None:
        await self._h._record_tool_action("recall", query[:100])

        pulse = await self._h._check_maintenance()
        hint = self._h._get_maintenance_hint(pulse)
        if hint:
            response["maintenance_hint"] = hint

        update_hint = self._h.get_update_hint()
        if update_hint:
            response["update_hint"] = update_hint

        await self._h.hooks.emit(
            HookEvent.POST_RECALL,
            {
                "query": query,
                "confidence": result.confidence,
                "neurons_activated": result.neurons_activated,
                "fibers_matched": result.fibers_matched,
            },
        )

        # Suggest related queries from learned patterns
        try:
            from surreal_memory.engine.query_pattern_mining import (
                extract_topics,
                suggest_follow_up_queries,
            )

            topics = extract_topics(query)
            if topics:
                related = await suggest_follow_up_queries(storage, topics, brain.config)
                if related:
                    response["related_queries"] = related
        except Exception:
            logger.debug("Query pattern suggestion failed", exc_info=True)

        # Onboarding hint for fresh brains
        onboarding = await self._h._check_onboarding()
        if onboarding:
            response["onboarding"] = onboarding

        # Pro hint: when many fibers matched but results were truncated
        fibers_count = getattr(result, "fibers_matched", 0)
        if isinstance(fibers_count, int) and fibers_count > 10:
            pro_hints = response.get("pro_hints", [])
            pro_hints.append(
                f"Showing top results from {fibers_count} matches. "
                "Pro: Cone queries return ALL relevant memories for exhaustive recall."
            )
            response["pro_hints"] = pro_hints

        # Surface pending alerts count
        alert_info = await self._h._surface_pending_alerts()
        if alert_info:
            response.update(alert_info)


class RecallHandler:
    """Mixin providing recall and context MCP tool handlers."""

    if TYPE_CHECKING:
        config: UnifiedConfig
        hooks: HookRegistry
        _surface_text: str
        _surface_brain: str

        async def get_storage(self) -> NeuralStorage:
            raise NotImplementedError

        def _fire_eternal_trigger(self, content: str) -> None:
            raise NotImplementedError

        async def _check_maintenance(self) -> HealthPulse | None:
            raise NotImplementedError

        def _get_maintenance_hint(self, pulse: HealthPulse | None) -> str | None:
            raise NotImplementedError

        async def _passive_capture(self, text: str) -> None:
            raise NotImplementedError

        async def _get_active_session(self, storage: NeuralStorage) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _check_onboarding(self) -> dict[str, Any] | None:
            raise NotImplementedError

        def get_update_hint(self) -> dict[str, Any] | None:
            raise NotImplementedError

        async def _surface_pending_alerts(self) -> dict[str, int] | None:
            raise NotImplementedError

        async def _record_tool_action(self, action_type: str, context: str = "") -> None:
            raise NotImplementedError

    # ──────────────────── Surface Depth Routing ────────────────────

    def _check_surface_depth(
        self,
        query: str,
    ) -> tuple[dict[str, Any] | None, int | None]:
        """Check the Knowledge Surface DEPTH MAP for recall routing.

        If the query matches a SUFFICIENT entity, returns surface context
        directly (no brain.db query needed). For NEEDS_DETAIL or NEEDS_DEEP,
        returns a suggested depth override.

        Args:
            query: The recall query string.

        Returns:
            Tuple of (surface_response_or_None, depth_override_or_None).
            If surface_response is not None, caller should return it immediately.
        """
        if not hasattr(self, "_surface_text") or not self._surface_text:
            return None, None

        try:
            from surreal_memory.surface.models import DepthLevel
            from surreal_memory.surface.parser import parse

            surface = parse(self._surface_text)
        except Exception:
            return None, None

        # Normalize query for matching
        query_lower = query.lower().strip()

        # Find matching entity in graph nodes
        for entry in surface.graph:
            node = entry.node
            if query_lower in node.content.lower() or node.content.lower() in query_lower:
                depth_level = surface.get_depth_hint(node.id)
                if depth_level == DepthLevel.SUFFICIENT:
                    # Build context from surface graph
                    context_parts = [f"[{node.id}] {node.content} ({node.node_type})"]
                    for edge in entry.edges:
                        if edge.target_id:
                            context_parts.append(
                                f"  →{edge.edge_type}→ [{edge.target_id}] {edge.target_text}"
                            )
                        else:
                            context_parts.append(f"  →{edge.edge_type}→ {edge.target_text}")

                    # Add cluster context if available
                    for cluster in surface.clusters:
                        if node.id in cluster.node_ids:
                            context_parts.append(f"  @{cluster.name}: {cluster.description}")

                    return {
                        "answer": "\n".join(context_parts),
                        "confidence": 0.8,
                        "source": "knowledge_surface",
                        "depth_hint": "SUFFICIENT",
                        "message": "Answered from Knowledge Surface (no brain.db query needed)",
                    }, None

                elif depth_level == DepthLevel.NEEDS_DEEP:
                    return None, 2

                elif depth_level == DepthLevel.NEEDS_DETAIL:
                    return None, 1

        return None, None

    async def _recall(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query memories via spreading activation (body lives in ``engine/recall_api``)."""
        # Cross-brain recall: early return if brains parameter is provided.
        # NOTE: retrieval tracing (trace=true / sampling) is intentionally NOT applied
        # to cross-brain recall — the merged multi-brain result has no single brain_id
        # or RetrievalResult to attribute a trace to. Documented in the smem_recall
        # 'trace' schema description.
        brain_names = args.get("brains")
        if brain_names and isinstance(brain_names, list) and len(brain_names) > 0:
            return await self._cross_brain_recall(args, brain_names)

        storage = await self.get_storage()
        if getattr(self, "_trace_tasks", None) is None:
            self._trace_tasks: set[asyncio.Task[None]] = set()
        outcome = await recall_api.recall(
            storage,
            args,
            config=self.config,
            tor=recall_api.TOR_MCP,
            engine_session_id=f"mcp-{id(self)}",
            hooks=self.hooks,
            extras=_McpRecallExtras(self),
            trace_tasks=self._trace_tasks,
        )
        return outcome.response

    async def _cross_brain_recall(
        self, args: dict[str, Any], brain_names: list[str]
    ) -> dict[str, Any]:
        """Handle cross-brain recall by querying multiple brains in parallel."""
        from surreal_memory.engine.cross_brain import cross_brain_recall
        from surreal_memory.mcp.tool_handler_utils import _parse_tags

        query = args.get("query", "")
        if not query:
            return {"error": "query is required"}

        # Validate and cap at 5 brains
        import re

        _brain_pattern = re.compile(r"^[a-zA-Z0-9_\-\.]+$")
        brain_names = [n for n in brain_names[:5] if isinstance(n, str) and _brain_pattern.match(n)]
        if not brain_names:
            return {"error": "No valid brain names provided"}
        try:
            depth = int(args.get("depth", 1))
            depth = max(0, min(depth, 3))
        except (TypeError, ValueError):
            depth = 1
        max_tokens = min(int(args.get("max_tokens", 500)), 10_000)

        tags = _parse_tags(args)

        # U8: geospatial filter also applies across brains.
        near = None
        if "near" in args:
            from surreal_memory.utils.geo import parse_geo_filter

            try:
                near = parse_geo_filter(args["near"])
            except (ValueError, TypeError) as exc:
                return {"error": f"Invalid near filter: {exc}"}

        try:
            result = await cross_brain_recall(
                config=self.config,
                brain_names=brain_names,
                query=query,
                depth=depth,
                max_tokens=max_tokens,
                tags=tags,
                near=near,
            )
        except Exception:
            logger.error("Cross-brain recall failed", exc_info=True)
            return {"error": "Cross-brain recall failed"}

        fibers_out = [
            {
                "fiber_id": f.fiber_id,
                "source_brain": f.source_brain,
                "summary": f.summary,
                "confidence": f.confidence,
            }
            for f in result.fibers
        ]

        response: dict[str, Any] = {
            "answer": result.merged_context,
            "brains_queried": result.brains_queried,
            "total_neurons_activated": result.total_neurons_activated,
            "fibers": fibers_out,
            "cross_brain": True,
        }
        # Only include errors when a brain query actually failed — an empty
        # dict here would be indistinguishable noise on every response.
        if result.errors:
            response["errors"] = result.errors
        return response

    async def _context(self, args: dict[str, Any]) -> dict[str, Any]:
        """Get recent context.

        Note: HOT-tier memories are always injected regardless of fresh_only.
        This is intentional — HOT memories represent always-in-context data
        (safety boundaries, pinned knowledge) that should never be excluded.
        """
        storage = await self.get_storage()

        limit = min(args.get("limit", 10), 200)
        fresh_only = args.get("fresh_only", False)

        fibers = await storage.get_fibers(
            limit=limit * 2 if fresh_only else limit, exclude_expired=True
        )
        if not fibers:
            result: dict[str, Any] = {"context": "No memories stored yet.", "count": 0}
            onboarding = await self._check_onboarding()
            if onboarding:
                result["onboarding"] = onboarding
            return result

        if fresh_only:
            from surreal_memory.safety.freshness import FreshnessLevel, evaluate_freshness

            now = utcnow()
            fresh_fibers = [
                f
                for f in fibers
                if evaluate_freshness(f.created_at, now).level
                in (FreshnessLevel.FRESH, FreshnessLevel.RECENT)
            ]
            fibers = fresh_fibers[:limit]

        # Inject HOT tier memories — always in context regardless of recency
        existing_ids = {f.id for f in fibers}
        try:
            hot_memories = await storage.find_typed_memories(
                tier="hot", limit=MAX_HOT_CONTEXT_MEMORIES
            )
            for tm in hot_memories:
                if tm.fiber_id not in existing_ids:
                    hot_fiber = await storage.get_fiber(tm.fiber_id)
                    if hot_fiber:
                        fibers.append(hot_fiber)
                        existing_ids.add(tm.fiber_id)
            if len(hot_memories) >= MAX_HOT_CONTEXT_MEMORIES:
                logger.warning(
                    "HOT memory limit reached (%d) — some HOT memories may be excluded from context",
                    MAX_HOT_CONTEXT_MEMORIES,
                )
        except Exception as e:
            logger.warning("HOT memory injection failed — tier filter unavailable: %s", e)

        # Smart context optimization: score, dedup, budget
        from surreal_memory.engine.context_optimizer import optimize_context

        try:
            max_tokens = int(self.config.brain.max_context_tokens)
            if max_tokens < 100:
                max_tokens = 4000
        except (TypeError, ValueError, AttributeError):
            max_tokens = 4000
        # Pass fidelity config — fetch from storage brain (BrainConfig has fidelity fields)
        # BrainSettings (self.config.brain) does NOT have fidelity fields
        brain_obj = await storage.get_brain(storage.brain_id) if storage.brain_id else None
        brain_config = brain_obj.config if brain_obj else None

        # Build embed_fn for anisotropic compression (if embedding enabled)
        embed_fn = None
        if brain_config and brain_config.embedding_enabled:
            try:
                from surreal_memory.engine.semantic_discovery import _create_provider

                provider = _create_provider(brain_config)
                embed_fn = provider.embed
            except Exception:
                logger.debug("Embedding provider unavailable for anisotropic compression")

        plan = await optimize_context(
            storage,
            fibers,
            max_tokens,
            fidelity_enabled=brain_config.fidelity_enabled if brain_config else True,
            fidelity_full_threshold=brain_config.fidelity_full_threshold if brain_config else 0.6,
            fidelity_summary_threshold=brain_config.fidelity_summary_threshold
            if brain_config
            else 0.3,
            fidelity_essence_threshold=brain_config.fidelity_essence_threshold
            if brain_config
            else 0.1,
            decay_rate=brain_config.decay_rate if brain_config else 0.1,
            decay_floor=brain_config.decay_floor if brain_config else 0.05,
            embed_fn=embed_fn,
        )

        include_ghosts = args.get("include_ghosts", True)

        if plan.items:
            # Separate ghost items from non-ghost items
            non_ghost = [item for item in plan.items if item.fidelity_level != "ghost"]
            ghost_items = [item for item in plan.items if item.fidelity_level == "ghost"]

            context_parts = [f"- {item.content}" for item in non_ghost]
            context_text = "\n".join(context_parts) if context_parts else ""

            # Append ghost section if enabled and ghosts exist
            if include_ghosts and ghost_items:
                ghost_parts = [f"- {item.content}" for item in ghost_items]
                ghost_section = "\n--- faded memories (use recall key to restore) ---\n"
                ghost_section += "\n".join(ghost_parts)
                context_text = (
                    (context_text + "\n" + ghost_section) if context_text else ghost_section
                )

            if not context_text:
                context_text = "No context available."
        else:
            context_text = "No context available."

        # Track ghost shown timestamps (only if ghosts were actually shown)
        if include_ghosts and plan.ghost_fiber_ids:
            try:
                from surreal_memory.utils.timeutils import utcnow as _utcnow

                now = _utcnow()
                # Batch update: single SQL for all ghost fibers (avoids N round-trips)
                await storage.batch_update_ghost_shown(plan.ghost_fiber_ids, now)
            except Exception:
                logger.debug("Ghost tracking update failed", exc_info=True)

        await self._record_tool_action("context")

        response: dict[str, Any] = {
            "context": context_text,
            "count": len(plan.items),
            "tokens_used": plan.total_tokens,
        }

        if plan.dropped_count > 0:
            response["optimization_stats"] = {
                "items_dropped": plan.dropped_count,
                "top_score": round(plan.items[0].score, 4) if plan.items else 0.0,
            }

        # Fidelity stats — always include when fidelity is enabled so callers
        # can distinguish "fidelity off" from "all items at FULL"
        fidelity_on = brain_config.fidelity_enabled if brain_config else True
        fs = plan.fidelity_stats
        if fidelity_on:
            response["fidelity_stats"] = {
                "full": fs.full,
                "summary": fs.summary,
                "essence": fs.essence,
                "ghost": fs.ghost,
            }

        # Expiry warnings (opt-in)
        warn_expiry_days = args.get("warn_expiry_days")
        if warn_expiry_days is not None and fibers:
            try:
                fiber_ids = [f.id for f in fibers]
                expiring = await storage.get_expiring_memories_for_fibers(
                    fiber_ids=fiber_ids,
                    within_days=int(warn_expiry_days),
                )
                if expiring:
                    response["expiry_warnings"] = [
                        {
                            "fiber_id": tm.fiber_id,
                            "memory_type": tm.memory_type.value,
                            "days_until_expiry": tm.days_until_expiry,
                            "priority": tm.priority.value,
                            "suggestion": "Re-store this memory if still relevant, or set a new expires_days.",
                        }
                        for tm in expiring
                    ]
            except Exception:
                logger.debug("Expiry warning check failed", exc_info=True)

        # Surface pending alerts count
        alert_info = await self._surface_pending_alerts()
        if alert_info:
            response.update(alert_info)

        return response
