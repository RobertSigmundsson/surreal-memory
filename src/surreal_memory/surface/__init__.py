"""Surreal-Memory Knowledge Surface (.nm format).

Two-tier memory: .nm flat file (working memory, always loaded) +
brain.db SQLite graph (long-term, queried on-demand).

The .nm file is an agent-navigable knowledge graph with:
- GRAPH: causal/decisional edges between key concepts
- CLUSTERS: topic groups from entity co-occurrence
- SIGNALS: urgent alerts, unresolved items, pending decisions
- DEPTH MAP: self-routing hints (SUFFICIENT / NEEDS_DETAIL / NEEDS_DEEP)
- META: brain richness stats for agent context
"""

from __future__ import annotations

from surreal_memory.surface.generator import SurfaceGenerator
from surreal_memory.surface.lifecycle import regenerate_surface, show_surface
from surreal_memory.surface.models import (
    Cluster,
    DepthHint,
    DepthLevel,
    GraphEntry,
    KnowledgeSurface,
    Signal,
    SignalLevel,
    SurfaceEdge,
    SurfaceFrontmatter,
    SurfaceMeta,
    SurfaceNode,
)
from surreal_memory.surface.parser import parse
from surreal_memory.surface.resolver import get_surface_path, load_surface_text, save_surface_text
from surreal_memory.surface.serializer import serialize
from surreal_memory.surface.token_budget import trim_to_budget

__all__ = [
    "Cluster",
    "DepthHint",
    "DepthLevel",
    "GraphEntry",
    "KnowledgeSurface",
    "Signal",
    "SignalLevel",
    "SurfaceEdge",
    "SurfaceFrontmatter",
    "SurfaceMeta",
    "SurfaceNode",
    "SurfaceGenerator",
    "get_surface_path",
    "load_surface_text",
    "parse",
    "regenerate_surface",
    "save_surface_text",
    "serialize",
    "show_surface",
    "trim_to_budget",
]
