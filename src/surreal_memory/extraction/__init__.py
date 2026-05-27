"""Extraction modules for parsing queries and content."""

from surreal_memory.extraction.entities import Entity, EntityExtractor
from surreal_memory.extraction.parser import (
    Perspective,
    QueryIntent,
    QueryParser,
    Stimulus,
)
from surreal_memory.extraction.relations import (
    RelationCandidate,
    RelationExtractor,
    RelationType,
)
from surreal_memory.extraction.router import (
    QueryRouter,
    QueryType,
    RouteConfidence,
    RouteDecision,
    route_query,
)
from surreal_memory.extraction.temporal import (
    TemporalExtractor,
    TimeGranularity,
    TimeHint,
)

__all__ = [
    # Temporal
    "TimeHint",
    "TimeGranularity",
    "TemporalExtractor",
    # Parser
    "Stimulus",
    "QueryIntent",
    "Perspective",
    "QueryParser",
    # Router (MemoCore integration)
    "QueryRouter",
    "QueryType",
    "RouteConfidence",
    "RouteDecision",
    "route_query",
    # Entities
    "Entity",
    "EntityExtractor",
    # Relations
    "RelationCandidate",
    "RelationExtractor",
    "RelationType",
]
