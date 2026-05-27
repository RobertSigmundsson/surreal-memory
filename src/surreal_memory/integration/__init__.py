"""External source integration layer for Surreal-Memory.

Import memories from competing systems (ChromaDB, Mem0, Graphiti, etc.)
into Surreal-Memory's neuron/synapse/fiber graph.
"""

from surreal_memory.integration.adapter import SourceAdapter
from surreal_memory.integration.mapper import MappingResult, RecordMapper
from surreal_memory.integration.models import (
    ExternalRecord,
    ExternalRelationship,
    ImportResult,
    SourceCapability,
    SourceSystemType,
    SyncState,
)
from surreal_memory.integration.sync_engine import SyncEngine

__all__ = [
    "ExternalRecord",
    "ExternalRelationship",
    "ImportResult",
    "MappingResult",
    "RecordMapper",
    "SourceAdapter",
    "SourceCapability",
    "SourceSystemType",
    "SyncEngine",
    "SyncState",
]
