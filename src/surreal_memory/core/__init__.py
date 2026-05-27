"""Core data models for Surreal-Memory."""

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.brain_mode import (
    BrainMode,
    BrainModeConfig,
    HybridConfig,
    SharedConfig,
    SyncStrategy,
)
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import (
    Confidence,
    MemoryType,
    Priority,
    Provenance,
    TypedMemory,
    suggest_memory_type,
)
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.core.project import MemoryScope, Project
from surreal_memory.core.source import Source, SourceStatus, SourceType
from surreal_memory.core.synapse import Direction, Synapse, SynapseType

__all__ = [
    # Brain
    "Brain",
    "BrainConfig",
    # Brain mode (local/shared toggle)
    "BrainMode",
    "BrainModeConfig",
    "SharedConfig",
    "HybridConfig",
    "SyncStrategy",
    # Memory structures
    "Fiber",
    "Neuron",
    "NeuronState",
    "NeuronType",
    "Synapse",
    "SynapseType",
    "Direction",
    # Memory types (MemoCore integration)
    "MemoryType",
    "Priority",
    "Confidence",
    "Provenance",
    "TypedMemory",
    "suggest_memory_type",
    # Source tracking
    "Source",
    "SourceType",
    "SourceStatus",
    # Project scoping
    "Project",
    "MemoryScope",
]
