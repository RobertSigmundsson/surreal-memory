"""Nanobot integration for Surreal-Memory.

Provides Surreal-Memory tools that conform to Nanobot's Tool interface,
plus a drop-in MemoryStore replacement backed by the neural graph.

Usage::

    from surreal_memory.integrations.nanobot import setup_surreal_memory

    nm_store = await setup_surreal_memory(registry, workspace, brain_id="my-brain")
    # Tools are now registered. nm_store can replace Nanobot's MemoryStore.
"""

from surreal_memory.integrations.nanobot.context import NMContext
from surreal_memory.integrations.nanobot.memory_store import NMMemoryStore
from surreal_memory.integrations.nanobot.setup import setup_surreal_memory
from surreal_memory.integrations.nanobot.tools import (
    NMContextTool,
    NMHealthTool,
    NMRecallTool,
    NMRememberTool,
)

__all__ = [
    "NMContext",
    "NMContextTool",
    "NMHealthTool",
    "NMMemoryStore",
    "NMRecallTool",
    "NMRememberTool",
    "setup_surreal_memory",
]
