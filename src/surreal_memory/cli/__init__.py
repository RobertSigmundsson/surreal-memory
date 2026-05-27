"""Surreal-Memory CLI.

Simple command-line interface for storing and retrieving memories.

Usage:
    smem remember "content"     Store a memory
    smem recall "query"         Query memories
    smem context                Get recent context
    smem brain list             List brains
    smem brain use <name>       Switch brain
"""

from surreal_memory.cli.main import app, main

__all__ = ["app", "main"]
