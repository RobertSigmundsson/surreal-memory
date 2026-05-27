"""MCP (Model Context Protocol) server for Surreal-Memory.

This module provides an MCP server that exposes Surreal-Memory tools
to Claude Code, Claude Desktop, and other MCP-compatible clients.
"""

from surreal_memory.mcp.server import create_mcp_server, main, run_mcp_server

__all__ = ["create_mcp_server", "main", "run_mcp_server"]
