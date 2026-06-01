"""Tests for MCP server auth error surfacing (StorageAuthError → -32001)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from surreal_memory.mcp.server import MCPServer, handle_message
from surreal_memory.storage.surrealdb.connection import StorageAuthError


def _make_server() -> MCPServer:
    from surreal_memory.unified_config import ResponseConfig, ToolTierConfig

    with patch("surreal_memory.mcp.server.get_config") as mock_get_config:
        mock_get_config.return_value = MagicMock(
            current_brain="test-brain",
            get_brain_db_path=MagicMock(return_value="/tmp/test.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
            auto=MagicMock(enabled=False),
        )
        return MCPServer()


class TestStorageAuthErrorSurfacing:
    """StorageAuthError from a tool handler must become -32001 with hint in message."""

    @pytest.mark.asyncio
    async def test_storage_auth_error_returns_32001(self) -> None:
        server = _make_server()

        msg = {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "smem_recall", "arguments": {"query": "test"}},
        }

        with patch.object(
            server,
            "call_tool",
            side_effect=StorageAuthError(
                "SurrealDB authentication failed for user 'root' at http://localhost:8001.",
                hint="Set SURREALDB_PASS in your MCP client config or run `smem doctor --fix`.",
            ),
        ):
            resp = await handle_message(server, msg)

        assert resp is not None
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 99
        assert "error" in resp
        assert resp["error"]["code"] == -32001
        assert "SURREALDB_PASS" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_generic_exception_still_returns_32000(self) -> None:
        """Regression: plain Exception must still return -32000."""
        server = _make_server()

        msg = {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {"name": "smem_recall", "arguments": {"query": "test"}},
        }

        with patch.object(
            server,
            "call_tool",
            side_effect=RuntimeError("unexpected failure"),
        ):
            resp = await handle_message(server, msg)

        assert resp is not None
        assert "error" in resp
        assert resp["error"]["code"] == -32000
        assert "failed unexpectedly" in resp["error"]["message"]

    @pytest.mark.asyncio
    async def test_storage_auth_error_does_not_log_password(self) -> None:
        """Password must not appear in any log output."""

        server = _make_server()

        msg = {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {"name": "smem_recall", "arguments": {"query": "test"}},
        }

        with patch.object(
            server,
            "call_tool",
            side_effect=StorageAuthError(
                "auth failed",
                hint="Set SURREALDB_PASS",
            ),
        ):
            with patch("surreal_memory.mcp.server.logger") as mock_logger:
                await handle_message(server, msg)

        # error should be logged, but not with the password
        assert mock_logger.error.called
        logged_args = str(mock_logger.error.call_args_list)
        assert "surrealmemory" not in logged_args.lower()
        assert "password" not in logged_args.lower()
