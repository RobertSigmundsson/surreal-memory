"""RecallHandler._recall delegates to engine/recall_api with tor="mcp".

The MCP tool schema has no ``tor`` argument and ``tor``/``agent_id`` are never read from
``args``: a client passing ``tor`` in the tool call cannot relabel its trace.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.engine import recall_api
from surreal_memory.mcp.server import MCPServer
from surreal_memory.unified_config import ResponseConfig, ToolTierConfig, TraceConfig


def _server() -> MCPServer:
    with patch("surreal_memory.mcp.server.get_config") as g:
        cfg = MagicMock(
            current_brain="b",
            get_brain_db_path=MagicMock(return_value="/tmp/b.db"),
            tool_tier=ToolTierConfig(tier="full"),
            response=ResponseConfig(),
            trace=TraceConfig(),
        )
        cfg.write_gate.enabled = False
        g.return_value = cfg
        return MCPServer()


@pytest.mark.asyncio
async def test_recall_delegates_with_mcp_tor_and_engine_session() -> None:
    server = _server()
    storage = object()
    captured: dict[str, Any] = {}

    async def _fake(st: Any, args: dict[str, Any], **kw: Any) -> recall_api.RecallOutcome:
        captured.update(kw, storage=st, args=args)
        return recall_api.RecallOutcome({"answer": "x"}, None, "pipeline", "off", "b")

    with (
        patch.object(server, "get_storage", AsyncMock(return_value=storage)),
        patch.object(recall_api, "recall", _fake),
    ):
        out = await server._recall({"query": "q", "tor": "http:gateway", "agent_id": "spoof"})
    assert out == {"answer": "x"}
    assert captured["tor"] == "mcp"
    assert "agent_id" not in captured  # MCP passes no agent identity (None default)
    assert captured["engine_session_id"] == f"mcp-{id(server)}"
    assert captured["hooks"] is server.hooks
    assert captured["trace_tasks"] is server._trace_tasks
    assert captured["storage"] is storage


@pytest.mark.asyncio
async def test_cross_brain_does_not_go_through_recall_api() -> None:
    server = _server()
    with (
        patch.object(server, "_cross_brain_recall", AsyncMock(return_value={"cb": 1})),
        patch.object(recall_api, "recall", AsyncMock(side_effect=AssertionError("called"))),
    ):
        assert await server._recall({"query": "q", "brains": ["a", "b"]}) == {"cb": 1}
