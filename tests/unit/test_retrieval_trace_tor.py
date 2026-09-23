"""RetrievalTrace carries the caller path (tor) and caller identity (agent_id).

Both are additive: old serialized traces (no such keys) read back as tor="" and
agent_id=None — never silently relabelled as "mcp".
"""

from __future__ import annotations

from typing import Any

import pytest

from surreal_memory.core.retrieval_trace import RetrievalTrace
from surreal_memory.engine.trace_builder import build_retrieval_trace
from surreal_memory.storage.surrealdb.retrieval_trace import (
    _PAYLOAD_KEYS,
    SurrealDBRetrievalTraceMixin,
    _row_to_retrieval_trace,
)


def test_roundtrip_keeps_tor_and_agent_id() -> None:
    t = RetrievalTrace(brain_id="b", query="q", tor="http:gateway", agent_id="agent:nemo_1")
    back = RetrievalTrace.from_dict(t.to_dict())
    assert (back.tor, back.agent_id) == ("http:gateway", "agent:nemo_1")


def test_legacy_dict_without_fields_reads_as_unknown_not_mcp() -> None:
    d = RetrievalTrace(brain_id="b", query="q").to_dict()
    d.pop("tor")
    d.pop("agent_id")
    back = RetrievalTrace.from_dict(d)
    assert back.tor == ""
    assert back.agent_id is None


def test_fields_are_bounded() -> None:
    t = RetrievalTrace(tor="http:" + "x" * 100, agent_id="a" * 500)
    assert len(t.tor) == 40
    assert t.agent_id is not None and len(t.agent_id) == 120


def test_builder_passes_tor_and_agent_id_and_defaults_are_neutral() -> None:
    res: Any = object()
    t = build_retrieval_trace(
        res, query="q", brain_id="b", mode="associative", tor="mcp", agent_id="c"
    )
    assert (t.tor, t.agent_id) == ("mcp", "c")
    t0 = build_retrieval_trace(res, query="q", brain_id="b", mode="associative")
    assert (t0.tor, t0.agent_id) == ("", None)


def test_surrealdb_payload_carries_tor_and_agent_id() -> None:
    assert "tor" in _PAYLOAD_KEYS and "agent_id" in _PAYLOAD_KEYS
    t = RetrievalTrace(brain_id="b", query="q", tor="http:hermes-pod", agent_id="agent:s_p_1")
    full = t.to_dict()
    row = {
        "id": "retrieval_trace:x",
        "brain_id": "b",
        "query": "q",
        "payload": {**{k: full[k] for k in _PAYLOAD_KEYS}, "_orig_id": t.id},
    }
    back = _row_to_retrieval_trace(row)
    assert (back.tor, back.agent_id, back.id) == ("http:hermes-pod", "agent:s_p_1", t.id)


@pytest.mark.asyncio
async def test_surrealdb_add_writes_payload_tor() -> None:
    sent: dict[str, Any] = {}

    class _Conn:
        async def query(self, sql: str, params: dict[str, Any]) -> list[Any]:
            sent.update(params)
            return []

    class _S(SurrealDBRetrievalTraceMixin):
        def _ensure_conn(self) -> Any:
            return _Conn()

        def _get_brain_id(self) -> str:
            return "b"

    await _S().add_retrieval_trace(RetrievalTrace(query="q", tor="http:test", agent_id="k2"))
    assert sent["data"]["payload"]["tor"] == "http:test"
    assert sent["data"]["payload"]["agent_id"] == "k2"
