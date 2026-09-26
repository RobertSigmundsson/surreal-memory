"""#255 context-overflow split on BOTH reranker transports.

A real local HTTP/1.1 server (same shape as test_reranker_pool.py) answers like LiteLLM in front of a
4096-token reranker. The pooled path (KEEPALIVE_S > 0, http.client) and the urllib path
(KEEPALIVE_S = 0) must split the same way. The server records the User-Agent, which tells the two
transports apart (urllib sends Python-urllib/x.y, http.client sends none).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from surreal_memory.engine import reranker as rr

_OVERFLOW = b"This model's maximum context length is 4096 tokens. However, you requested 5000 tokens (input_tokens)"


class _State:
    def __init__(self) -> None:
        self.posts: list[list[str]] = []
        self.user_agents: list[str] = []
        self.limit = 350
        self.status_for_all: int | None = None
        self.body_for_all = b""


@pytest.fixture
def serwer():
    st = _State()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._send(200, b'{"status":"ok"}')

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            docs = body["documents"]
            st.posts.append(docs)
            st.user_agents.append(self.headers.get("User-Agent") or "")
            if st.status_for_all is not None:
                self._send(st.status_for_all, st.body_for_all)
                return
            if sum(map(len, docs)) > st.limit:
                self._send(400, json.dumps({"error": {"message": _OVERFLOW.decode()}}).encode())
                return
            scores = [
                {"index": i, "relevance_score": 9.0 if "needle" in d else 0.1}
                for i, d in enumerate(docs)
            ]
            self._send(200, json.dumps({"results": scores}).encode())

        def log_message(self, *a: Any) -> None:
            return None

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    st.url = f"http://127.0.0.1:{srv.server_address[1]}"  # type: ignore[attr-defined]
    rr._POOL.clear()
    yield st
    for p in list(rr._POOL.values()):
        p.conn.close()
    rr._POOL.clear()
    srv.shutdown()
    srv.server_close()


_DOCS = ["short", "a" * 300 + "needle" + "b" * 300, "other"]


@pytest.mark.parametrize("keepalive", [4.0, 0.0], ids=["pooled", "urllib"])
def test_context_overflow_splits_on_both_transports(serwer, monkeypatch, keepalive: float) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", keepalive)
    opened_before = rr.POOL_STATS["opened"]
    splits_before = rr.POOL_STATS["context_split"]
    scores = rr.HttpReranker(endpoint=serwer.url, model_name="m")._raw_scores("q", list(_DOCS))
    assert scores == [0.1, 9.0, 0.1]
    assert len(serwer.posts) > 3  # the batch split AND the single long document split
    assert rr.POOL_STATS["context_split"] - splits_before >= 2
    for call in serwer.posts:
        for doc in call:
            assert doc in ("short", "other") or doc in _DOCS[1]
    urllib_ua = [ua for ua in serwer.user_agents if ua.startswith("Python-urllib")]
    if keepalive > 0:
        assert urllib_ua == [], "pooled run leaked onto the urllib transport"
        assert serwer.url in rr._POOL
        assert rr.POOL_STATS["opened"] - opened_before == 1  # one kept-alive connection
    else:
        assert len(urllib_ua) == len(serwer.posts)
        assert rr._POOL == {}


@pytest.mark.parametrize("keepalive", [4.0, 0.0], ids=["pooled", "urllib"])
def test_unrelated_400_is_not_split(serwer, monkeypatch, keepalive: float) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", keepalive)
    serwer.status_for_all = 400
    serwer.body_for_all = b'{"error":"unknown model"}'
    with pytest.raises(Exception) as exc:
        rr.HttpReranker(endpoint=serwer.url, model_name="m")._raw_scores("q", ["first", "second"])
    assert "400" in str(exc.value)
    assert len(serwer.posts) == 1


def test_pooled_401_degrades_loudly_with_status(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    serwer.status_for_all = 401
    serwer.body_for_all = b'{"error":"invalid key"}'
    reasons: list[str] = []
    from surreal_memory.engine.activation import ActivationResult

    acts = {
        "n1": ActivationResult(
            neuron_id="n1", activation_level=0.9, hop_distance=0, path=["n1"], source_anchor="n1"
        ),
        "n2": ActivationResult(
            neuron_id="n2", activation_level=0.5, hop_distance=0, path=["n2"], source_anchor="n2"
        ),
    }
    out = rr.rerank_activations(
        "q", acts, {"n1": "alpha", "n2": "beta"}, endpoint=serwer.url, on_degraded=reasons.append
    )
    assert out == acts  # unchanged SA order, not a silent success
    assert len(reasons) == 1 and "401" in reasons[0]
    assert len(serwer.posts) == 2  # two attempts, no split
