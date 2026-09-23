"""HTTP reranker connection reuse (SURREAL_MEMORY_RERANK_KEEPALIVE_S).

A real local HTTP/1.1 server stands in for the reranker; it can drop idle connections the way
uvicorn does after its keep-alive. Same request, same response: scores are identical with and
without pooling.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from surreal_memory.engine import reranker as rr


class _State:
    def __init__(self) -> None:
        self.connections = 0
        self.requests = 0
        self.close_after_each = False


@pytest.fixture
def serwer():
    st = _State()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self) -> None:
            super().setup()
            st.connections += 1

        def _send(self, obj: Any) -> None:
            data = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            if st.close_after_each:
                # Close the TCP connection silently (no "Connection: close" header), the way an
                # idle keep-alive timeout does: the client only learns on its next request.
                self.close_connection = True

        def do_GET(self) -> None:
            st.requests += 1
            self._send({"status": "ok"})

        def do_POST(self) -> None:
            st.requests += 1
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            n = len(body["documents"])
            self._send(
                {"results": [{"index": i, "relevance_score": float(n - i)} for i in range(n)]}
            )

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


def _scores(url: str) -> list[float]:
    return rr.HttpReranker(endpoint=url, model_name="m")._raw_scores("q", ["a", "b", "c"])


def test_pool_off_by_default_is_todays_path(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 0.0)
    _scores(serwer.url)
    _scores(serwer.url)
    assert serwer.connections == 2 and rr._POOL == {}


def test_pool_reuses_one_connection_and_scores_are_identical(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 0.0)
    bez = _scores(serwer.url)
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    polaczenia_przed = serwer.connections
    z1 = _scores(serwer.url)
    z2 = _scores(serwer.url)
    assert bez == z1 == z2 == [3.0, 2.0, 1.0]
    assert serwer.connections - polaczenia_przed == 1


def test_warm_up_then_rerank_uses_the_warm_connection(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    assert rr.rozgrzej(serwer.url) is True
    _scores(serwer.url)
    assert serwer.connections == 1 and serwer.requests == 2


def test_server_closed_idle_connection_is_reopened_not_degraded(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    serwer.close_after_each = True
    reconnect_before = rr.POOL_STATS["reconnect"]
    assert _scores(serwer.url) == [3.0, 2.0, 1.0]
    assert _scores(serwer.url) == [3.0, 2.0, 1.0]  # pooled conn was closed by the server
    assert rr.POOL_STATS["reconnect"] - reconnect_before >= 1


def test_stale_pooled_connection_is_not_reused(serwer, monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    _scores(serwer.url)
    rr._POOL[serwer.url].used_at -= 10.0  # older than KEEPALIVE_S
    _scores(serwer.url)
    assert serwer.connections == 2


def test_warm_up_never_raises_and_skips_non_http(monkeypatch) -> None:
    monkeypatch.setattr(rr, "KEEPALIVE_S", 4.0)
    assert rr.rozgrzej("not-a-url") is False
    assert rr.rozgrzej("http://127.0.0.1:1") is False  # refused
    monkeypatch.setattr(rr, "KEEPALIVE_S", 0.0)
    assert rr.rozgrzej("http://127.0.0.1:1") is False
