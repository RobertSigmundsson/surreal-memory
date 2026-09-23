"""HTTP reranker body with SURREAL_MEMORY_RERANK_NO_CACHE (LiteLLM response cache bypass).

A real local HTTP server records the request body. Off (default): the body is today's
{model, query, documents} byte for byte. On: the same body plus {"cache": {"no-cache": true}}.
Scores are the same either way.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from surreal_memory.engine import reranker as rr


@pytest.fixture
def serwer():
    ciala: list[dict[str, Any]] = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ciala.append(body)
            n = len(body["documents"])
            data = json.dumps(
                {"results": [{"index": i, "relevance_score": float(n - i)} for i in range(n)]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a: Any) -> None:
            return None

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    rr._POOL.clear()
    yield f"http://127.0.0.1:{srv.server_address[1]}", ciala
    rr._POOL.clear()
    srv.shutdown()
    srv.server_close()


def test_no_cache_off_by_default_sends_todays_body(serwer, monkeypatch) -> None:
    url, ciala = serwer
    monkeypatch.setattr(rr, "KEEPALIVE_S", 0.0)
    monkeypatch.setattr(rr, "NO_CACHE", False)
    s = rr.HttpReranker(endpoint=url, model_name="m")._raw_scores("q", ["a", "b"])
    assert ciala == [{"model": "m", "query": "q", "documents": ["a", "b"]}]
    assert s == [2.0, 1.0]


def test_no_cache_on_adds_litellm_cache_bypass_and_keeps_scores(serwer, monkeypatch) -> None:
    url, ciala = serwer
    monkeypatch.setattr(rr, "KEEPALIVE_S", 0.0)
    monkeypatch.setattr(rr, "NO_CACHE", True)
    s = rr.HttpReranker(endpoint=url, model_name="m")._raw_scores("q", ["a", "b"])
    assert ciala == [
        {"model": "m", "query": "q", "documents": ["a", "b"], "cache": {"no-cache": True}}
    ]
    assert s == [2.0, 1.0]
