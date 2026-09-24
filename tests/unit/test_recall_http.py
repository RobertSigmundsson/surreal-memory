"""recall_http shim: auth before routing, strict payload, engine-only recall with a forced trace.

The engine call (``recall_api.recall``) and storage are replaced — this file tests the HTTP
layer. The live path on a brain copy is in ``test_recall_http_live.py`` (opt-in).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from typer.testing import CliRunner

from surreal_memory.engine import recall_api
from surreal_memory.recall_http import create_app

KEY = "k" * 40
AUTH = {"Authorization": f"Bearer {KEY}"}
OPENAPI = Path(__file__).resolve().parents[2] / "docs" / "api" / "recall-http-openapi.json"
BODY = {"query": "gdzie mieszka emma", "agent_id": "agent:s_p_1", "tor": "http:hermes-pod"}


class _Storage:
    brain_id = "b1"

    async def get_brain(self, _bid: str) -> Any:
        return SimpleNamespace(id="b1")

    async def get_fiber(self, fid: str) -> Any:
        return SimpleNamespace(anchor_neuron_id=f"n-{fid}", summary="s", metadata={})

    async def get_neuron(self, nid: str) -> Any:
        return SimpleNamespace(content=f"content of {nid}", type=SimpleNamespace(value="fact"))

    async def get_typed_memory(self, _fid: str) -> Any:
        return None


def _outcome(path: str = "pipeline", **resp: Any) -> recall_api.RecallOutcome:
    response = {
        "answer": "Emma lives in Bergen.",
        "confidence": 0.7,
        "neurons_activated": 5,
        "fibers_matched": ["f1", "f2", "f3"],
        "trace_id": "t-1",
        **resp,
    }
    result = SimpleNamespace(
        metadata={"activation_levels": {"n-f1": 0.9, "n-f2": 0.5}},
        synthesis_method="single",
        latency_ms=12.0,
    )
    return recall_api.RecallOutcome(response, result, path, "sync", "b1")  # type: ignore[arg-type]


class _Engine:
    def __init__(self, outcome: recall_api.RecallOutcome | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.outcome = outcome or _outcome()

    async def __call__(self, storage: Any, args: dict[str, Any], **kw: Any) -> Any:
        self.calls.append({"args": args, **kw})
        return self.outcome


async def _post(
    engine: _Engine, body: Any, headers: dict[str, str] | None = None, **app_kw: Any
) -> httpx.Response:
    app = create_app(key=KEY, **app_kw)

    async def _storage() -> Any:
        return _Storage()

    with (
        patch.object(recall_api, "recall", engine),
        patch("surreal_memory.unified_config.get_shared_storage", _storage),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            return await c.post("/v1/recall", json=body, headers=headers or {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY}, {"Authorization": "Basic x"}],
)
async def test_401_without_valid_bearer_and_engine_not_called(headers: dict[str, str]) -> None:
    eng = _Engine()
    r = await _post(eng, BODY, headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"
    assert eng.calls == []


@pytest.mark.asyncio
async def test_401_precedes_payload_validation() -> None:
    eng = _Engine()
    r = await _post(eng, {"nonsense": 1})
    assert r.status_code == 401
    assert eng.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_body",
    [
        {"query": None},
        {"query": ""},
        {"limit": 21},
        {"limit": 0},
        {"depth": 4},
        {"extra": 1},
        {"tor": "mcp"},
        {"tor": "cli"},
        {"tor": "gateway"},
        {"tor": "http:"},
        {"agent_id": None},
        {"agent_id": "bad id"},
    ],
)
async def test_422_on_bad_payload(patch_body: dict[str, Any]) -> None:
    body = {**BODY, **patch_body}
    body = {k: v for k, v in body.items() if v is not None}
    eng = _Engine()
    r = await _post(eng, body, AUTH)
    assert r.status_code == 422
    assert eng.calls == []


@pytest.mark.asyncio
async def test_200_engine_only_forced_trace_and_body_shape() -> None:
    eng = _Engine()
    r = await _post(eng, {**BODY, "limit": 2, "session_id": "s-1"}, AUTH)
    assert r.status_code == 200
    call = eng.calls[0]
    assert call["args"]["trace"] is True
    assert call["args"]["query"] == BODY["query"]
    assert (call["tor"], call["agent_id"]) == ("http:hermes-pod", "agent:s_p_1")
    assert call["extras"] is None and call["hooks"] is None
    assert call["engine_session_id"] == "http:hermes-pod|s-1"
    b = r.json()
    assert [m["id"] for m in b["memories"]] == ["f1", "f2"]  # limit cuts, order = rank
    assert [m["rank"] for m in b["memories"]] == [1, 2]
    assert b["memories"][0] == {
        "id": "f1",
        "neuron_id": "n-f1",
        "type": "fact",
        "content": "content of n-f1",
        "score": 0.9,
        "rank": 1,
    }
    assert b["trace_id"] == "t-1" and b["tor"] == "http:hermes-pod" and b["sufficient"] is True
    assert b["score_kind"] == "anchor_activation"


@pytest.mark.asyncio
async def test_default_session_is_per_agent() -> None:
    eng = _Engine()
    await _post(eng, BODY, AUTH)
    assert eng.calls[0]["engine_session_id"] == "http:hermes-pod|agent:s_p_1"


@pytest.mark.asyncio
async def test_trace_config_mode_does_not_force() -> None:
    eng = _Engine()
    await _post(eng, BODY, AUTH, trace_mode="config")
    assert "trace" not in eng.calls[0]["args"]


@pytest.mark.asyncio
async def test_engine_error_path_maps_to_422_and_no_brain_to_503() -> None:
    r = await _post(_Engine(_outcome("error", error="Invalid depth")), BODY, AUTH)
    assert r.status_code == 422
    r = await _post(_Engine(_outcome("error", error="No brain configured")), BODY, AUTH)
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_insufficient_signal_is_not_sufficient() -> None:
    out = _outcome()
    out.result.synthesis_method = "insufficient_signal"
    r = await _post(_Engine(out), BODY, AUTH)
    assert r.status_code == 200 and r.json()["sufficient"] is False


@pytest.mark.asyncio
async def test_lone_surrogate_query_is_not_5xx() -> None:
    eng = _Engine()
    app = create_app(key=KEY)

    async def _storage() -> Any:
        return _Storage()

    raw = '{"query": "a\\ud800b", "agent_id": "x", "tor": "http:test"}'
    with (
        patch.object(recall_api, "recall", eng),
        patch("surreal_memory.unified_config.get_shared_storage", _storage),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r = await c.post(
                "/v1/recall", content=raw, headers={**AUTH, "content-type": "application/json"}
            )
    assert r.status_code in (200, 422)  # never 5xx
    if r.status_code == 200:
        assert eng.calls[0]["args"]["query"].encode("utf-8")
    else:
        assert "ud800" not in r.text and eng.calls == []


@pytest.mark.asyncio
async def test_422_does_not_echo_the_query() -> None:
    r = await _post(_Engine(), {**BODY, "limit": 99, "query": "sekretna-tresc-zapytania"}, AUTH)
    assert r.status_code == 422
    assert "sekretna-tresc-zapytania" not in r.text


@pytest.mark.asyncio
async def test_busy_returns_503() -> None:
    import asyncio

    class _Slow(_Engine):
        async def __call__(self, storage: Any, args: dict[str, Any], **kw: Any) -> Any:
            await asyncio.sleep(0.5)
            return await super().__call__(storage, args, **kw)

    eng = _Slow()
    app = create_app(key=KEY, max_concurrency=1, queue_timeout_s=0.05)

    async def _storage() -> Any:
        return _Storage()

    with (
        patch.object(recall_api, "recall", eng),
        patch("surreal_memory.unified_config.get_shared_storage", _storage),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            rs = await asyncio.gather(
                c.post("/v1/recall", json=BODY, headers=AUTH),
                c.post("/v1/recall", json=BODY, headers=AUTH),
            )
    assert sorted(r.status_code for r in rs) == [200, 503]
    assert app.state.counters["503"] == 1 and app.state.counters["ok"] == 1


@pytest.mark.asyncio
async def test_health_open_and_reports_storage() -> None:
    app = create_app(key=KEY)

    async def _ok() -> Any:
        return _Storage()

    async def _down() -> Any:
        raise ConnectionError("db down")

    for fake, code in ((_ok, 200), (_down, 503)):
        with patch("surreal_memory.unified_config.get_shared_storage", fake):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://t"
            ) as c:
                r = await c.get("/health")
        assert r.status_code == code


def test_short_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        create_app(key="short")


def test_openapi_contract_file_matches_app() -> None:
    got = create_app(key=KEY).openapi()
    assert json.loads(OPENAPI.read_text(encoding="utf-8")) == got


def test_no_docs_routes_exposed() -> None:
    paths = {getattr(r, "path", "") for r in create_app(key=KEY).routes}
    assert paths == {"/health", "/v1/recall"}


def test_import_does_not_load_daemon_app() -> None:
    code = (
        "import surreal_memory.recall_http, sys; print('surreal_memory.server.app' in sys.modules)"
    )
    out = subprocess.run(  # noqa: S603 -- fixed cmd, no shell, no untrusted input
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
    src = Path(sys.modules["surreal_memory.recall_http"].__file__ or "").read_text(encoding="utf-8")
    for word in ("_consolidation_loop", "_decay_loop", "create_task"):
        assert word not in src


def test_cli_registered_and_refuses_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.cli.main import app

    runner = CliRunner()
    assert runner.invoke(app, ["recall-http", "--help"]).exit_code == 0
    monkeypatch.delenv("SMEM_RECALL_HTTP_KEY", raising=False)
    assert runner.invoke(app, ["recall-http"]).exit_code == 78
    monkeypatch.setenv("SMEM_RECALL_HTTP_KEY", "short")
    assert runner.invoke(app, ["recall-http"]).exit_code == 78


def test_cli_logging_emits_info_line_once() -> None:
    import logging

    from surreal_memory.cli.commands.recall_http import configure_logging

    log = configure_logging()
    configure_logging()  # idempotent: no duplicate handler
    assert log.level == logging.INFO and log.propagate is False
    assert sum(1 for h in log.handlers if getattr(h, "_recall_http", False)) == 1


@pytest.mark.asyncio
async def test_odroczone_barrier_awaits_previous_side_effects_before_next_recall() -> None:
    import asyncio

    zdarzenia: list[str] = []

    async def _skutki(i: int) -> str:
        await asyncio.sleep(0.2)
        zdarzenia.append(f"skutki-{i}")
        return "sync"

    licznik = {"n": 0}

    async def _engine(storage: Any, args: dict[str, Any], **kw: Any) -> Any:
        licznik["n"] += 1
        i = licznik["n"]
        zdarzenia.append(f"recall-{i}")
        assert kw["skutki"] == "odroczone" and kw["dekoracje"] is False
        out = _outcome()
        return recall_api.RecallOutcome(
            out.response, out.result, "pipeline", "deferred", "b1", asyncio.create_task(_skutki(i))
        )

    app = create_app(key=KEY, skutki="odroczone", bariera_s=5.0)

    async def _storage() -> Any:
        return _Storage()

    with (
        patch.object(recall_api, "recall", _engine),
        patch("surreal_memory.unified_config.get_shared_storage", _storage),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            r1 = await c.post("/v1/recall", json=BODY, headers=AUTH)
            r2 = await c.post("/v1/recall", json=BODY, headers=AUTH)
            for t in list(app.state.odroczone):
                await t
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["trace_status"] == "deferred"
    assert zdarzenia == ["recall-1", "skutki-1", "recall-2", "skutki-2"]
    assert app.state.counters["odroczone_ok"] == 2 and app.state.counters["bariera_timeout"] == 0


@pytest.mark.asyncio
async def test_bariera_zero_does_not_wait_negative_control() -> None:
    import asyncio

    zdarzenia: list[str] = []

    async def _skutki() -> str:
        await asyncio.sleep(0.3)
        zdarzenia.append("skutki")
        return "sync"

    async def _engine(storage: Any, args: dict[str, Any], **kw: Any) -> Any:
        zdarzenia.append("recall")
        out = _outcome()
        return recall_api.RecallOutcome(
            out.response, out.result, "pipeline", "deferred", "b1", asyncio.create_task(_skutki())
        )

    app = create_app(key=KEY, skutki="odroczone", bariera_s=0.0)

    async def _storage() -> Any:
        return _Storage()

    with (
        patch.object(recall_api, "recall", _engine),
        patch("surreal_memory.unified_config.get_shared_storage", _storage),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            await c.post("/v1/recall", json=BODY, headers=AUTH)
            await c.post("/v1/recall", json=BODY, headers=AUTH)
            for t in list(app.state.odroczone):
                await t
    assert zdarzenia[:2] == ["recall", "recall"]  # without the barrier the 2nd recall overtakes
    assert app.state.counters["bariera_timeout"] >= 1
