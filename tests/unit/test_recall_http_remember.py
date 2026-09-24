"""recall-http: disjoint read/write scopes, POST /v1/remember and POST /v1/recall-cli.

The app runs over httpx ASGITransport against a REAL InMemoryStorage (only get_shared_storage and
the unified config are patched). Keys are test constants, never real secrets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from typer.testing import CliRunner

from surreal_memory.cli.main import app as cli_app
from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine import remember_api
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.engine.retrieval import ReflexPipeline
from surreal_memory.recall_http import create_app
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import TraceConfig, UnifiedConfig

KEY = "r" * 40
WKEY = "w" * 40
READ = {"Authorization": f"Bearer {KEY}"}
WRITE = {"Authorization": f"Bearer {WKEY}"}
TEXT = "K4-PARYTET 5d1c weryfikacja zapisu shimu dla modulu rozliczen"
BODY = {
    "content": TEXT,
    "type": "fact",
    "tags": ["k4"],
    "priority": 6,
    "agent_id": "agent:pod_a",
    "tor": "http:claude-pod",
    "session_id": "sesja-1",
}
SECRET = "moje haslo password=Sup3rTajne!2026 do bazy"  # noqa: S105 — sensitive-content gate fixture
runner = CliRunner()


async def _astorage(texts: tuple[str, ...] = ()) -> InMemoryStorage:
    s = InMemoryStorage()
    cfg = BrainConfig(activation_threshold=0.1, max_spread_hops=4)
    brain = Brain.create(name="t", config=cfg)
    await s.save_brain(brain)
    s.set_brain(brain.id)
    enc = MemoryEncoder(s, cfg)
    for t in texts:
        await enc.encode(t)
    return s


def _storage(texts: tuple[str, ...] = ()) -> InMemoryStorage:
    return asyncio.run(_astorage(texts))


def _counts(s: InMemoryStorage) -> tuple[int, int, int, int]:
    b = s.brain_id or ""
    return len(s._neurons[b]), len(s._fibers[b]), len(s._synapses[b]), len(s._typed_memories[b])


def _ucfg(tmp_path: Path) -> UnifiedConfig:
    return UnifiedConfig(
        data_dir=tmp_path / ".sm", current_brain="t", trace=TraceConfig(enabled=True)
    )


async def _call(
    app: Any,
    storage: InMemoryStorage,
    path: str,
    body: Any = None,
    headers: dict[str, str] | None = None,
    raw: bytes | None = None,
    tmp_path: Path | None = None,
) -> httpx.Response:
    async def _get() -> Any:
        return storage

    with (
        patch("surreal_memory.unified_config.get_shared_storage", _get),
        patch(
            "surreal_memory.unified_config.get_config",
            return_value=_ucfg(tmp_path or Path("/nonexistent")),
        ),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            if path == "/health":
                return await c.get(path)
            if raw is not None:
                return await c.post(path, content=raw, headers=headers or {})
            return await c.post(path, json=body, headers=headers or {})


def _app(**kw: Any) -> Any:
    return create_app(key=KEY, remember_key=kw.pop("remember_key", WKEY), **kw)


# ── scopes and order ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong"}, {"Authorization": WKEY}, {"Authorization": "Basic x"}],
)
async def test_401_on_remember_without_a_valid_key(headers: dict[str, str]) -> None:
    s = await _astorage()
    before = _counts(s)
    r = await _call(_app(), s, "/v1/remember", BODY, headers)
    assert r.status_code == 401 and _counts(s) == before


@pytest.mark.asyncio
async def test_read_key_on_remember_is_403_scope() -> None:
    s, app = await _astorage(), _app()
    before = _counts(s)
    r = await _call(app, s, "/v1/remember", BODY, READ)
    assert (r.status_code, r.json()["powod"]) == (403, "zakres")
    assert _counts(s) == before and app.state.counters["403"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/v1/recall", "/v1/recall-cli"])
async def test_write_key_on_read_routes_is_403(path: str) -> None:
    body = {"query": "q", "agent_id": "a", "tor": "http:claude-pod"}
    r = await _call(_app(), await _astorage(), path, body, WRITE)
    assert (r.status_code, r.json()["powod"]) == (403, "zakres")


@pytest.mark.asyncio
async def test_writing_disabled_without_remember_key(tmp_path: Path) -> None:
    app, s = create_app(key=KEY), await _astorage()
    r = await _call(app, s, "/v1/remember", BODY, READ)
    assert (r.status_code, r.json()["powod"]) == (403, "zapis_wylaczony")
    h = await _call(app, s, "/health", tmp_path=tmp_path)
    assert h.json()["zapis"] == "wylaczony"
    assert (await _call(_app(), s, "/health", tmp_path=tmp_path)).json()["zapis"] == "wlaczony"


@pytest.mark.asyncio
async def test_check_order_401_then_403_then_422() -> None:
    bad = {"nonsense": 1}
    s = await _astorage()
    assert (await _call(_app(), s, "/v1/remember", bad, {})).status_code == 401
    assert (await _call(_app(), s, "/v1/remember", bad, READ)).status_code == 403
    assert (await _call(_app(), s, "/v1/remember", bad, WRITE)).status_code == 422


@pytest.mark.asyncio
async def test_411_413_only_after_auth() -> None:
    s = await _astorage()
    big = json.dumps({**BODY, "content": "x" * 200_000}).encode()
    r = await _call(
        _app(), s, "/v1/remember", headers={**WRITE, "content-type": "application/json"}, raw=big
    )
    assert r.status_code == 413
    r = await _call(
        _app(), s, "/v1/remember", headers={"content-type": "application/json"}, raw=big
    )
    assert r.status_code == 401


# ── validation and content policy ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patch_body",
    [
        {"content": None},
        {"content": ""},
        {"content": "x" * 8001},
        {"ephemeral": True},
        {"force": True},
        {"project": "p"},
        {"type": "boundary"},
        {"type": "instruction"},
        {"type": "preference"},
        {"type": "FACT"},
        {"type": None},
        {"tags": [f"t{i}" for i in range(51)]},
        {"tags": ["x" * 101]},
        {"tags": ["zly\x07tag"]},
        {"priority": 11},
        {"priority": -1},
        {"agent_id": "zly agent"},
        {"tor": "cli"},
        {"tor": "mcp"},
        {"tor": "http:"},
        {"session_id": "z spacja"},
    ],
)
async def test_422_never_echoes_content(patch_body: dict[str, Any]) -> None:
    body = {k: v for k, v in {**BODY, **patch_body}.items() if v is not None}
    s = await _astorage()
    before = _counts(s)
    r = await _call(_app(), s, "/v1/remember", body, WRITE)
    assert r.status_code == 422, r.text
    assert "K4-PARYTET" not in r.text and "zly agent" not in r.text
    assert _counts(s) == before


@pytest.mark.asyncio
async def test_sensitive_content_refused_without_the_match() -> None:
    s, app = await _astorage(), _app()
    before = _counts(s)
    r = await _call(app, s, "/v1/remember", {**BODY, "content": SECRET}, WRITE)
    assert r.status_code == 422
    assert r.json() == {"error": "sensitive_content", "typy": ["password"], "liczba": 1}
    assert "Sup3rTajne" not in r.text and _counts(s) == before
    assert app.state.counters["remember_odrzucone"] == 1 and app.state.counters["422"] == 1


@pytest.mark.asyncio
async def test_lone_surrogate_never_5xx() -> None:
    raw = b'{"content":"alice \\ud800 api","type":"fact","agent_id":"a","tor":"http:claude-pod"}'
    r = await _call(
        _app(),
        await _astorage(),
        "/v1/remember",
        headers={**WRITE, "content-type": "application/json"},
        raw=raw,
    )
    assert r.status_code in (200, 422)


# ── the write ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_200_writes_graph_with_attribution(caplog: pytest.LogCaptureFixture) -> None:
    s, app = await _astorage(), _app()
    before = _counts(s)
    caplog.set_level(logging.INFO, logger="surreal_memory.recall_http")
    r = await _call(app, s, "/v1/remember", BODY, WRITE)
    assert r.status_code == 200, r.text
    out = r.json()
    after = _counts(s)
    assert all(a > b for a, b in zip(after[:2], before[:2], strict=True)) and after[2] > before[2]
    assert out["content_sha256"] == hashlib.sha256(TEXT.encode()).hexdigest()
    assert (out["agent_id"], out["tor"], out["oczyszczone"]) == (
        "agent:pod_a",
        "http:claude-pod",
        False,
    )
    anchor = await s.get_neuron(out["anchor_neuron_id"])
    fiber = await s.get_fiber(out["fiber_id"])
    tm = await s.get_typed_memory(out["fiber_id"])
    assert anchor is not None and fiber is not None and tm is not None
    assert anchor.metadata["stored_by"]["agent_id"] == "agent:pod_a"
    assert fiber.metadata["stored_by"]["session_id"] == "sesja-1"
    assert (tm.source, tm.provenance.created_by) == ("http:claude-pod", "agent:pod_a")
    assert app.state.counters["remember_ok"] == 1
    line = next(r.message for r in caplog.records if r.message.startswith("remember "))
    assert f"tresc={out['content_sha256'][:8]}" in line and "agent=agent:pod_a" in line
    assert "K4-PARYTET" not in line and "k4" not in line.split("tagi=")[0]


@pytest.mark.asyncio
async def test_parity_with_cli_remember_on_identical_brains() -> None:
    shim_s, cli_s = await _astorage(), await _astorage()
    r = await _call(_app(), shim_s, "/v1/remember", BODY, WRITE)
    assert r.status_code == 200
    with (
        patch("surreal_memory.cli.commands.memory.get_config", MagicMock()),
        patch("surreal_memory.cli.commands.memory.get_storage", new=AsyncMock(return_value=cli_s)),
    ):
        res = await asyncio.to_thread(
            runner.invoke,
            cli_app,
            ["remember", TEXT, "--type", "fact", "--tag", "k4", "--priority", "6", "--json"],
        )
    assert res.exit_code == 0, res.output
    cli_out, shim_out = json.loads(res.stdout), r.json()
    assert _counts(shim_s) == _counts(cli_s)
    for k in ("neurons_created", "neurons_linked", "synapses_created", "memory_type", "priority"):
        assert shim_out[k] == cli_out[k], k
    shim_anchor = await shim_s.get_neuron(shim_out["anchor_neuron_id"])
    cli_fiber = await cli_s.get_fiber(cli_out["fiber_id"])
    assert cli_fiber is not None and shim_anchor is not None
    cli_anchor = await cli_s.get_neuron(cli_fiber.anchor_neuron_id)
    assert cli_anchor is not None and cli_anchor.content == shim_anchor.content


def test_parity_text_is_a_firewall_fixed_point() -> None:
    from surreal_memory.safety.input_firewall import sanitize_explicit_content

    assert sanitize_explicit_content(TEXT) == TEXT


@pytest.mark.asyncio
async def test_busy_write_lock_gives_503_and_writes_are_serialized() -> None:
    s, app = await _astorage(), _app(queue_timeout_s=0.05)
    gate = asyncio.Event()
    orig = remember_api.encode_and_store
    order: list[str] = []

    async def slow(*a: Any, **kw: Any) -> Any:
        order.append("start")
        await gate.wait()
        out = await orig(*a, **kw)
        order.append("end")
        return out

    async def _get() -> Any:
        return s

    with (
        patch.object(remember_api, "encode_and_store", slow),
        patch("surreal_memory.unified_config.get_shared_storage", _get),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            first = asyncio.create_task(c.post("/v1/remember", json=BODY, headers=WRITE))
            await asyncio.sleep(0.02)
            second = await c.post(
                "/v1/remember", json={**BODY, "content": TEXT + " drugi"}, headers=WRITE
            )
            gate.set()
            r1 = await first
    assert (r1.status_code, second.status_code) == (200, 503)
    assert order == ["start", "end"]


@pytest.mark.asyncio
async def test_storage_exception_is_500_and_counted() -> None:
    s, app = await _astorage(), _app()

    async def boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("db down")

    async def _get() -> Any:
        return s

    # Starlette re-raises after the 500 handler answered; the client must not re-raise it.
    with (
        patch.object(remember_api, "encode_and_store", boom),
        patch("surreal_memory.unified_config.get_shared_storage", _get),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t"
        ) as c:
            r = await c.post("/v1/remember", json=BODY, headers=WRITE)
    assert (r.status_code, r.json()) == (500, {"error": "internal error"})
    assert app.state.counters["remember_err"] == 1 and app.state.counters["5xx"] == 1


def test_create_app_rejects_short_or_equal_write_key() -> None:
    with pytest.raises(ValueError):
        create_app(key=KEY, remember_key="short")
    with pytest.raises(ValueError):
        create_app(key=KEY, remember_key=KEY)


@pytest.mark.parametrize(
    "env",
    [
        {"SMEM_RECALL_HTTP_KEY": KEY},
        {"SMEM_RECALL_HTTP_KEY": KEY, "SMEM_REMEMBER_HTTP_KEY": "short"},
        {"SMEM_RECALL_HTTP_KEY": KEY, "SMEM_REMEMBER_HTTP_KEY": KEY},
    ],
)
def test_cli_zapis_without_a_valid_write_key_exits_78(
    env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SMEM_REMEMBER_HTTP_KEY", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    res = runner.invoke(cli_app, ["recall-http", "--zapis"])
    assert res.exit_code == 78
    assert KEY not in res.output


def test_health_keeps_old_counters_and_adds_new(tmp_path: Path) -> None:
    got = asyncio.run(_call(_app(), _storage(), "/health", tmp_path=tmp_path)).json()
    old = {"ok", "401", "422", "503", "5xx", "odroczone_ok", "odroczone_err", "bariera_timeout"}
    new = {
        "403",
        "411",
        "413",
        "remember_ok",
        "remember_odrzucone",
        "remember_err",
        "recall_cli_ok",
    }
    assert old | new == set(got["liczniki"])


def test_no_background_work_and_no_daemon_app() -> None:
    import surreal_memory.engine.cli_recall_api as cra
    import surreal_memory.recall_http as rh

    for mod in (rh, remember_api, cra):
        src = Path(mod.__file__ or "").read_text(encoding="utf-8")
        for bad in ("create_task", "ensure_future", "Thread("):
            assert bad not in src, (mod.__name__, bad)
    code = (
        "import sys, surreal_memory.recall_http\n"
        "print('surreal_memory.server.app' in sys.modules, 'surreal_memory.cli.main' in sys.modules)"
    )
    out = subprocess.run(  # noqa: S603 -- fixed cmd, no shell, no untrusted input
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False False"


# ── /v1/recall-cli = smem recall semantics ────────────────────────────────────────────────────

RECALL_TEXTS = (
    "Met with Alice at the coffee shop to discuss API design",
    "Alice suggested adding rate limiting to the API",
    "Completed the authentication module for the API gateway",
)
QUERY = "What did Alice suggest for the API?"


@pytest.mark.asyncio
async def test_recall_cli_calls_the_cli_pipeline_and_traces_the_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s, app = await _astorage(RECALL_TEXTS), _app()
    seen: list[dict[str, Any]] = []
    orig = ReflexPipeline.query

    async def spy(self: ReflexPipeline, *a: Any, **kw: Any) -> Any:
        seen.append(kw)
        return await orig(self, *a, **kw)

    monkeypatch.setattr(ReflexPipeline, "query", spy)
    body = {
        "query": QUERY,
        "agent_id": "agent:pod_b",
        "tor": "http:claude-pod",
        "session_id": "s-b",
    }
    r = await _call(app, s, "/v1/recall-cli", body, READ, tmp_path=tmp_path)
    assert r.status_code == 200, r.text
    out = r.json()
    assert set(seen[0]) == {"query", "depth", "max_tokens", "reference_time"}
    assert out["fibers_matched"] and out["trace_status"] == "sync"
    traces = s._retrieval_traces[s.brain_id or ""]
    assert [(t.tor, t.agent_id, t.session_id) for t in traces] == [
        ("http:claude-pod", "agent:pod_b", "s-b")
    ]
    assert app.state.counters["recall_cli_ok"] == 1 and app.state.counters["ok"] == 0


@pytest.mark.asyncio
async def test_recall_cli_keys_match_host_cli_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("SMEM_AGENT_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID"):
        monkeypatch.delenv(name, raising=False)
    body = {"query": QUERY, "agent_id": "a", "tor": "http:claude-pod"}
    shim_out = (
        await _call(
            _app(), await _astorage(RECALL_TEXTS), "/v1/recall-cli", body, READ, tmp_path=tmp_path
        )
    ).json()
    cli_s = await _astorage(RECALL_TEXTS)
    with (
        patch("surreal_memory.cli.commands.memory.get_config", MagicMock()),
        patch("surreal_memory.cli.commands.memory.get_storage", new=AsyncMock(return_value=cli_s)),
        patch("surreal_memory.unified_config.get_config", return_value=_ucfg(tmp_path)),
    ):
        res = await asyncio.to_thread(runner.invoke, cli_app, ["recall", QUERY, "--json"])
    cli_out = json.loads(res.stdout)
    assert list(shim_out) == list(cli_out)
