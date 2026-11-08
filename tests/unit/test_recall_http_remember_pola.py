"""recall-http ``POST /v1/remember``: pola ``intencja`` / ``idem`` / ``autor``, ``powtorzenie`` i dziennik SHADOW.

Program hermes-zapis-przez-silnik (U1, KZ.5 / KZ.8). Aplikacja biegnie przez httpx ASGITransport na PRAWDZIWYM
``InMemoryStorage`` (podmienione tylko ``get_shared_storage`` i konfiguracja); jedyne atrapy to ``log_gate_decision`` /
``check_write_gate`` (zliczanie wywołań) oraz ``_query`` udające SurrealDB w teście ścieżki wyszukiwania po ``idem``.
Klucze to stałe testowe, nie prawdziwe sekrety.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.engine import remember_api
from surreal_memory.engine.gate_telemetry import log_gate_decision
from surreal_memory.engine.quality_scorer import QualityResult
from surreal_memory.recall_http import create_app
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import TraceConfig, UnifiedConfig

KEY = "r" * 40
WKEY = "w" * 40
WRITE = {"Authorization": f"Bearer {WKEY}"}
TEXT = "ZAPIS-SILNIK 7b3e weryfikacja nowych pol intencja idem autor w shimie"
BODY = {
    "content": TEXT,
    "type": "fact",
    "tags": ["zapis-silnik"],
    "agent_id": "agent:pod_a",
    "tor": "http:hermes-pod",
    "session_id": "sesja-1",
}
INTENCJE = (
    "jawny",
    "tura",
    "przed_kompresja",
    "delegacja",
    "lustro",
    "wynik_zadania",
    "domkniecie",
)


async def _astorage() -> InMemoryStorage:
    s = InMemoryStorage()
    brain = Brain.create(name="t", config=BrainConfig(activation_threshold=0.1, max_spread_hops=4))
    await s.save_brain(brain)
    s.set_brain(brain.id)
    return s


def _counts(s: InMemoryStorage) -> tuple[int, int, int, int]:
    b = s.brain_id or ""
    return len(s._neurons[b]), len(s._fibers[b]), len(s._synapses[b]), len(s._typed_memories[b])


async def _post(
    app: Any, storage: InMemoryStorage, body: Any, tmp: Path | None = None
) -> httpx.Response:
    async def _get() -> Any:
        return storage

    cfg = UnifiedConfig(
        data_dir=(tmp or Path("/nonexistent")) / ".sm",
        current_brain="t",
        trace=TraceConfig(enabled=True),
    )
    with (
        patch("surreal_memory.unified_config.get_shared_storage", _get),
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            return await c.post("/v1/remember", json=body, headers=WRITE)


async def _post_n(app: Any, storage: InMemoryStorage, body: Any, n: int) -> list[httpx.Response]:
    """n równoległych żądań w jednym kontekście patch (zagnieżdżone patche z gather wyciekałyby podmiany)."""

    async def _get() -> Any:
        return storage

    cfg = UnifiedConfig(
        data_dir=Path("/nonexistent") / ".sm", current_brain="t", trace=TraceConfig(enabled=True)
    )
    with (
        patch("surreal_memory.unified_config.get_shared_storage", _get),
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://t"
        ) as c:
            return list(
                await asyncio.gather(
                    *[c.post("/v1/remember", json=body, headers=WRITE) for _ in range(n)]
                )
            )


def _app(**kw: Any) -> Any:
    return create_app(key=KEY, remember_key=WKEY, **kw)


async def _fiber(s: InMemoryStorage, fiber_id: str) -> Any:
    f = await s.get_fiber(fiber_id)
    assert f is not None
    return f


# ── zgodność wsteczna: stary klient bez nowych pól ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stary_klient_bez_nowych_pol_dostaje_200_bez_powtorzenia_i_intencje_jawny() -> None:
    s = await _astorage()
    r = await _post(_app(), s, BODY)
    assert r.status_code == 200, r.text
    out = r.json()
    assert "powtorzenie" not in out
    bazowe = {
        "message",
        "fiber_id",
        "memory_type",
        "priority",
        "neurons_created",
        "neurons_linked",
        "synapses_created",
        "anchor_neuron_id",
        "content_sha256",
        "oczyszczone",
        "agent_id",
        "tor",
        "elapsed_ms",
    }
    assert bazowe <= set(out) and set(out) - bazowe <= {"expires_in_days"}
    sb = (await _fiber(s, out["fiber_id"])).metadata["stored_by"]
    assert sb["intencja"] == "jawny"
    assert "autor" not in sb and "idem" not in sb
    assert (sb["agent_id"], sb["tor"], sb["kanal"], sb["session_id"]) == (
        "agent:pod_a",
        "http:hermes-pod",
        "recall-http",
        "sesja-1",
    )


# ── walidacja nowych pól (extra="forbid" zostaje) ────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("patch_body", "pole"),
    [
        ({"intencja": "nieznana"}, "intencja"),
        ({"intencja": "TURA"}, "intencja"),
        ({"intencja": ""}, "intencja"),
        ({"idem": "x" * 65}, "idem"),
        ({"idem": ""}, "idem"),
        ({"idem": "z spacja"}, "idem"),
        ({"idem": "slash/nie"}, "idem"),
        ({"idem": "ż-polskie"}, "idem"),
        ({"autor": "a" * 121}, "autor"),
        ({"autor": ""}, "autor"),
        ({"autor": "z spacja"}, "autor"),
        ({"nieznane_pole": 1}, "nieznane_pole"),
    ],
)
async def test_nowe_pola_sa_walidowane_nie_ignorowane(
    patch_body: dict[str, Any], pole: str
) -> None:
    s = await _astorage()
    before = _counts(s)
    r = await _post(_app(), s, {**BODY, **patch_body})
    assert r.status_code == 422, r.text
    assert pole in r.text
    # Odmowa ma wynikać z NARUSZENIA ograniczenia znanego pola, nie z „pole nieznane” (tak odmawia kod sprzed
    # zmiany — bez tego asercja przechodziłaby przed zmianą z niewłaściwego powodu). Pole spoza kontraktu
    # (`nieznane_pole`) nadal ma być odrzucone właśnie jako nieznane (extra="forbid").
    jako_nieznane = "Extra inputs are not permitted" in r.text
    assert jako_nieznane is (pole == "nieznane_pole"), r.text
    assert "ZAPIS-SILNIK" not in r.text
    assert _counts(s) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("intencja", INTENCJE)
async def test_kazda_intencja_z_listy_trafia_do_stored_by(intencja: str) -> None:
    s = await _astorage()
    r = await _post(
        _app(), s, {**BODY, "intencja": intencja, "autor": "Robert", "idem": f"t.{intencja}:1"}
    )
    assert r.status_code == 200, r.text
    sb = (await _fiber(s, r.json()["fiber_id"])).metadata["stored_by"]
    assert (sb["intencja"], sb["autor"], sb["idem"]) == (intencja, "Robert", f"t.{intencja}:1")


@pytest.mark.asyncio
async def test_granice_poprawne_64_znaki_idem_i_120_autor_sa_przyjete() -> None:
    s = await _astorage()
    r = await _post(_app(), s, {**BODY, "idem": "I" * 64, "autor": "bot:" + "a" * 116})
    assert r.status_code == 200, r.text


# ── idempotencja ─────────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_powtorzenie_z_tym_samym_idem_daje_200_powtorzenie_ten_sam_fiber_bez_drugiego_zapisu() -> (
    None
):
    s, app = await _astorage(), _app()
    body = {**BODY, "idem": "tura-0001"}
    r1 = await _post(app, s, body)
    assert r1.status_code == 200 and "powtorzenie" not in r1.json()
    po_pierwszym = _counts(s)
    with patch.object(remember_api, "encode_and_store", wraps=remember_api.encode_and_store) as spy:
        r2 = await _post(app, s, body)
    assert r2.status_code == 200, r2.text
    o1, o2 = r1.json(), r2.json()
    assert o2["powtorzenie"] is True
    assert o2["fiber_id"] == o1["fiber_id"] and o2["anchor_neuron_id"] == o1["anchor_neuron_id"]
    assert (o2["neurons_created"], o2["neurons_linked"], o2["synapses_created"]) == (0, 0, 0)
    assert (o2["agent_id"], o2["tor"], o2["content_sha256"]) == (
        o1["agent_id"],
        o1["tor"],
        o1["content_sha256"],
    )
    spy.assert_not_called()
    assert _counts(s) == po_pierwszym
    assert (
        app.state.counters["remember_ok"] == 1 and app.state.counters["remember_powtorzenie"] == 1
    )
    liczba_z_idem = sum(
        1
        for f in s._fibers[s.brain_id or ""].values()
        if f.metadata.get("stored_by", {}).get("idem") == "tura-0001"
    )
    assert liczba_z_idem == 1


@pytest.mark.asyncio
async def test_bez_idem_dwa_identyczne_zapisy_daja_dwa_fibry_zachowanie_dzisiejsze() -> None:
    s, app = await _astorage(), _app()
    r1 = await _post(app, s, BODY)
    r2 = await _post(app, s, BODY)
    assert r1.status_code == r2.status_code == 200
    assert r1.json()["fiber_id"] != r2.json()["fiber_id"]
    assert "powtorzenie" not in r1.json() and "powtorzenie" not in r2.json()
    assert _counts(s)[1] == 2


@pytest.mark.asyncio
async def test_rozne_idem_to_rozne_zapisy() -> None:
    s, app = await _astorage(), _app()
    r1 = await _post(app, s, {**BODY, "idem": "a-1"})
    r2 = await _post(app, s, {**BODY, "idem": "a-2"})
    assert r1.json()["fiber_id"] != r2.json()["fiber_id"] and "powtorzenie" not in r2.json()


@pytest.mark.asyncio
async def test_rownolegle_zadania_z_tym_samym_idem_daja_jeden_fiber() -> None:
    s, app = await _astorage(), _app()
    body = {**BODY, "idem": "rownolegle-1"}
    r = await _post_n(app, s, body, 4)
    assert all(x.status_code == 200 for x in r), [x.text for x in r]
    assert len({x.json()["fiber_id"] for x in r}) == 1
    assert sum(1 for x in r if x.json().get("powtorzenie")) == 3
    assert _counts(s)[1] == 1


@pytest.mark.asyncio
async def test_wyszukanie_po_idem_na_surrealdb_to_zapytanie_po_stored_by_idem_i_zwraca_publiczne_id() -> (
    None
):
    from surreal_memory.recall_http import _fiber_po_idem

    storage = MagicMock()
    storage.brain_id = "uruboros"
    storage._query = AsyncMock(
        return_value=[{"id": "fiber:ab12_cd34", "anchor_neuron_id": "kotwica-1"}]
    )
    wynik = await _fiber_po_idem(storage, "tura-9")
    assert wynik == ("ab12-cd34", "kotwica-1")
    sql = storage._query.await_args.args[0]
    assert (
        "metadata.stored_by.idem = $idem" in sql
        and "brain_id = $brain_id" in sql
        and "LIMIT 1" in sql
    )
    assert storage._query.await_args.kwargs == {"brain_id": "uruboros", "idem": "tura-9"}
    storage._query = AsyncMock(return_value=[])
    assert await _fiber_po_idem(storage, "brak") is None


# ── dziennik SHADOW bramki jakości ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_shadow_gate_decision_jeden_wiersz_na_zapis_z_source_recall_http() -> None:
    s, app = await _astorage(), _app()
    teksty = [TEXT + " pierwszy", TEXT + " drugi inny", TEXT + " trzeci zupelnie inny"]
    with patch(
        "surreal_memory.engine.gate_telemetry.log_gate_decision", new_callable=AsyncMock
    ) as log:
        for t, intencja in zip(teksty, ("tura", "jawny", "wynik_zadania"), strict=True):
            r = await _post(app, s, {**BODY, "content": t, "intencja": intencja})
            assert r.status_code == 200, r.text
    assert log.await_count == 3
    for call, intencja in zip(log.await_args_list, ("tura", "jawny", "wynik_zadania"), strict=True):
        assert call.kwargs["source"] == "recall-http" and call.kwargs["mode"] == "shadow"
        assert call.kwargs["intent"] == intencja and call.kwargs["agent_id"] == "agent:pod_a"
        assert isinstance(call.kwargs["accepted"], bool) and call.kwargs["content"].startswith(
            "ZAPIS-SILNIK"
        )


@pytest.mark.asyncio
async def test_shadow_nie_blokuje_zapisu_gdy_bramka_by_odrzucila() -> None:
    s, app = await _astorage(), _app()
    odrzucone = QualityResult(
        score=0, quality="low", hints=(), rejected=True, rejection_reason="wall_of_text"
    )
    with (
        patch("surreal_memory.engine.quality_scorer.check_write_gate", return_value=odrzucone),
        patch(
            "surreal_memory.engine.gate_telemetry.log_gate_decision", new_callable=AsyncMock
        ) as log,
    ):
        r = await _post(app, s, BODY)
    assert r.status_code == 200 and _counts(s)[1] == 1
    assert (
        log.await_args.kwargs["accepted"] is False
        and log.await_args.kwargs["reason"] == "wall_of_text"
    )
    assert log.await_args.kwargs["mode"] == "shadow"


@pytest.mark.asyncio
async def test_awaria_oceny_bramki_nie_psuje_zapisu() -> None:
    s, app = await _astorage(), _app()
    with patch(
        "surreal_memory.engine.quality_scorer.check_write_gate",
        side_effect=RuntimeError("bramka"),
    ) as bramka:
        r = await _post(app, s, BODY)
    assert bramka.call_count == 1, (
        "ocena bramki musi zostać PODJĘTA (SHADOW), a jej awaria nie blokuje"
    )
    assert r.status_code == 200 and _counts(s)[1] == 1


@pytest.mark.asyncio
async def test_awaria_zapisu_dziennika_nie_psuje_zapisu() -> None:
    s, app = await _astorage(), _app()
    with patch(
        "surreal_memory.engine.gate_telemetry.log_gate_decision",
        new_callable=AsyncMock,
        side_effect=RuntimeError("log"),
    ) as log:
        r = await _post(app, s, BODY)
    assert log.await_count == 1, (
        "wpis do dziennika musi zostać PODJĘTY, a jego awaria nie blokuje zapisu"
    )
    assert r.status_code == 200 and _counts(s)[1] == 1


@pytest.mark.asyncio
async def test_powtorzenie_i_odmowy_nie_pisza_wiersza_gate_decision() -> None:
    s, app = await _astorage(), _app()
    with patch(
        "surreal_memory.engine.gate_telemetry.log_gate_decision", new_callable=AsyncMock
    ) as log:
        assert (await _post(app, s, {**BODY, "idem": "g-1"})).status_code == 200
        assert log.await_count == 1
        assert (await _post(app, s, {**BODY, "idem": "g-1"})).json()["powtorzenie"] is True
        assert (await _post(app, s, {**BODY, "idem": "z spacja"})).status_code == 422
        assert (
            await _post(app, s, {**BODY, "content": "moje haslo password=Sup3rTajne!2026 do bazy"})
        ).status_code == 422
    assert log.await_count == 1


# ── telemetria: parametr source z wartością domyślną ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_log_gate_decision_domyslnie_source_engine_a_na_zadanie_recall_http() -> None:
    conn = MagicMock()
    conn.insert = AsyncMock()
    storage = MagicMock()
    storage._ensure_conn.return_value = conn
    wspolne: dict[str, Any] = {
        "intent": "manual",
        "accepted": True,
        "reason": "accept",
        "score": 7,
        "mode": "shadow",
        "content": "treść testowa",
    }
    await log_gate_decision(storage, **wspolne)
    await log_gate_decision(storage, source="recall-http", **wspolne)
    wiersze = [c.args[1] for c in conn.insert.await_args_list]
    assert [w["source"] for w in wiersze] == ["engine", "recall-http"]
    assert all(w["intent"] == "manual" and w["mode"] == "shadow" for w in wiersze)


# ── kontrakt OpenAPI: tylko dopisane pola ────────────────────────────────────────────────────────


def test_openapi_remember_in_ma_dokladnie_trzy_nowe_pola_opcjonalne_a_out_opcjonalne_powtorzenie() -> (
    None
):
    schema = _app().openapi()
    wej = schema["components"]["schemas"]["RememberIn"]
    assert set(wej["properties"]) == {
        "content",
        "type",
        "tags",
        "priority",
        "agent_id",
        "tor",
        "session_id",
        "intencja",
        "idem",
        "autor",
    }
    assert set(wej["required"]) == {"content", "type", "agent_id", "tor"}
    assert wej["additionalProperties"] is False
    wyj = schema["components"]["schemas"]["RememberOut"]
    assert "powtorzenie" in wyj["properties"] and "powtorzenie" not in wyj["required"]
