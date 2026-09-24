"""Per-process Jev key override (program jev-uzycie-wdrozenie, R2).

The only HTTP seam is `jev_gate._blocking_post` (no real network, as in test_refusal_observe.py).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from surreal_memory.engine import jev_gate
from surreal_memory.engine.jev_gate import (
    ENV_NADPISANIE_KLUCZA,
    rozwiaz_klucz,
    sekrety_do_redakcji,
    zapytaj_jev,
)

if TYPE_CHECKING:
    import pytest

_OK = json.dumps(
    {
        "model": "jev-1.13.0",
        "answers": {
            "odpowiada": {"type": "noul", "noul": 0.9},
            "sensowne": {"type": "noul", "noul": 0.9},
            "ta_domena": {"type": "noul", "noul": 0.9},
            "jakosc": {"type": "score", "score": 3, "confidence": 0.8},
        },
        "usage": {"input_tokens": 10},
    }
).encode()


def _capture(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    def _fake(
        url: str, body: bytes, headers: dict[str, str], timeout_s: float
    ) -> tuple[int, bytes]:
        seen.append({"auth": headers["Authorization"], "body": body.decode()})
        return 200, _OK

    monkeypatch.setattr(jev_gate, "_blocking_post", _fake)
    return seen


async def _call(klucz: Any, sekrety: list[str] | None = None, memories: str = "m") -> Any:
    return await zapytaj_jev(
        query="q",
        memories=memories,
        gateway_url="http://brama.invalid/typesafe/v1/systemone",
        api_key=klucz.wartosc,
        model="jev-1.13.0",
        timeout_ms=1000,
        sekrety=sekrety or [],
        klucz_env=klucz.env_name if klucz.zrodlo == "nadpisanie" else None,
    )


def test_no_override_uses_config_name() -> None:
    k = rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", "", {"LITELLM_KEY_ROJ_JEV": "wartosc-roju"})
    assert (k.env_name, k.wartosc, k.zrodlo) == ("LITELLM_KEY_ROJ_JEV", "wartosc-roju", "config")


def test_blank_override_is_no_override() -> None:
    for pusty in ("", "   "):
        env = {"LITELLM_KEY_ROJ_JEV": "wartosc-roju", ENV_NADPISANIE_KLUCZA: pusty}
        assert rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", "", env).zrodlo == "config"


async def test_override_sends_the_channel_key_not_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    env = {
        "LITELLM_KEY_ROJ_JEV": "wartosc-roju",
        "LITELLM_KEY_NEMO_JEV_HOOK": "wartosc-hooka",
        ENV_NADPISANIE_KLUCZA: "LITELLM_KEY_NEMO_JEV_HOOK",
    }
    k = rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", "", env)
    assert (k.env_name, k.zrodlo) == ("LITELLM_KEY_NEMO_JEV_HOOK", "nadpisanie")
    odp = await _call(k)
    assert odp.status == "OK"
    assert seen[0]["auth"] == "Bearer wartosc-hooka"


async def test_override_to_missing_name_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch)
    env = {"LITELLM_KEY_ROJ_JEV": "wartosc-roju", ENV_NADPISANIE_KLUCZA: "LITELLM_KEY_NIE_MA"}
    k = rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", "", env)
    assert k.wartosc is None
    odp = await _call(k)
    assert odp.status == "JEV_NIEDOSTEPNY"
    assert odp.powod == "brak klucza: LITELLM_KEY_NIE_MA"
    assert odp.odpowiada is None and seen == []


def test_override_ignores_the_configured_key_file(tmp_path: Path) -> None:
    plik = tmp_path / "klucz"
    plik.write_text("wartosc-z-pliku\n")
    env = {ENV_NADPISANIE_KLUCZA: "LITELLM_KEY_NIE_MA"}
    assert rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", str(plik), env).wartosc is None
    # without the override the file fallback still works as before
    assert rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", str(plik), {}).wartosc == "wartosc-z-pliku"


async def test_override_that_looks_like_a_secret_is_rejected_and_not_echoed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen = _capture(monkeypatch)
    wklejona = "sk-to-jest-wartosc-klucza-a-nie-nazwa"
    k = rozwiaz_klucz(
        "LITELLM_KEY_ROJ_JEV",
        "",
        {ENV_NADPISANIE_KLUCZA: wklejona, "LITELLM_KEY_ROJ_JEV": "x" * 20},
    )
    assert (k.env_name, k.wartosc, k.zrodlo) == (None, None, "nadpisanie-nieprawidlowe")
    with caplog.at_level(logging.DEBUG):
        odp = await zapytaj_jev(
            query="q",
            memories="m",
            gateway_url="http://brama.invalid",
            api_key=k.wartosc,
            model="jev-1.13.0",
            timeout_ms=1000,
            sekrety=[],
            klucz_env="SURREAL_MEMORY_JEV_API_KEY_ENV (nieprawidłowa nazwa)",
        )
    assert odp.status == "JEV_NIEDOSTEPNY" and seen == []
    assert wklejona not in (odp.powod or "") and wklejona not in caplog.text


async def test_redaction_covers_active_key_and_every_litellm_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch)
    env = {
        "LITELLM_KEY_NEMO_JEV_HOOK": "aktywny-klucz-12345",
        "LITELLM_KEY_ROJ_MAKER": "inny-klucz-roju-6789",
        "FOO_BAR": "dluga-wartosc-spoza-klasy",
    }
    sek = sekrety_do_redakcji((), "aktywny-klucz-12345", env)
    assert "aktywny-klucz-12345" in sek and "inny-klucz-roju-6789" in sek
    assert "dluga-wartosc-spoza-klasy" not in sek
    odp = await zapytaj_jev(
        query="q",
        memories="x aktywny-klucz-12345 y inny-klucz-roju-6789 z dluga-wartosc-spoza-klasy",
        gateway_url="http://brama.invalid",
        api_key="aktywny-klucz-12345",
        model="jev-1.13.0",
        timeout_ms=1000,
        sekrety=sek,
    )
    assert odp.zredagowano == 2
    body = seen[0]["body"]
    assert "aktywny-klucz-12345" not in body and "inny-klucz-roju-6789" not in body
    assert "dluga-wartosc-spoza-klasy" in body


def test_override_never_reaches_the_saved_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W13: long-lived processes call save(); the override must not be pinned into config.toml."""
    from surreal_memory.unified_config import UnifiedConfig

    monkeypatch.setenv(ENV_NADPISANIE_KLUCZA, "LITELLM_KEY_NEMO_JEV_MCP")
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    cfg = UnifiedConfig.load(tmp_path / "config.toml")
    assert cfg.jev.api_key_env == "LITELLM_KEY_ROJ_JEV"
    cfg.save()
    tekst = (tmp_path / "config.toml").read_text()
    assert "LITELLM_KEY_NEMO_JEV_MCP" not in tekst
    assert 'api_key_env = "LITELLM_KEY_ROJ_JEV"' in tekst


async def test_missing_key_without_override_keeps_the_historical_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _capture(monkeypatch)
    k = rozwiaz_klucz("LITELLM_KEY_ROJ_JEV", "", {})
    odp = await _call(k)
    assert odp.powod == "brak klucza"
