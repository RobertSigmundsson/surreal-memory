"""Hook UserPromptSubmit: licznik wyjątków recallu (REK-4) i zagregowany licznik „za-krótki" (REK-7).

REK-4: wyjątek recallu w `main()` dotąd szedł WYŁĄCZNIE na stderr (nieprzechowywany) — klasa błędów bez licznika,
więc bramka K10b „0 nowych wyjątków" byłaby niemierzalna. Teraz: ten sam trwały plik co timeout / sync_error /
identity_error (`prompt_recall_slad_bledy.jsonl`), status `wyjatek`, powód = NAZWA klasy (nigdy treść wyjątku ani prompta).
REK-7: bramka długości nie zapisywała nic (0 wpisów „za-krótki" nie było pomiarem); teraz licznik dobowy.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

_RECALL = "surreal_memory.hooks.user_prompt_submit.get_prompt_recall"


def _cfg(**kw: object):
    from surreal_memory.unified_config import PromptRecallConfig

    return PromptRecallConfig.from_dict({"enabled": True, **kw})


def _wiersze(katalog: Path) -> list[dict[str, str]]:
    plik = katalog / "prompt_recall_slad_bledy.jsonl"
    return [json.loads(x) for x in plik.read_text().splitlines()] if plik.exists() else []


class _BladTestowyError(RuntimeError):
    pass


def test_wyjatek_recallu_zostawia_trwaly_wpis_z_nazwa_klasy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from surreal_memory.hooks import user_prompt_submit as ups

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    tajne = "TAJNA-TRESC-PROMPTA-i-komunikatu"

    async def _pada(_hi: dict) -> str:
        raise _BladTestowyError(tajne)

    monkeypatch.setattr(ups, "read_hook_input", lambda: {"prompt": tajne, "session_id": "s-wyj"})
    with patch(_RECALL, _pada), patch("surreal_memory.unified_config.get_config") as gc:
        gc.return_value.prompt_recall = _cfg()
        with pytest.raises(SystemExit) as exit_info:
            ups.main()
    assert exit_info.value.code == 0  # prompt nigdy nie zablokowany
    wiersze = _wiersze(tmp_path)
    assert len(wiersze) == 1 and wiersze[0]["sesja"] == "s-wyj"
    assert (
        wiersze[0]["blad"] == "SMEM-SLAD-BLAD tor=cli status=wyjatek powod=recall-_BladTestowyError"
    )
    zapisane = (tmp_path / "prompt_recall_slad_bledy.jsonl").read_text()
    assert (
        tajne not in zapisane and tajne not in capsys.readouterr().out
    )  # ani prompt, ani komunikat wyjątku


def test_timeout_i_wyjatek_ida_do_tego_samego_pliku_i_sa_rozroznialne(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from surreal_memory.hooks import user_prompt_submit as ups

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))

    async def _wolno(_hi: dict) -> str:
        await asyncio.sleep(5)
        return ""

    with patch(_RECALL, _wolno):
        asyncio.run(ups._recall_within_timeout({"session_id": "s1"}, 0.05))
    ups._record_trace_error(ups._linia_wyjatku(ValueError("x")), "s2")
    statusy = sorted(w["blad"].split("status=")[1].split()[0] for w in _wiersze(tmp_path))
    assert statusy == ["timeout", "wyjatek"]


def test_brak_wyjatku_brak_wpisu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from surreal_memory.hooks import user_prompt_submit as ups

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))

    async def _ok(_hi: dict) -> str:
        return ""

    monkeypatch.setattr(ups, "read_hook_input", lambda: {"prompt": "x" * 300})
    with patch(_RECALL, _ok), patch("surreal_memory.unified_config.get_config") as gc:
        gc.return_value.prompt_recall = _cfg()
        with pytest.raises(SystemExit):
            ups.main()
    assert _wiersze(tmp_path) == []


@pytest.mark.asyncio
async def test_krotki_prompt_zwieksza_dobowy_licznik_bez_zapytania(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surreal_memory.hooks import liczniki
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    storage = AsyncMock()
    with (
        patch("surreal_memory.unified_config.get_config") as gc,
        patch(
            "surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)
        ) as gss,
    ):
        gc.return_value.prompt_recall = _cfg(min_prompt_chars=40)
        assert await get_prompt_recall({"prompt": "ok, dalej"}) == ""
        assert await get_prompt_recall({"prompt": "tak"}) == ""
        assert gss.await_count == 0  # zero połączeń z bazą
    dane = json.loads((tmp_path / liczniki.LICZNIKI_PLIK).read_text())
    assert sum(v["za_krotki"] for v in dane.values()) == 2


@pytest.mark.asyncio
async def test_prompt_wystarczajaco_dlugi_nie_zwieksza_licznika(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from surreal_memory.hooks import liczniki
    from surreal_memory.hooks.user_prompt_submit import get_prompt_recall

    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    storage = AsyncMock()
    storage.brain_id = "b"
    storage.get_brain = AsyncMock(return_value=None)  # brak mózgu ⇒ "" po bramce długości
    with (
        patch("surreal_memory.unified_config.get_config") as gc,
        patch("surreal_memory.unified_config.get_shared_storage", AsyncMock(return_value=storage)),
    ):
        gc.return_value.prompt_recall = _cfg(min_prompt_chars=40)
        gc.return_value.current_brain = "b"
        assert await get_prompt_recall({"prompt": "x" * 100}) == ""
    assert not (tmp_path / liczniki.LICZNIKI_PLIK).exists()
