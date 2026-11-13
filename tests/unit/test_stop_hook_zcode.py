"""Hook Stop a transkrypt hooka ZCode (zcode-parytet U7-B, Z-5).

ZCode pisze na czas hooka `<tmpdir>/zcode-hook-*/transcript.jsonl` (jedna linia `{"message": {"role": "assistant", "content":
[{"type": "text", "text": …}]}}`) i usuwa katalog po hooku (`configured-runner-input.ts`). Hook Stop przyjmuje dziś tylko ścieżki pod
`~/.claude` — transkrypt ZCode był pomijany bez śladu. Testy: ścieżka ZCode przyjęta i policzona; ścieżka Claude Code bez zmian
i bez nowego licznika; każda inna ścieżka (dowolny katalog tymczasowy, `..`, dowiązanie, zła nazwa) pominięta i POLICZONA.
Jedyna atrapa: `capture_text` (zapis do bazy) — podstawiony rejestrator tekstu, żeby test nie pisał do SurrealDB.
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from surreal_memory.hooks import liczniki, stop

TEKST = "Decision: chose the port of the ZCode profile parser because the fallback gate must match ZCode exactly."


@pytest.fixture
def stanowisko(tmp_path, monkeypatch):
    home, tmpdir, dane = tmp_path / "home", tmp_path / "tmp", tmp_path / "dane"
    for d in (home / ".claude/projects/p", tmpdir, dane):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(dane))
    monkeypatch.setattr(tempfile, "tempdir", str(tmpdir))
    zapisane: list[str] = []

    async def rejestrator(
        text, project_name=None, **_kw
    ):  # atrapa zapisu do bazy (nazwana w docstringu modułu); **_kw: session_id z wejścia hooka
        zapisane.append(text)
        return {"saved": 1}

    monkeypatch.setattr(stop, "capture_text", rejestrator)
    return {"home": home, "tmp": tmpdir, "dane": dane, "zapisane": zapisane}


def _linia(tekst: str) -> str:
    return (
        json.dumps({"message": {"content": [{"text": tekst, "type": "text"}], "role": "assistant"}})
        + "\n"
    )


def _linia_czlowieka(tekst: str) -> str:
    """Wpis wypowiedzi człowieka w formacie transkryptu Claude Code."""
    return (
        json.dumps(
            {
                "type": "user",
                "isSidechain": False,
                "origin": {"kind": "human"},
                "message": {"role": "user", "content": [{"type": "text", "text": tekst}]},
            }
        )
        + "\n"
    )


def _bieg(monkeypatch, sciezka: Path | str) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "hook_event_name": "Stop",
                    "session_id": "s",
                    "transcript_path": str(sciezka),
                    "cwd": "/tmp",
                }
            )
        ),
    )
    monkeypatch.setattr("sys.argv", ["smem-hook-stop"])
    kod: object = None
    try:
        stop.main()
    except SystemExit as e:
        kod = e.code
    assert kod in (0, None)  # Stop nigdy nie blokuje zakończenia sesji


def _liczniki(dane: Path) -> dict[str, int]:
    p = dane / liczniki.LICZNIKI_PLIK
    if not p.exists():
        return {}
    wiersze = json.loads(p.read_text())
    return {k: v for doba in wiersze.values() for k, v in doba.items()}


def test_transkrypt_hooka_zcode_przyjety_i_policzony(stanowisko, monkeypatch):
    k = stanowisko["tmp"] / "zcode-hook-a1B2c3"
    k.mkdir()
    (k / "transcript.jsonl").write_text(_linia(TEKST))
    _bieg(monkeypatch, k / "transcript.jsonl")
    # W-3 (smem-higiena-autocapture): transkrypt hooka ZCode niesie jedną linię roli `assistant`, a detektor
    # czyta wyłącznie wypowiedzi człowieka — skutek ZAMIERZONY: ścieżka przyjęta i POLICZONA, zapisu brak.
    # Przed zmianą ten test wymagał zapisu `[TEKST]`.
    assert stanowisko["zapisane"] == []
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_zcode": 1}


def test_transkrypt_claude_code_bez_zmian_i_bez_licznika(stanowisko, monkeypatch):
    """Kontrola: ścieżka CC pod `~/.claude` przechwytywana jak dotąd; nowy kod nie pisze licznika na ścieżce CC (zero nowego I/O)."""
    p = stanowisko["home"] / ".claude/projects/p/sesja.jsonl"
    # W-3: ścieżka CC przechwytuje wypowiedź człowieka (przed zmianą: linia roli `assistant`).
    p.write_text(_linia_czlowieka(TEKST))
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {}


@pytest.mark.parametrize(
    "budowa", ["dowolny-tmp", "ucieczka", "zla-nazwa", "dowiazanie", "poza-tmp"]
)
def test_inna_sciezka_pominieta_i_policzona(stanowisko, monkeypatch, budowa):
    t = stanowisko["tmp"]
    if budowa == "dowolny-tmp":
        (t / "inny").mkdir()
        p = t / "inny/transcript.jsonl"
        p.write_text(_linia(TEKST))
    elif budowa == "ucieczka":
        (t / "zcode-hook-x").mkdir()
        cel = stanowisko["home"] / "transcript.jsonl"
        cel.write_text(_linia(TEKST))
        p = t / "zcode-hook-x/../../home/transcript.jsonl"
    elif budowa == "zla-nazwa":
        (t / "zcode-hook-x").mkdir()
        p = t / "zcode-hook-x/inny.jsonl"
        p.write_text(_linia(TEKST))
    elif budowa == "dowiazanie":
        (t / "zcode-hook-x").mkdir()
        cel = stanowisko["home"] / "sekret.jsonl"
        cel.write_text(_linia(TEKST))
        p = t / "zcode-hook-x/transcript.jsonl"
        p.symlink_to(cel)
    else:
        (stanowisko["home"] / "zcode-hook-x").mkdir()
        p = stanowisko["home"] / "zcode-hook-x/transcript.jsonl"
        p.write_text(_linia(TEKST))
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == []
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_pominiety": 1}


def test_krotka_odpowiedz_zcode_bez_zapisu_ale_policzona(stanowisko, monkeypatch):
    """Ścieżka przyjęta (licznik +1), ale treść < 50 znaków — jak w CC: brak zapisu."""
    k = stanowisko["tmp"] / "zcode-hook-k"
    k.mkdir()
    (k / "transcript.jsonl").write_text(_linia("Decision: OK."))
    _bieg(monkeypatch, k / "transcript.jsonl")
    assert stanowisko["zapisane"] == []
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_zcode": 1}
