"""Trwałe liczniki DOBOWE hooków sesji Claude Code.

Agregat per doba (UTC), nie wiersz na zdarzenie: bramka długości hooka UserPromptSubmit i lista SessionStart
odpalają się na KAŻDY prompt / start sesji, więc wiersz na zdarzenie zalewałby plik, a bez zapisu (stan sprzed
zmiany) „0 wpisów" nie było pomiarem. Rozmiar ograniczony z konstrukcji: jeden klucz na dobę.

Plik: ``<data_dir>/prompt_recall_liczniki.json`` = ``{"YYYY-MM-DD": {"<licznik>": n, ...}}``.
Zapis: read-modify-write pod ``fcntl.flock`` (osobny plik zamka) + atomowy ``os.replace`` — dwa procesy hooka
uruchomione równocześnie nie gubią inkrementów. Plik nieczytelny jest ODSUWANY (``.uszkodzony-<ts>``, dowód
zostaje), liczenie zaczyna się od nowa. Każda porażka: jedna linia na stderr i ``False``; hook nigdy nie blokuje prompta.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

LICZNIKI_PLIK = "prompt_recall_liczniki.json"
_ZAMEK = ".prompt_recall_liczniki.lock"


def _data_dir() -> Path:
    custom = os.environ.get("SURREAL_MEMORY_DIR", "")
    return Path(custom) if custom else (Path.home() / ".surrealmemory")


def _wczytaj(sciezka: Path, teraz: dt.datetime) -> dict[str, dict[str, int]]:
    if not sciezka.exists():
        return {}
    try:
        dane = json.loads(sciezka.read_text(encoding="utf-8"))
        if not isinstance(dane, dict):
            raise ValueError("korzeń nie jest obiektem")
        return dane
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        odsuniety = sciezka.with_name(
            f"{sciezka.name}.uszkodzony-{teraz.strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        )
        os.replace(sciezka, odsuniety)
        print(  # noqa: T201
            f"[Surreal-Memory] licznik dobowy: plik nieczytelny, odsunięty do {odsuniety.name}",
            file=sys.stderr,
        )
        return {}


def zwieksz(delta: Mapping[str, int], *, teraz: dt.datetime | None = None) -> bool:
    """Zwiększa liczniki bieżącej doby (UTC). ``True`` = zapisano; ``False`` = porażka (jedna linia stderr)."""
    if not delta:
        return True
    moment = teraz or dt.datetime.now(dt.UTC)
    doba = moment.strftime("%Y-%m-%d")
    katalog = _data_dir()
    try:
        katalog.mkdir(parents=True, exist_ok=True)
        with open(katalog / _ZAMEK, "a+", encoding="utf-8") as zamek:
            fcntl.flock(zamek, fcntl.LOCK_EX)
            sciezka = katalog / LICZNIKI_PLIK
            dane = _wczytaj(sciezka, moment)
            wiersz = dane.setdefault(doba, {})
            for klucz, n in delta.items():
                wiersz[klucz] = int(wiersz.get(klucz, 0)) + int(n)
            tmp = katalog / f"{LICZNIKI_PLIK}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps(dane, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, sciezka)
        return True
    except OSError as exc:
        print(  # noqa: T201
            f"[Surreal-Memory] licznik dobowy: zapis nieudany ({type(exc).__name__})",
            file=sys.stderr,
        )
        return False
