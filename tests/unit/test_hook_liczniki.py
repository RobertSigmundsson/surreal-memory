"""Trwałe liczniki DOBOWE hooków sesji (`surreal_memory.hooks.liczniki`) — REK-7 programu smem-jeden-silnik-odczytu.

Agregat per doba (UTC) zamiast wiersza na zdarzenie; zapis pod flock + atomowy os.replace; porażka nigdy
nie blokuje prompta i nigdy nie jest cicha (jedna linia stderr).
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from surreal_memory.hooks import liczniki

if TYPE_CHECKING:
    import pytest


def _wczytaj(katalog: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = json.loads((katalog / liczniki.LICZNIKI_PLIK).read_text())
    return out


def test_zwieksz_tworzy_plik_z_kluczem_doby(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    teraz = dt.datetime(2026, 9, 29, 23, 59, 59, tzinfo=dt.UTC)
    assert liczniki.zwieksz({"za_krotki": 1}, teraz=teraz) is True
    assert _wczytaj(tmp_path) == {"2026-09-29": {"za_krotki": 1}}


def test_sumuje_w_obrebie_doby_i_rozdziela_doby(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    d1 = dt.datetime(2026, 9, 29, 12, tzinfo=dt.UTC)
    d2 = dt.datetime(2026, 9, 30, 0, 0, 1, tzinfo=dt.UTC)
    liczniki.zwieksz({"za_krotki": 2, "inny": 1}, teraz=d1)
    liczniki.zwieksz({"za_krotki": 3}, teraz=d1)
    liczniki.zwieksz({"za_krotki": 1}, teraz=d2)
    assert _wczytaj(tmp_path) == {
        "2026-09-29": {"inny": 1, "za_krotki": 5},
        "2026-09-30": {"za_krotki": 1},
    }


def test_pusta_delta_nie_tworzy_pliku(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    assert liczniki.zwieksz({}) is True
    assert not (tmp_path / liczniki.LICZNIKI_PLIK).exists()


def test_dwa_procesy_po_100_inkrementow_to_200(tmp_path: Path) -> None:
    """Prawdziwa współbieżność (osobne procesy) — bez blokady read-modify-write gubi inkrementy."""
    kod = (
        "import sys;from surreal_memory.hooks import liczniki;"
        "[liczniki.zwieksz({'za_krotki': 1}) for _ in range(100)]"
    )
    env = {"SURREAL_MEMORY_DIR": str(tmp_path), "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    procesy = [
        subprocess.Popen([sys.executable, "-c", kod], env={**env, "PYTHONPATH": ":".join(sys.path)})  # noqa: S603
        for _ in range(2)
    ]
    assert [p.wait(timeout=120) for p in procesy] == [0, 0]
    dane = _wczytaj(tmp_path)
    assert sum(v.get("za_krotki", 0) for v in dane.values()) == 200


def test_plik_nieczytelny_odsuniety_licznik_dziala_dalej(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    (tmp_path / liczniki.LICZNIKI_PLIK).write_text("{ nie json")
    teraz = dt.datetime(2026, 9, 29, 1, tzinfo=dt.UTC)
    assert liczniki.zwieksz({"za_krotki": 1}, teraz=teraz) is True
    assert _wczytaj(tmp_path) == {"2026-09-29": {"za_krotki": 1}}
    odsuniete = list(tmp_path.glob(liczniki.LICZNIKI_PLIK + ".uszkodzony-*"))
    assert (
        len(odsuniete) == 1 and odsuniete[0].read_text() == "{ nie json"
    )  # dowód zachowany, nie skasowany
    assert "nieczytelny" in capsys.readouterr().err


def test_katalog_niezapisywalny_nie_rzuca_i_mowi_na_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plik = tmp_path / "to-jest-plik"
    plik.write_text("x")
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(plik))
    assert liczniki.zwieksz({"za_krotki": 1}) is False
    err = capsys.readouterr().err
    assert err.count("[Surreal-Memory] licznik dobowy: zapis nieudany") == 1


def test_plik_z_recznie_zepsuta_wartoscia_nie_rzuca_i_mowi_na_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    teraz = dt.datetime(2026, 9, 29, 1, tzinfo=dt.UTC)
    (tmp_path / liczniki.LICZNIKI_PLIK).write_text(json.dumps({"2026-09-29": {"za_krotki": "abc"}}))
    assert liczniki.zwieksz({"za_krotki": 1}, teraz=teraz) is False
    assert capsys.readouterr().err.count("[Surreal-Memory] licznik dobowy: zapis nieudany") == 1
