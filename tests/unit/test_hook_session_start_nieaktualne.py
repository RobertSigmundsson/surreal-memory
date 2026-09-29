"""SessionStart: lista „Recent Memories" bez wpisów nieaktualnych (`valid_until`) + dobowy licznik.

Filtr działa PO wycięciu pierwszej dziesiątki (`typed[:CONTEXT_LIMIT]`): lista jest ŚCISŁYM podzbiorem dzisiejszej —
żadna dalsza pozycja nie wchodzi w miejsce odfiltrowanej (to byłaby zmiana treści widocznej dla Roberta, nie samo
odfiltrowanie). Ten sam predykat co recall (`is_excluded_by_validity`, z furtką SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from surreal_memory.hooks import liczniki
from surreal_memory.hooks.session_start import CONTEXT_LIMIT, get_recent_memories


def _tm(i: int, *, nieaktualny: bool = False) -> MagicMock:
    tm = MagicMock()
    tm.fiber_id = f"f{i}"
    tm.valid_until = datetime(2026, 9, 1) if nieaktualny else None
    return tm


def _storage(typed: list[MagicMock]) -> AsyncMock:
    def _fiber(fid: str) -> MagicMock:
        f = MagicMock()
        f.summary = f"wspomnienie {fid}"
        f.essence = None
        return f

    st = AsyncMock()
    st.get_project_memories = AsyncMock(return_value=typed)
    st.get_fiber = AsyncMock(side_effect=_fiber)
    st.close = AsyncMock()
    return st


async def _lista(typed: list[MagicMock]) -> list[str]:
    cfg = MagicMock()
    cfg.current_brain = "b"
    with (
        patch("surreal_memory.unified_config.get_config", return_value=cfg),
        patch("surreal_memory.unified_config.get_shared_storage", return_value=_storage(typed)),
    ):
        return (await get_recent_memories("proj")).splitlines()


@pytest.mark.asyncio
async def test_nieaktualny_wpis_nie_trafia_na_liste(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    linie = await _lista([_tm(0), _tm(1, nieaktualny=True), _tm(2)])
    assert linie == ["- wspomnienie f0", "- wspomnienie f2"]


@pytest.mark.asyncio
async def test_lista_jest_scislym_podzbiorem_dzisiejszej_zadna_pozycja_nie_wchodzi_w_miejsce_odfiltrowanej(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    typed = [_tm(i, nieaktualny=(i == 3)) for i in range(CONTEXT_LIMIT + 2)]
    linie = await _lista(typed)
    przed = [f"- wspomnienie f{i}" for i in range(CONTEXT_LIMIT)]  # tyle dawało kod sprzed zmiany
    assert set(linie) < set(przed) and len(linie) == CONTEXT_LIMIT - 1
    assert f"- wspomnienie f{CONTEXT_LIMIT}" not in linie  # 11. NIE wskakuje
    assert linie == [x for x in przed if x != "- wspomnienie f3"]  # kolejność zachowana


@pytest.mark.asyncio
async def test_furtka_env_zostawia_nieaktualne_kontrola_dodatnia(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    monkeypatch.setenv("SURREAL_MEMORY_DISABLE_SUPERSEDED_FILTER", "1")
    assert await _lista([_tm(0), _tm(1, nieaktualny=True)]) == [
        "- wspomnienie f0",
        "- wspomnienie f1",
    ]


@pytest.mark.asyncio
async def test_licznik_dobowy_wywolania_pokazane_odfiltrowane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    await _lista([_tm(0), _tm(1, nieaktualny=True), _tm(2, nieaktualny=True), _tm(3)])
    dane = json.loads((tmp_path / liczniki.LICZNIKI_PLIK).read_text())
    (doba,) = dane.values()
    assert doba == {
        "sessionstart_wywolania": 1,
        "sessionstart_pokazane": 2,
        "sessionstart_odfiltrowane_nieaktualne": 2,
    }


@pytest.mark.asyncio
async def test_bez_projektu_i_bez_wpisow_nie_liczy_wywolania(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(tmp_path))
    assert await _lista([]) == []
    assert not (tmp_path / liczniki.LICZNIKI_PLIK).exists()
