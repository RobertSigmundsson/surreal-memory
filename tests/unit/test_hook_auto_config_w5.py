"""W-5 (smem-higiena-autocapture): haki Stop i PreCompact czytają ``config.auto`` (``capture_<typ>``, ``min_confidence``).

Przed zmianą oba haki wołały detektor z ``capture_*=True`` na sztywno, więc ``[auto] capture_<typ> = false`` w configu nie
miało na nie wpływu, a PreCompact ignorował ``min_confidence``. Testy biegną na PRAWDZIWYM detektorze i izolowanej
bazie w pamięci (jak ``test_hook_capture_idempotency.py``): treść kontrolna zawiera po jednej linii każdego z sześciu typów;
dla każdego ``capture_<typ>=False`` linia tego typu NIE jest zapisana, a pozostałe pięć TAK. Kontrola pozytywna: wszystko
włączone ⇒ zapisanych sześć. Mutant „``True`` na sztywno" ⇒ czerwony (``test_*[typ]`` dla wyłączonego typu).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from surreal_memory.hooks import pre_compact as pre_compact_hook
from surreal_memory.hooks import stop as stop_hook
from surreal_memory.unified_config import AutoConfig

pytestmark = pytest.mark.asyncio

# typ -> (linia kontrolna, rozpoznawalny fragment w zapisanej treści)
KONTROLNE: dict[str, tuple[str, str]] = {
    "decision": ("We decided to keep the ledger under the project folder alpha.", "ledger"),
    "error": ("Error: the loader crashes when the manifest file is empty.", "loader crashes"),
    "todo": (
        "TODO: rotate the credentials of the staging bucket next week.",
        "rotate the credentials",
    ),
    "fact": (
        "The solution is: restart the worker after changing the schema.",
        "restart the worker",
    ),
    "insight": ("Turns out: the cache ignored the tenant prefix entirely.", "tenant prefix"),
    "preference": (
        "I prefer: small commits with descriptive messages for review.",
        "small commits",
    ),
}
TEKST = "\n".join(linia for linia, _ in KONTROLNE.values())
# typ -> nazwa pola w ``AutoConfig``
POLE: dict[str, str] = {
    "decision": "capture_decisions",
    "error": "capture_errors",
    "todo": "capture_todos",
    "fact": "capture_facts",
    "insight": "capture_insights",
    "preference": "capture_preferences",
}


@pytest.fixture(autouse=True)
def _izolacja(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Izolowana baza w pamięci pod HOME w tmp_path; żadnego dostępu do żywej bazy ani ``~/.surrealmemory``."""
    monkeypatch.setenv("SURREAL_MEMORY_STORAGE", "memory")
    monkeypatch.delenv("SURREAL_MEMORY_DIR", raising=False)
    monkeypatch.delenv("SURREALDB_URL", raising=False)
    monkeypatch.delenv("SURREALDB_PASS", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TMPDIR", "/tmp")
    monkeypatch.setenv("SURREAL_MEMORY_BRAIN", "w5" + uuid.uuid4().hex[:12])
    monkeypatch.setenv("CLAUDE_SESSION_ID", "sess-" + uuid.uuid4().hex[:12])

    from surreal_memory.unified_config import get_config

    get_config(reload=True)


def _auto(**ustaw: object) -> AutoConfig:
    from surreal_memory.unified_config import get_config

    auto = get_config().auto
    for klucz, wartosc in ustaw.items():
        setattr(auto, klucz, wartosc)
    return auto


async def _bieg(haki: str) -> dict[str, object]:
    if haki == "stop":
        return await stop_hook.capture_text(TEKST, project_name=None)
    return await pre_compact_hook.flush_text(TEKST, project_name=None)


def _zapisane(wynik: dict[str, object]) -> str:
    memories = wynik.get("memories", [])
    assert isinstance(memories, list)
    return " | ".join(str(m) for m in memories)


@pytest.mark.parametrize("haki", ["stop", "pre_compact"])
async def test_kontrola_pozytywna_wszystkie_typy_wlaczone_zapisane(haki: str) -> None:
    """Domyślny config (wszystkie ``capture_*`` true): sześć linii kontrolnych ⇒ sześć zapisów."""
    wynik = await _bieg(haki)
    zapisane = _zapisane(wynik)
    assert wynik["saved"] == len(KONTROLNE), wynik
    for typ, (_, fragment) in KONTROLNE.items():
        assert fragment in zapisane, f"{haki}: brak {typ} przy włączonych wszystkich typach"


@pytest.mark.parametrize("typ", sorted(KONTROLNE))
@pytest.mark.parametrize("haki", ["stop", "pre_compact"])
async def test_capture_typ_false_nie_zapisuje_tego_typu_a_pozostale_tak(
    haki: str, typ: str
) -> None:
    _auto(**{POLE[typ]: False})
    wynik = await _bieg(haki)
    zapisane = _zapisane(wynik)
    assert KONTROLNE[typ][1] not in zapisane, f"{haki}: {typ} zapisany mimo {POLE[typ]}=false"
    assert wynik["saved"] == len(KONTROLNE) - 1, wynik
    for inny, (_, fragment) in KONTROLNE.items():
        if inny != typ:
            assert fragment in zapisane, f"{haki}: {inny} zgubiony przy wyłączeniu tylko {typ}"


@pytest.mark.parametrize("haki", ["stop", "pre_compact"])
async def test_wszystkie_capture_false_nic_nie_zapisane(haki: str) -> None:
    _auto(**dict.fromkeys(POLE.values(), False))
    wynik = await _bieg(haki)
    assert wynik["saved"] == 0, wynik


# pewności detektora dla linii kontrolnych: decision 0,8 · error 0,85 · todo 0,75 · fact 0,7 · insight 0,8 · preference 0,85
@pytest.mark.parametrize("haki", ["stop", "pre_compact"])
async def test_min_confidence_08_odrzuca_pozycje_o_pewnosci_075(haki: str) -> None:
    """``min_confidence=0.8``: ``todo`` (0,75) i ``fact`` (0,7) NIE zapisane w obu haktach; pozostałe cztery tak."""
    _auto(min_confidence=0.8)
    wynik = await _bieg(haki)
    zapisane = _zapisane(wynik)
    assert KONTROLNE["todo"][1] not in zapisane, (
        f"{haki}: todo (0,75) zapisany przy min_confidence=0,8"
    )
    assert KONTROLNE["fact"][1] not in zapisane, (
        f"{haki}: fact (0,7) zapisany przy min_confidence=0,8"
    )
    for typ in ("decision", "error", "insight", "preference"):
        assert KONTROLNE[typ][1] in zapisane, f"{haki}: {typ} zgubiony"
    assert wynik["saved"] == 4, wynik


@pytest.mark.parametrize("haki", ["stop", "pre_compact"])
async def test_min_confidence_07_zapisuje_pozycje_o_pewnosci_075(haki: str) -> None:
    """Para pozytywna: ``min_confidence=0.7`` (domyślne) ⇒ ``todo`` (0,75) zapisany w obu haktach."""
    _auto(min_confidence=0.7)
    zapisane = _zapisane(await _bieg(haki))
    assert KONTROLNE["todo"][1] in zapisane


async def test_pre_compact_zwraca_efektywny_prog() -> None:
    """Pre-compact raportuje próg faktycznie użyty: ``max(EMERGENCY_THRESHOLD, min_confidence)``."""
    _auto(min_confidence=0.8)
    assert (await _bieg("pre_compact"))["threshold"] == 0.8
    _auto(min_confidence=0.3)
    assert (await _bieg("pre_compact"))["threshold"] == pre_compact_hook.EMERGENCY_THRESHOLD
