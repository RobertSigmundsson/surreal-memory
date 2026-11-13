"""W-4 (smem-higiena-autocapture): bramka punktowa na korpusie — POS-K, NEG i selektywność S0.

* POS-K (``pos-kontrolne.jsonl``, 50 świadomych zapisów ``source='user_input'``) przez ``check_write_gate`` z kontekstem haka
  stop i domyślną konfiguracją gałęzi (+ ``enforce``) ⇒ test RAPORTUJE ``przyjęte POS-K = n/N`` (kryteria v1.4: liczba
  raportowana z dolnym progiem regresji n ≥ 18; W-4b, próg domyślny 5); kontrola pozytywna: pozycji ≥ 40; mutant „odrzuca
  wszystko” ⇒ 0 (asercja dolnego progu CZERWONA), „akceptuje wszystko” ⇒ N (próg spełniony) — przyrząd liczy, nie zakłada.
* W∩POS-K (pozycje WARTOSCIOWE u A-r1 ∩ B-r3): TYLKO pomiar ``przyjęte W∩POS-K = m/k``, bez progu (próg liczy przyrząd runnera).
* NEG (150 wpisów, które nie są wypowiedzią człowieka) przez ``read_transcript_tail`` haka stop, detektor i bramkę ⇒ 0 przyjętych.
* S0 (selektywność): przyrząd ``qa/u1b-pomiary/s0_selektywnosc.py`` uruchomiony na tym drzewie ⇒ odrzuca ≥ 14 i > 5 z 31
  (``WYNIK: PASS``); kontrola: ten sam przyrząd na wariancie „kod przed W-4” (``V0_kod_dzis_prog5``) ⇒ ``FAIL``.

Korpus: zmienna ``KORPUS_DETEKTORA`` (katalog ``qa/korpus-detektora`` z repo ``nautilus-ops``); bez niej testy korpusowe
SKIP z powodem. Test S0 wymaga też ``<qa>/u1b-pomiary/s0_selektywnosc.py`` i ``<qa>/vgate/etykiety-{A,B}-r1.jsonl``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import surreal_memory
from surreal_memory.engine.quality_scorer import check_write_gate
from surreal_memory.hooks import stop
from surreal_memory.mcp.auto_capture import analyze_text_for_memories, build_capture_context
from surreal_memory.unified_config import WriteGateConfig

KORPUS_ENV = "KORPUS_DETEKTORA"
POSK_MIN_LICZNOSC = 40
POSK_DOLNY_PROG = 18  # pomiar projektu H1 + próg 5 (13.11); „nie gorzej niż zmierzony projekt” — dolny próg regresji, nie cel
TAGI_HAKA = ["stop_hook", "session_end", "project:uruboros"]


def _katalog() -> Path | None:
    surowy = os.environ.get(KORPUS_ENV, "")
    if not surowy:
        return None
    k = Path(surowy)
    return k if (k / "pos-kontrolne.jsonl").is_file() else None


_KORPUS = _katalog()
wymaga_korpusu = pytest.mark.skipif(
    _KORPUS is None,
    reason=f"{KORPUS_ENV} nie wskazuje katalogu z pos-kontrolne.jsonl (korpus żyje w repo nautilus-ops, qa/korpus-detektora/)",
)


def _wczytaj(nazwa: str) -> list[dict[str, Any]]:
    assert _KORPUS is not None
    with open(_KORPUS / nazwa, encoding="utf-8") as f:
        return [json.loads(w) for w in f if w.strip()]


def _cfg_galezi() -> WriteGateConfig:
    """Wartości domyślne ``WriteGateConfig`` gałęzi + tryby ``enforce`` (jak KA.10)."""
    return WriteGateConfig(enabled=True, mode="enforce", auto_capture_mode="enforce")


def _posk_wyniki(bramka: Callable[..., Any]) -> dict[str, bool]:
    """id pozycji POS-K → czy przyjęta (kontekst jak hak stop, ``build_capture_context``)."""
    wyniki: dict[str, bool] = {}
    for p in _wczytaj("pos-kontrolne.jsonl"):
        wynik = bramka(
            p["tresc"],
            gate_config=_cfg_galezi(),
            is_auto_capture=True,
            memory_type=p["memory_type"],
            tags=list(TAGI_HAKA),
            context=build_capture_context(
                {"type": p["memory_type"], "confidence": 0.8}, source="stop_hook"
            ),
        )
        wyniki[p["id"]] = not wynik.rejected
    return wyniki


def _posk_przyjete(bramka: Callable[..., Any]) -> tuple[int, int]:
    """(liczba pozycji POS-K, przyjęte)."""
    wyniki = _posk_wyniki(bramka)
    return len(wyniki), sum(wyniki.values())


def _asercja_dolnego_progu(przyjete: int) -> None:
    """Dolny próg regresji POS-K (kryteria v1.4 KA.10): przyjęte ≥ ``POSK_DOLNY_PROG`` (H1 + próg 5 = 18/50)."""
    assert przyjete >= POSK_DOLNY_PROG, (
        f"POS-K: przyjęto {przyjete} (< dolny próg {POSK_DOLNY_PROG})"
    )


@wymaga_korpusu
def test_posk_kontrola_pozytywna_liczebnosc() -> None:
    """Kontrola pozytywna przyrządu: korpus ma ≥ 40 pozycji — inaczej liczba przyjętych niczego nie mówi."""
    n, _ = _posk_przyjete(check_write_gate)
    assert n >= POSK_MIN_LICZNOSC


@wymaga_korpusu
def test_posk_raportuje_przyjete_liczone_przyrzadem(capsys: pytest.CaptureFixture[str]) -> None:
    """POS-K to liczba RAPORTOWANA, nie próg (kryteria v1.4, D-18): przy progu 5 część świadomych zapisów odpada.

    Test nie zakłada wyniku — liczy go przyrząd (``_posk_przyjete`` = ``check_write_gate`` z domyślną konfiguracją
    gałęzi) i drukuje ``przyjęte POS-K = n/N``; asercje: poprawność przyrządu (liczność, zakres) i dolny próg regresji
    ``przyjęte ≥ 18`` (kryteria v1.4; nie cel — wartość z pomiaru projektu H1 + próg 5).
    """
    n, przyjete = _posk_przyjete(check_write_gate)
    with capsys.disabled():
        print(f"\nprzyjęte POS-K = {przyjete}/{n}")
    assert n >= POSK_MIN_LICZNOSC
    assert 0 <= przyjete <= n
    _asercja_dolnego_progu(przyjete)


@wymaga_korpusu
def test_mutant_bramka_odrzuca_wszystko_jest_wykryty() -> None:
    """Mutant: bramka odrzucająca wszystko ⇒ ten sam przyrząd daje 0 przyjętych (przyrząd liczy, nie zakłada)."""

    def odrzuca_wszystko(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(rejected=True)

    n, przyjete = _posk_przyjete(odrzuca_wszystko)
    assert n >= POSK_MIN_LICZNOSC
    assert przyjete == 0
    with pytest.raises(
        AssertionError, match="dolny próg"
    ):  # kontrola zerowa: asercja dolnego progu CZERWONA przy 0/N
        _asercja_dolnego_progu(przyjete)


@wymaga_korpusu
def test_mutant_bramka_akceptuje_wszystko_daje_cala_liczbe_pozycji() -> None:
    """Mutant: bramka akceptująca wszystko ⇒ ten sam przyrząd daje n/n; dolny próg spełniony (próg dolny, nie górny)."""

    def akceptuje_wszystko(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(rejected=False)

    n, przyjete = _posk_przyjete(akceptuje_wszystko)
    assert n >= POSK_MIN_LICZNOSC
    assert przyjete == n
    _asercja_dolnego_progu(przyjete)


def _etykiety_wartosciowe() -> set[str] | None:
    """id oznaczone WARTOSCIOWE przez OBIE rodziny (A-r1 ∩ B-r3) albo None, gdy pliki etykiet są poza zasięgiem testu."""
    if _KORPUS is None:
        return None
    vgate = _KORPUS.parent / "vgate"
    zbiory = []
    for nazwa in ("etykiety-A-r1.jsonl", "etykiety-B-r3.jsonl"):
        plik = vgate / nazwa
        if not plik.is_file():
            return None
        with open(plik, encoding="utf-8") as f:
            wiersze = [json.loads(w) for w in f if w.strip()]
        zbiory.append({w["id"] for w in wiersze if w["etykieta"] == "WARTOSCIOWE"})
    return zbiory[0] & zbiory[1]


@wymaga_korpusu
def test_w_posk_raportuje_przyjete_bez_progu(capsys: pytest.CaptureFixture[str]) -> None:
    """W∩POS-K (etykiety zgodne A∩B): TYLKO pomiar — drukuje ``przyjęte W∩POS-K = m/k``; progu tu nie ma (próg: przyrząd runnera)."""
    w = _etykiety_wartosciowe()
    if w is None:
        pytest.skip(
            "brak etykiet qa/vgate/etykiety-{A-r1,B-r3}.jsonl obok korpusu — W∩POS-K liczy przyrząd w qa/scripts"
        )
    wyniki = _posk_wyniki(check_write_gate)
    wspolne = {i: v for i, v in wyniki.items() if i in w}
    with capsys.disabled():
        print(f"\nprzyjęte W∩POS-K = {sum(wspolne.values())}/{len(wspolne)}")
    assert len(wspolne) > 0  # kontrola pozytywna: etykiety trafiają w id korpusu
    _, akceptuje = _posk_przyjete_wspolne(lambda *a, **k: SimpleNamespace(rejected=False), w)
    assert akceptuje == len(wspolne)  # mutant „akceptuje wszystko” ⇒ k/k
    _, odrzuca = _posk_przyjete_wspolne(lambda *a, **k: SimpleNamespace(rejected=True), w)
    assert odrzuca == 0  # mutant „odrzuca wszystko” ⇒ 0/k


def _posk_przyjete_wspolne(bramka: Callable[..., Any], w: set[str]) -> tuple[int, int]:
    wspolne = {i: v for i, v in _posk_wyniki(bramka).items() if i in w}
    return len(wspolne), sum(wspolne.values())


@wymaga_korpusu
def test_neg_nie_przechodzi_przez_hak_detektor_i_bramke(tmp_path: Path) -> None:
    """NEG (150) → ``read_transcript_tail`` haka stop → detektor → bramka ⇒ 0 przyjętych przechwyceń."""
    neg = _wczytaj("neg.jsonl")
    assert len(neg) == 150
    cfg = _cfg_galezi()
    przyjete = 0
    for i, poz in enumerate(neg):
        plik = tmp_path / f"neg-{i}.jsonl"
        plik.write_text(json.dumps(poz["wpis"], ensure_ascii=False) + "\n", encoding="utf-8")
        tekst = stop.read_transcript_tail(str(plik))
        if len(tekst.strip()) < 50:
            continue
        for it in analyze_text_for_memories(
            tekst,
            capture_decisions=True,
            capture_errors=True,
            capture_todos=True,
            capture_facts=True,
            capture_insights=True,
            capture_preferences=True,
        ):
            r = check_write_gate(
                it["content"],
                gate_config=cfg,
                is_auto_capture=True,
                memory_type=it.get("type"),
                tags=list(TAGI_HAKA),
                context=build_capture_context(it, source="stop_hook"),
            )
            if not r.rejected:
                przyjete += 1
    assert przyjete == 0


# --- S0: przyrząd selektywności -------------------------------------------------------------------------------------


def _przyrzad_s0() -> Path | None:
    if _KORPUS is None:
        return None
    qa = _KORPUS.parent
    skrypt = qa / "u1b-pomiary" / "s0_selektywnosc.py"
    etykiety = [qa / "vgate" / f"etykiety-{x}-r1.jsonl" for x in ("A", "B")]
    return skrypt if skrypt.is_file() and all(e.is_file() for e in etykiety) else None


_S0 = _przyrzad_s0()
wymaga_s0 = pytest.mark.skipif(
    _S0 is None,
    reason=f"{KORPUS_ENV} bez sąsiadów qa/u1b-pomiary/s0_selektywnosc.py i qa/vgate/etykiety-{{A,B}}-r1.jsonl",
)


def _uruchom_s0(*argumenty: str) -> subprocess.CompletedProcess[str]:
    assert _S0 is not None
    src = str(Path(surreal_memory.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src}
    return subprocess.run(  # noqa: S603 -- stały przyrząd z korpusu, bez powłoki
        [sys.executable, str(_S0), *argumenty],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )


@wymaga_s0
def test_s0_selektywnosc_wykonanie_na_tym_drzewie_przechodzi() -> None:
    """Odrzucone ≥ 14 i > 5 z 31 (kod tego drzewa z domyślnym ``WriteGateConfig``)."""
    r = _uruchom_s0("--wykonanie", "VG_kod_w_drzewie_domyslne")
    assert "WYNIK: PASS" in r.stdout, r.stdout[-600:] + r.stderr[-600:]
    assert r.returncode == 0


@wymaga_s0
def test_s0_kontrola_zerowa_mutant_akceptuje_wszystko_odrzuca_zero() -> None:
    """Kontrola zerowa: bramka akceptująca wszystko odrzuca 0/31 i NIE przechodzi progu."""
    r = _uruchom_s0("--wykonanie", "VG_kod_w_drzewie_domyslne")
    assert (
        "ZERO_akceptuje_wszystko (akceptuje-wszystko): odrzucone 0/31; przechodzi próg: NIE"
        in r.stdout
    )


@wymaga_s0
def test_s0_kontrola_zerowa_przyrzad_liczy_kolumne_mutanta() -> None:
    """Mutant „odrzuca wszystko” w tej samej pętli ⇒ kolumna ZERO == 31, próg przechodzi, przyrząd kończy rc 1."""
    r = _uruchom_s0("--mutant-zero", "odrzuca-wszystko")
    assert "odrzucone 31/31; przechodzi próg: TAK" in r.stdout, r.stdout[-600:]
    assert r.returncode == 1


@wymaga_s0
def test_s0_kontrola_pozytywna_prototyp_h1_prog4_odrzuca_14() -> None:
    """Kontrola pozytywna przyrządu: prototyp W-4 (H1 + próg 4) odrzuca 14/31 i przechodzi próg (rc 0)."""
    r = _uruchom_s0("--wykonanie", "V1_W4_H1_prog4")
    assert "odrzucone 14/31" in r.stdout and r.returncode == 0, r.stdout[-600:]


@wymaga_s0
def test_s0_nieznany_wariant_to_rc_64() -> None:
    assert _uruchom_s0("--wykonanie", "NIEISTNIEJACY").returncode == 64
