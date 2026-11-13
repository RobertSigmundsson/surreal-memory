"""W-4 (smem-higiena-autocapture): bramka punktowa na korpusie — POS-K, NEG i selektywność S0.

* POS-K (``pos-kontrolne.jsonl``, 50 świadomych zapisów ``source='user_input'``) przez ``check_write_gate`` z kontekstem haka
  stop i domyślną konfiguracją gałęzi (+ ``enforce``) ⇒ ≥ 95 % przyjętych (KA.10); kontrola pozytywna: pozycji ≥ 40;
  kontrola zerowa / mutant „bramka odrzuca wszystko” ⇒ ten sam przyrząd daje 0 % ⇒ czerwony.
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
POSK_PROG = 0.95
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


def _posk_przyjete(bramka: Callable[..., Any]) -> tuple[int, int]:
    """(liczba pozycji POS-K, przyjęte) — kontekst jak hak stop (``build_capture_context``)."""
    pozycje = _wczytaj("pos-kontrolne.jsonl")
    przyjete = 0
    for p in pozycje:
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
        if not wynik.rejected:
            przyjete += 1
    return len(pozycje), przyjete


@wymaga_korpusu
def test_posk_kontrola_pozytywna_liczebnosc() -> None:
    """Kontrola pozytywna przyrządu: korpus ma ≥ 40 pozycji — inaczej „95 %” niczego nie dowodzi."""
    n, _ = _posk_przyjete(check_write_gate)
    assert n >= POSK_MIN_LICZNOSC


@wymaga_korpusu
def test_posk_przyjete_co_najmniej_95_procent() -> None:
    n, przyjete = _posk_przyjete(check_write_gate)
    assert n >= POSK_MIN_LICZNOSC
    assert przyjete / n >= POSK_PROG, f"POS-K: przyjęto {przyjete}/{n} (< {POSK_PROG:.0%})"


@wymaga_korpusu
def test_mutant_bramka_odrzuca_wszystko_jest_wykryty() -> None:
    """Mutant: bramka odrzucająca wszystko ⇒ ten sam przyrząd daje 0 % ⇒ asercja 95 % byłaby CZERWONA."""

    def odrzuca_wszystko(*_a: Any, **_k: Any) -> Any:
        return SimpleNamespace(rejected=True)

    n, przyjete = _posk_przyjete(odrzuca_wszystko)
    assert n >= POSK_MIN_LICZNOSC
    assert przyjete == 0
    assert not (przyjete / n >= POSK_PROG)


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
