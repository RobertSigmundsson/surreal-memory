"""W-3 (smem-higiena-autocapture): detektor haków czyta WYŁĄCZNIE wypowiedzi człowieka.

Korpus ``KORPUS_DETEKTORA`` (katalog z ``neg.jsonl`` / ``pos.jsonl`` / ``MANIFEST.sha256``; żyje w repo ``nautilus-ops``,
``qa/korpus-detektora/`` v1.1) przechodzi przez PRAWDZIWE punkty wejścia obu haków — ``read_transcript_tail`` hooka stop i
hooka pre-compact — a potem przez detektor ``analyze_text_for_memories`` z progiem danego haka:

* NEG (150 wpisów, które NIE są wypowiedzią człowieka) ⇒ 0 przechwyceń w KAŻDYM haku;
* POS (53 wypowiedzi człowieka, w których stary detektor coś znajduje) ⇒ przechwycenia IDENTYCZNE ze ``stary_detektor``
  zapisanym w pozycji (W-3 nie może odebrać detektorowi tekstu człowieka);
* kontrola zerowa: stary tor (czytanie bez filtra) na KAŻDEJ z 150 pozycji NEG daje ≥ 1 przechwycenie — inaczej test
  niczego nie mierzy;
* s1-s3 (syntetyczne, bez danych z transkryptów): wklejka z zamknięciem Z ATRYBUTAMI, wklejka bez zamknięcia, zamknięcie gołe;
* mutanty (opisane przy testach): ten sam przyrząd musi dać CZERWONY na każdym z nich.

Bez zmiennej ``KORPUS_DETEKTORA`` testy korpusowe są pomijane z powodem (korpus nie leży w tym repo); s1-s3 i mutanty
syntetyczne działają zawsze.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from surreal_memory.hooks import pre_compact, stop
from surreal_memory.mcp.auto_capture import analyze_text_for_memories

KORPUS_ENV = "KORPUS_DETEKTORA"
# Długość, poniżej której hook w ``main`` w ogóle nie woła detektora (stop.py / pre_compact.py: ``len(text.strip()) < 50``).
MIN_TEKST_HAKA = 50

HAKI: dict[str, tuple[Any, float]] = {
    "stop": (stop, stop.DEFAULT_CONFIDENCE),
    "pre_compact": (pre_compact, pre_compact.EMERGENCY_THRESHOLD),
}


def _katalog_korpusu() -> Path | None:
    surowy = os.environ.get(KORPUS_ENV, "")
    if not surowy:
        return None
    katalog = Path(surowy)
    return (
        katalog if (katalog / "neg.jsonl").is_file() and (katalog / "pos.jsonl").is_file() else None
    )


_KORPUS = _katalog_korpusu()
wymaga_korpusu = pytest.mark.skipif(
    _KORPUS is None,
    reason=f"{KORPUS_ENV} nie wskazuje katalogu z neg.jsonl/pos.jsonl (korpus żyje w repo nautilus-ops, qa/korpus-detektora/)",
)


def _wczytaj(nazwa: str) -> list[dict[str, Any]]:
    assert _KORPUS is not None
    with open(_KORPUS / nazwa, encoding="utf-8") as f:
        return [json.loads(w) for w in f if w.strip()]


def _wykryj(tekst: str, prog: float) -> list[tuple[str, str, float]]:
    """Dokładnie to, co robi hook po przeczytaniu transkryptu: bramka długości, detektor, próg haka."""
    if len(tekst.strip()) < MIN_TEKST_HAKA:
        return []
    znalezione = analyze_text_for_memories(
        tekst,
        capture_decisions=True,
        capture_errors=True,
        capture_todos=True,
        capture_facts=True,
        capture_insights=True,
        capture_preferences=True,
    )
    return [
        (x["type"], x["content"][:160], x["confidence"])
        for x in znalezione
        if x["confidence"] >= prog
    ]


def _przez_hak(haki: str, wpis: dict[str, Any], tmp_path: Path) -> list[tuple[str, str, float]]:
    """Wpis jako jednolinijkowy transkrypt → ``read_transcript_tail`` haka → detektor z progiem haka."""
    modul, prog = HAKI[haki]
    plik = tmp_path / f"{haki}-transkrypt.jsonl"
    plik.write_text(json.dumps(wpis, ensure_ascii=False) + "\n", encoding="utf-8")
    return _wykryj(modul.read_transcript_tail(str(plik)), prog)


def _stary_tor(wpis: dict[str, Any]) -> list[tuple[str, str, float]]:
    """Czytanie BEZ filtra W-3 (jak ``stop.read_transcript_tail`` sprzed zmiany) — kontrola zerowa przyrządu."""
    rola = stop._get_entry_role(wpis)
    if rola == "tool":
        return []
    tekst = stop._extract_text(wpis)
    if not tekst or len(tekst) <= 20:
        return []
    if rola == "assistant" and not stop._has_memory_markers(tekst):
        return []
    return _wykryj(tekst, stop.DEFAULT_CONFIDENCE)


def _porazki_neg(haki: str, tmp_path: Path) -> list[str]:
    return [p["id"] for p in _wczytaj("neg.jsonl") if _przez_hak(haki, p["wpis"], tmp_path)]


def _niezgodne_pos(haki: str, tmp_path: Path) -> list[str]:
    wynik = []
    for p in _wczytaj("pos.jsonl"):
        oczekiwane = [(d["typ"], d["tresc"], d["conf"]) for d in p["stary_detektor"]]
        if _przez_hak(haki, p["wpis"], tmp_path) != oczekiwane:
            wynik.append(p["id"])
    return wynik


# --------------------------------------------------------------------------------------------------------------------
# Korpus
# --------------------------------------------------------------------------------------------------------------------


@wymaga_korpusu
def test_korpus_zgodny_z_manifestem() -> None:
    """Korpus zamrożony: sha256 neg/pos zgodne z MANIFEST.sha256 (dryf korpusu = czerwony, nie cichy zielony)."""
    assert _KORPUS is not None
    manifest = {}
    for linia in (_KORPUS / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if linia.strip():
            suma, sciezka = linia.split(None, 1)
            manifest[Path(sciezka.strip()).name] = suma
    for nazwa in ("neg.jsonl", "pos.jsonl"):
        assert hashlib.sha256((_KORPUS / nazwa).read_bytes()).hexdigest() == manifest[nazwa]


@wymaga_korpusu
def test_korpus_liczebnosc() -> None:
    assert len(_wczytaj("neg.jsonl")) == 150
    assert len(_wczytaj("pos.jsonl")) == 53


@wymaga_korpusu
def test_kontrola_zerowa_stary_tor_przechwytuje_kazda_pozycje_neg() -> None:
    """Kontrola zerowa (n = 0 ⇒ błąd): czytanie bez filtra daje ≥ 1 przechwycenie na KAŻDEJ z 150 pozycji NEG."""
    bez = [p["id"] for p in _wczytaj("neg.jsonl") if not _stary_tor(p["wpis"])]
    assert bez == [], f"przyrząd nie widzi śmieci w {len(bez)}/150 pozycji NEG: {bez[:5]}"


@wymaga_korpusu
def test_kontrola_przyrzadu_stary_tor_zgodny_ze_stary_detektor_w_pos() -> None:
    """Pozytywna kontrola przyrządu: ``_stary_tor`` odtwarza ``stary_detektor`` zapisany w korpusie (53/53)."""
    niezgodne = [
        p["id"]
        for p in _wczytaj("pos.jsonl")
        if _stary_tor(p["wpis"]) != [(d["typ"], d["tresc"], d["conf"]) for d in p["stary_detektor"]]
    ]
    assert niezgodne == []


@wymaga_korpusu
@pytest.mark.parametrize("haki", sorted(HAKI))
def test_neg_zero_przechwyczen(haki: str, tmp_path: Path) -> None:
    porazki = _porazki_neg(haki, tmp_path)
    assert porazki == [], (
        f"{haki}: {len(porazki)}/150 pozycji NEG z przechwyceniem, np. {porazki[:5]}"
    )


@wymaga_korpusu
@pytest.mark.parametrize("haki", sorted(HAKI))
def test_pos_przechwycenia_rowne_staremu_detektorowi(haki: str, tmp_path: Path) -> None:
    niezgodne = _niezgodne_pos(haki, tmp_path)
    assert niezgodne == [], (
        f"{haki}: {len(niezgodne)}/53 pozycji POS inne niż stary_detektor, np. {niezgodne[:5]}"
    )


# --------------------------------------------------------------------------------------------------------------------
# s1-s3 (syntetyczne — bez danych z transkryptów)
# --------------------------------------------------------------------------------------------------------------------

WKLEJKA = (
    "TODO: wyzwalacz we wklejce — trzeba zapisać tę rzecz na pewno, bo inaczej zniknie z projektu."
)
ZDANIE_CZLOWIEKA = "Decision: we chose the structural filter over regex stripping because the detector must read only the human."
PRZED_WKLEJKA = (
    "Zerknij proszę na wklejony niżej fragment i powiedz, co o nim sądzisz, bez żadnych zapisów."
)


def _wpis_czlowieka(tekst: str) -> dict[str, Any]:
    return {
        "type": "user",
        "isSidechain": False,
        "origin": {"kind": "human"},
        "promptSource": "sdk",
        "userType": "external",
        "message": {"role": "user", "content": [{"type": "text", "text": tekst}]},
    }


def _s1() -> tuple[dict[str, Any], list[tuple[str, str, float]]]:
    """Wklejka z zamknięciem Z ATRYBUTAMI + zdanie człowieka z wyzwalaczem PO wklejce."""
    tekst = f'<pasted_content id="x">{WKLEJKA}</pasted_content id="x">\n{ZDANIE_CZLOWIEKA}'
    return _wpis_czlowieka(tekst), _wykryj(ZDANIE_CZLOWIEKA, 0.7)


def _s2() -> tuple[dict[str, Any], list[tuple[str, str, float]]]:
    """Wklejka bez zamknięcia (jak w korpusie przyciętym) poprzedzona zdaniem człowieka bez wyzwalacza ⇒ 0."""
    tekst = f'{PRZED_WKLEJKA}\n<pasted_content id="x">{WKLEJKA}'
    return _wpis_czlowieka(tekst), []


def _s3() -> tuple[dict[str, Any], list[tuple[str, str, float]]]:
    """Zamknięcie gołe ``</pasted_content>`` ⇒ jak s1."""
    tekst = f"{PRZED_WKLEJKA}\n<pasted_content>{WKLEJKA}</pasted_content>\n{ZDANIE_CZLOWIEKA}"
    return _wpis_czlowieka(tekst), _wykryj(f"{PRZED_WKLEJKA}\n\n{ZDANIE_CZLOWIEKA}", 0.7)


def _tekst_syntetyka(syntetyk: Callable[[], tuple[dict[str, Any], list[Any]]]) -> str:
    wpis, _ = syntetyk()
    return str(wpis["message"]["content"][0]["text"])


@pytest.mark.parametrize("haki", sorted(HAKI))
@pytest.mark.parametrize("syntetyk", [_s1, _s2, _s3], ids=["s1", "s2", "s3"])
def test_syntetyczne_wklejki(
    haki: str, syntetyk: Callable[[], tuple[dict[str, Any], list[Any]]], tmp_path: Path
) -> None:
    wpis, oczekiwane = syntetyk()
    assert _przez_hak(haki, wpis, tmp_path) == oczekiwane


def test_kontrola_zerowa_syntetykow_stary_tor_widzi_wyzwalacz_wklejki() -> None:
    """Przyrząd s1-s3 coś mierzy: bez filtra wyzwalacz z wklejki JEST przechwycony, a zdanie człowieka też ma wyzwalacz."""
    for syntetyk in (_s1, _s2, _s3):
        wpis, _ = syntetyk()
        assert any(t == "todo" for t, _, _ in _stary_tor(wpis)), syntetyk.__name__
    assert _wykryj(ZDANIE_CZLOWIEKA, 0.7), (
        "zdanie człowieka musi mieć wyzwalacz (s1/s3 bez niego nic nie mierzą)"
    )
    assert _wykryj(WKLEJKA, 0.7), "wklejka musi mieć wyzwalacz"
    assert _wykryj(PRZED_WKLEJKA, 0.7) == [], "część człowieka w s2 nie może mieć wyzwalacza"
    assert len(PRZED_WKLEJKA.strip()) >= MIN_TEKST_HAKA, (
        "s2: zero nie może pochodzić z bramki długości"
    )


# --------------------------------------------------------------------------------------------------------------------
# Mutanty — przyrząd musi zareagować CZERWONYM (asercja: mutant jest wykryty)
# --------------------------------------------------------------------------------------------------------------------


def _s_ok(syntetyk: Callable[[], tuple[dict[str, Any], list[Any]]], tmp_path: Path) -> bool:
    wpis, oczekiwane = syntetyk()
    return all(_przez_hak(h, wpis, tmp_path) == oczekiwane for h in HAKI)


def test_mutant_zamkniecie_wymagane_gole_czerwony(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutant = reguła prototypu U1 (``</pasted_content>`` gołe, bez gałęzi „do końca”): s1 czerwone, na korpusie NEG z przechwyceniem."""
    from surreal_memory.hooks import transcript_filter

    monkeypatch.setattr(
        transcript_filter,
        "BLOK_WKLEJKA",
        re.compile(r"<pasted_content\b[^>]*>.*?</pasted_content>", re.S),
    )
    assert not _s_ok(_s1, tmp_path)
    if _KORPUS is not None:
        assert _porazki_neg("stop", tmp_path), (
            "korpus NEG nie wykrył mutanta „zamknięcie wymagane gołe”"
        )


def test_mutant_bez_obslugi_niezamknietego_czerwony(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutant = zamknięcie z atrybutami, ale bez gałęzi „do końca wypowiedzi”: s2 czerwone, korpus NEG z przechwyceniem."""
    from surreal_memory.hooks import transcript_filter

    monkeypatch.setattr(
        transcript_filter,
        "BLOK_WKLEJKA",
        re.compile(r"<pasted_content\b[^>]*>.*?</pasted_content\b[^>]*>", re.S),
    )
    assert _s_ok(_s1, tmp_path), "ten mutant ma obsługiwać s1 (zamknięcie z atrybutami)"
    assert not _s_ok(_s2, tmp_path)
    if _KORPUS is not None:
        assert _porazki_neg("stop", tmp_path), (
            "korpus NEG nie wykrył mutanta „bez obsługi niezamkniętego”"
        )


def test_mutant_wycinanie_do_konca_bez_zamkniecia_czerwony(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutant A4 = od otwarcia do końca, bez szukania zamknięcia: gubi tekst człowieka PO wklejce ⇒ s1 czerwone."""
    from surreal_memory.hooks import transcript_filter

    monkeypatch.setattr(
        transcript_filter, "BLOK_WKLEJKA", re.compile(r"<pasted_content\b[^>]*>.*\Z", re.S)
    )
    assert not _s_ok(_s1, tmp_path)
    assert _s_ok(_s2, tmp_path), "A4 poprawnie obsługuje s2 — różni się tylko na s1"


def test_mutant_filtr_przepuszcza_api_request_blob_czerwony(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutant = filtr traktuje wpis ``api-request-blob`` jak wypowiedź (czyta jego tekst): korpus NEG czerwony."""
    from surreal_memory.hooks import transcript_filter

    oryginal = transcript_filter.tekst_wypowiedzi_uzytkownika

    def mutant(entry: Any) -> str:
        if isinstance(entry, dict) and entry.get("type") == "api-request-blob":
            return stop._extract_text(entry)
        return oryginal(entry)

    # Haki importują funkcję po nazwie — mutant podstawiamy tam, gdzie jest wołana.
    monkeypatch.setattr(stop, "tekst_wypowiedzi_uzytkownika", mutant)
    monkeypatch.setattr(pre_compact, "tekst_wypowiedzi_uzytkownika", mutant)
    blob = {
        "type": "api-request-blob",
        "message": {"role": "system", "content": [{"type": "text", "text": ZDANIE_CZLOWIEKA}]},
    }
    assert _przez_hak("stop", blob, tmp_path), (
        "mutant przepuszczający bloby API musi dawać przechwycenie"
    )
    if _KORPUS is not None:
        assert len(_porazki_neg("stop", tmp_path)) >= 24, (
            "korpus NEG ma 24 pozycje api-request-blob"
        )


def test_filtr_odrzuca_api_request_blob_bez_mutanta(tmp_path: Path) -> None:
    """Para pozytywna do mutanta powyżej: bez mutanta ten sam blob daje 0 przechwyceń w obu haktach."""
    blob = {
        "type": "api-request-blob",
        "message": {"role": "system", "content": [{"type": "text", "text": ZDANIE_CZLOWIEKA}]},
    }
    assert _przez_hak("stop", blob, tmp_path) == []
    assert _przez_hak("pre_compact", blob, tmp_path) == []
