"""Filtr transkryptu dla haków Stop i PreCompact: tekst do detektora = WYŁĄCZNIE wypowiedź człowieka.

Haki czytają ogon transkryptu JSONL Claude Code i oddają go detektorowi ``analyze_text_for_memories``. Transkrypt zawiera
jednak znacznie więcej niż rozmowę: wyniki narzędzi, przypomnienia harnessu, bloby żądań API, streszczenia kompakcji,
powiadomienia zadań, wpisy kolejki, kontekst wstrzyknięty przez same haki (``Captured by:``) i prozę agenta. Detektor
łapie w tym wzorce ``TODO:`` / ``Decision:`` i zapisuje je jako „wspomnienia”. Ta funkcja przepuszcza wyłącznie tekst, który
napisał człowiek, i wycina z niego bloki, które człowiek tylko wkleił.

Reguły (specyfikacja: smem-higiena-autocapture, PROJEKT §2.3, Q7):

* format Claude Code (wpis ma klucz ``type``): tylko ``type="user"`` z ``message.role="user"``, bez ``isMeta``,
  ``isCompactSummary``, ``isSidechain``, z ``origin`` brak albo ``origin.kind="human"``, ``promptSource`` ≠ ``"system"``;
  tekst tylko z elementów ``type="text"``;
* format ogólny (brak ``type``): tylko ``role="user"`` na górze wpisu (proza agenta i narzędzi odpada);
* wypowiedź zaczynająca się znacznikiem harnessu jest odrzucana w całości;
* wycinane bloki ``<system-reminder>…</system-reminder>`` i wklejki ``<pasted_content …>`` — zamknięcie wklejki niesie
  atrybuty (``</pasted_content id="…">``), a blok bez zamknięcia (np. przycięty) idzie do końca wypowiedzi;
* linie zawierające ``Captured by:`` są usuwane (autoreferencja: kontekst haka wraca do treści wspomnienia).
"""

from __future__ import annotations

import re
from typing import Any

# Wypowiedź zaczynająca się jednym z tych znaczników nie jest pisana przez człowieka.
ZNACZNIKI_HARNESSU = re.compile(
    r"^\s*<(task-notification|system-reminder|bash-input|bash-stdout|local-command-stdout|local-command-caveat|"
    r"command-name|command-message|scheduled-task|create-pr-command|user-prompt-submit-hook)\b"
)
# Przypomnienia harnessu wewnątrz tekstu człowieka zamykają się gołym znacznikiem.
BLOK_PRZYPOMNIENIA = re.compile(r"<system-reminder\b[^>]*>.*?</system-reminder\b[^>]*>", re.S)
# Wklejka: zamknięcie Z ATRYBUTAMI albo do końca wypowiedzi. Gołe ``</pasted_content>`` nie występuje w transkryptach
# (49/49 zamknięć z atrybutem), więc reguła wymagająca gołego zamknięcia nigdy nie wycinała prawdziwej wklejki.
BLOK_WKLEJKA = re.compile(r"<pasted_content\b[^>]*>.*?(?:</pasted_content\b[^>]*>|\Z)", re.S)
ZNACZNIK_AUTOREFERENCJI = "Captured by:"
# Wpisy o długości <= tej wartości nie niosą treści (jak dotychczasowy filtr „trywialnych” wpisów).
MIN_DLUGOSC_WYPOWIEDZI = 20


def _tekst_z_tresci(tresc: Any) -> str | None:
    """Tekst z ``content``: łańcuch albo lista elementów — brane tylko ``type="text"``. ``None`` = brak tekstu."""
    if isinstance(tresc, str):
        return tresc
    if isinstance(tresc, list):
        return "\n".join(
            str(el.get("text") or "")
            for el in tresc
            if isinstance(el, dict) and el.get("type") == "text"
        )
    return None


def tekst_wypowiedzi_uzytkownika(entry: Any) -> str:
    """Zwraca tekst wypowiedzi człowieka z jednego wpisu transkryptu albo ``""``, gdy wpis nią nie jest."""
    if not isinstance(entry, dict):
        return ""
    if "type" in entry:
        if entry.get("type") != "user":
            return ""
        wiadomosc = entry.get("message")
        if not isinstance(wiadomosc, dict):
            return ""
        if wiadomosc.get("role") != "user":
            return ""
        if entry.get("isMeta") or entry.get("isCompactSummary") or entry.get("isSidechain"):
            return ""
        pochodzenie = entry.get("origin")
        if isinstance(pochodzenie, dict) and pochodzenie.get("kind") not in (None, "human"):
            return ""
        if entry.get("promptSource") == "system":
            return ""
        tresc = wiadomosc.get("content")
    else:
        if entry.get("role") != "user":
            return ""
        tresc = entry.get("content")

    tekst = _tekst_z_tresci(tresc)
    if tekst is None or ZNACZNIKI_HARNESSU.match(tekst):
        return ""
    tekst = BLOK_WKLEJKA.sub("", BLOK_PRZYPOMNIENIA.sub("", tekst))
    tekst = "\n".join(w for w in tekst.splitlines() if ZNACZNIK_AUTOREFERENCJI not in w)
    return tekst if len(tekst) > MIN_DLUGOSC_WYPOWIEDZI else ""
