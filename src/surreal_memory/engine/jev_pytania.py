"""Program smem-recall-trzy-warstwy, unit U3 — pytania i progi Jev w JEDNYM
wersjonowanym pliku.

Treść pytań DOSŁOWNIE odtwarza pilot ``~/expertP/typesafe-jev-analiza/pilot/
pilot1_recall_gate.py`` (słownik ``Q``) — nie tłumaczona, nie skracana.
Wersja pytań NIE przechodzi do ledgera bramy (kontrakt D-U3.1: ani nagłówkiem
``x-uruboros-pytania``, ani polem top-level — dodatkowe pole top-level daje
400), więc dataset po stronie ledgera grupuje się po sumie kontrolnej pytań
(:func:`sha256_pytan`), nie po ``PYTANIA_WERSJA`` samej w sobie.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Bump przy KAŻDEJ zmianie treści `PYTANIA` — samo w sobie nie trafia do bramy
# (patrz docstring modułu), ale służy do etykietowania lokalnych logów/QA.
PYTANIA_WERSJA = "recall-w4-v1"

PYTANIA: dict[str, dict[str, Any]] = {
    "odpowiada": {
        "type": "noul",
        "instructions": "Czy `memories` zawiera informację, która odpowiada na `query` "
        "(ten sam temat i konkretny fakt, nie tylko podobne słowa)?",
    },
    "sensowne": {
        "type": "noul",
        "instructions": "Czy `query` jest sensownym pytaniem lub frazą w jakimkolwiek "
        "języku (a nie losowym ciągiem liter)?",
    },
    "ta_domena": {
        "type": "noul",
        "instructions": "Czy `query` dotyczy tej samej dziedziny co `memories` (agenty AI, "
        "pamięć roju, infrastruktura serwerów, pozyskiwanie danych prawnych i "
        "przetargowych, praca Roberta)?",
    },
    "jakosc": {
        "type": "score",
        "instructions": "Jak dobrze `memories` odpowiada na `query`?",
        "criteria": [
            "nic na temat — zapytanie spoza bazy albo bełkot",
            "luźno powiązane, bez odpowiedzi",
            "częściowa odpowiedź",
            "pełna, konkretna odpowiedź",
        ],
    },
}

# 🛑 HIPOTEZA, NIE KRYTERIUM. H4 z pilotu (`typesafe-jev-analiza/pilot/
# pilot1_recall_gate.py`) mierzył `odpowiada` na TEKŚCIE PO SYNTEZIE (ostatecznej
# odpowiedzi recallu, `rc.get("answer")`). Wpięcie U3 w `engine/retrieval.py`
# podaje Jevowi KANDYDATÓW po kroku 4.9 (treści top-K aktywowanych neuronów, PRZED
# rekonstrukcją odpowiedzi) — inny obiekt pomiaru. Próg jest więc PRZELICZANY OD
# NOWA w pomiarze U3 (na kandydatach, nie na syntezie); ta stała jest punktem
# startowym dla tego pomiaru, NIE jest kryterium akceptacji samo w sobie.
PROG_ODPOWIADA_STARTOWY = 0.3


def sha256_pytan() -> str:
    """Suma kontrolna treści `PYTANIA` — klucz grupowania datasetu.

    Powód: wersja pytań (`PYTANIA_WERSJA`) nie przechodzi do ledgera bramy —
    ani nagłówkiem (`x-uruboros-pytania` gubi się przed callbackiem ledgera,
    zmierzone D-U3.1), ani polem top-level (dałoby 400, też zmierzone). Jedyne,
    co ledger widzi, to samo ciało wysłane do Jeva — więc treść pytań musi
    identyfikować się SAMA, przez sumę kontrolną z tego samego słownika, który
    faktycznie poszedł do `questions`.
    """
    return hashlib.sha256(
        json.dumps(PYTANIA, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
