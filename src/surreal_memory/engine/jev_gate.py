"""Program smem-recall-trzy-warstwy, unit U3 — klient bramy Jev (TypeSafe
System One, `/typesafe/v1/systemone`).

🛑 `urllib.request` ze stdlib, NIE `httpx`: `httpx` jest w `pyproject.toml`
wyłącznie w `[project.optional-dependencies]`, a ten kod siedzi w ścieżce
recallu, która musi działać na instalacji bez extras. Wzorzec identyczny do
`engine/reranker.py:213` (`urllib.request.urlopen(req, timeout=...)`,
`# noqa: S310`).

Kontrakt bramy zmierzony na żywo (D-U3.1, LiteLLM `:4001`):

    POST http://127.0.0.1:4001/typesafe/v1/systemone
    Authorization: Bearer <klucz>
    Content-Type: application/json
    {"state": {...}, "model": "jev-1.13.0", "questions": {...}}

| przypadek                                  | wynik |
|---------------------------------------------|-------|
| pin `jev-1.13.0` w ciele                     | 200   |
| brak pola `model`                            | 422   |
| `model` z prefiksem `typesafe/jev-1.13.0`    | 403   |
| dodatkowe pole top-level                     | 400   |

Wersji pytań NIE da się przemycić ani nagłówkiem (`x-uruboros-pytania` gubi
się przed callbackiem ledgera bramy), ani polem top-level (400) — patrz
`engine/jev_pytania.sha256_pytan`.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from surreal_memory.engine.jev_pytania import PYTANIA

_MAX_POWOD_ECHO = 200


def resolve_api_key(env_name: str, file_path: str) -> str | None:
    """Klucz z `os.environ[env_name]`; gdy go brak — pierwsza linia
    `file_path` (`strip()`).

    Fallback na plik istnieje, bo proces smem sprzed rotacji SSOT może nie
    widzieć zmiennej środowiskowej. NIGDY nie czyta SSOT bramy
    (`~/repos/github/uruboros/.secrets/uruboros.env`) samodzielnie — tylko
    `file_path`, dosłownie tak, jak podane w configu.
    """
    wartosc = os.environ.get(env_name, "").strip()
    if wartosc:
        return wartosc
    if not file_path:
        return None
    try:
        with open(file_path, encoding="utf-8") as f:
            linia = f.readline().strip()
    except OSError:
        return None
    return linia or None


ENV_NADPISANIE_KLUCZA = "SURREAL_MEMORY_JEV_API_KEY_ENV"
"""Zmienna PROCESU niosąca NAZWĘ zmiennej z kluczem Jev dla tego kanału (program jev-uzycie-wdrozenie, R2).

Alias klucza wirtualnego to jedyna etykieta źródła wiersza w `decyzja_jev` (na torze pass-through
`wersja_pytan` jest zawsze `none`), więc każdy kanał — hook promptów, MCP, CLI, aparat pomiarowy —
woła Jev własnym kluczem. 🛑 Czytana WYŁĄCZNIE w chwili wywołania, nigdy w `JevConfig`: długowieczne
procesy wołają `UnifiedConfig.save()`, który zapisałby nadpisanie do `config.toml` i przełączył klucz
WSZYSTKIM procesom (także shimowi roju, który ma zostać przy `roj-jev`)."""

_NAZWA_ZMIENNEJ = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


@dataclass(frozen=True)
class KluczJev:
    """Wynik rozwiązania klucza: NAZWA zmiennej (diagnostyka, nie sekret), wartość, skąd."""

    env_name: str | None
    wartosc: str | None
    zrodlo: Literal["config", "nadpisanie", "nadpisanie-nieprawidlowe"]


def rozwiaz_klucz(
    cfg_env: str, cfg_file: str, environ: Mapping[str, str] | None = None
) -> KluczJev:
    """Klucz Jev dla TEGO procesu.

    - brak (albo pusty) `SURREAL_MEMORY_JEV_API_KEY_ENV` ⇒ jak dotąd: `cfg_env`, fallback `cfg_file`;
    - poprawna nazwa ⇒ wartość tej zmiennej i NIC więcej — bez fallbacku na domyślny klucz ani plik
      (cichy powrót do `roj-jev` dałby błędną atrybucję, która wygląda jak poprawna);
    - nazwa niezgodna ze wzorcem (np. wklejona WARTOŚĆ klucza) ⇒ brak klucza; napis nigdy nie jest
      powtarzany w powodzie ani w logu.
    """
    env = os.environ if environ is None else environ
    nadpisanie = env.get(ENV_NADPISANIE_KLUCZA, "").strip()
    if not nadpisanie:
        wartosc = env.get(cfg_env, "").strip() or None
        if wartosc is None and cfg_file:
            wartosc = resolve_api_key("", cfg_file)
        return KluczJev(cfg_env, wartosc, "config")
    if not _NAZWA_ZMIENNEJ.match(nadpisanie):
        return KluczJev(None, None, "nadpisanie-nieprawidlowe")
    return KluczJev(nadpisanie, env.get(nadpisanie, "").strip() or None, "nadpisanie")


def sekrety_do_redakcji(
    redact_env: tuple[str, ...] | list[str],
    aktywny_klucz: str | None,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Wartości do wycięcia z `state` przed wysyłką: nazwy z `redact_env` plus KAŻDA `LITELLM_KEY_*`
    (klucze kanałów powstają w czasie życia programu, a krotka `redact_env` bywa przypięta w
    `config.toml` przez `save()`) plus aktywny klucz niezależnie od nazwy. Najdłuższe najpierw."""
    env = os.environ if environ is None else environ
    wart = {v for name in redact_env if (v := env.get(name))}
    wart |= {v for k, v in env.items() if k.startswith("LITELLM_KEY_") and len(v) >= 8}
    if aktywny_klucz:
        wart.add(aktywny_klucz)
    return sorted(wart, key=len, reverse=True)


@dataclass(frozen=True)
class OdpowiedzJev:
    """Wynik jednego wywołania Jev.

    🛑 Pola liczbowe (`odpowiada`, `sensowne`, `ta_domena`, `jakosc`,
    `jakosc_conf`, `tok`) są `None` przy KAŻDYM statusie innym niż `"OK"`.
    "Nie udało się" nigdy nie może wyglądać jak "zmierzono nisko" (cisza nie
    jest sukcesem).
    """

    status: str  # "OK" | "JEV_NIEDOSTEPNY" | "JEV_ODRZUCIL" (JEV_POMINIETY — patrz niżej)
    odpowiada: float | None
    sensowne: float | None
    ta_domena: float | None
    jakosc: int | None
    jakosc_conf: float | None
    ms: float
    tok: int | None
    zredagowano: int
    powod: str | None


JEV_POMINIETY = "JEV_POMINIETY"
"""Status ustawiany WYŁĄCZNIE przez silnik (`engine/retrieval.py`), nigdy z odpowiedzi bramy:
Jev NIE został zawołany, bo recall nie miał dla niego wejścia (wczesne wyjście na bramce 4.8 albo
brak treści kandydatów po 4.9). To nie jest awaria — awaria (sieć, timeout, brak klucza,
nieparsowalna odpowiedź) to `JEV_NIEDOSTEPNY`, odmowa bramy/Jev to `JEV_ODRZUCIL`. Rozdzielone,
zanim ktoś podepnie alarm pod `JEV_NIEDOSTEPNY` (przegląd 2026-09-24: 30 z 33 „niedostępnych"
było w rzeczywistości „nie wołany"). Pola liczbowe przy tym statusie też są `None`."""


def redaguj(tekst: str, sekrety: list[str]) -> tuple[str, int]:
    """Wycina wartości `sekrety` z `tekst`. Zwraca `(tekst, ile_wystąpień)`.

    Wzorzec domowy: `~/repos/github/uruboros/litellm/callbacks/
    uruboros_surreal.py:367`.

    🛑 Licznik JEST CZĘŚCIĄ WYNIKU, nie ozdobą — `zredagowano > 0` znaczy, że
    KTOŚ wkłada sekrety do `state`. Redakcja to złapała, ale ŹRÓDŁO trzeba
    naprawić; bez licznika ta informacja ginie po cichu.
    """
    ile = 0
    for sekret in sekrety:
        if sekret and sekret in tekst:
            ile += tekst.count(sekret)
            tekst = tekst.replace(sekret, "***")
    return tekst, ile


def _blocking_post(
    url: str, body: bytes, headers: dict[str, str], timeout_s: float
) -> tuple[int, bytes]:
    """POST blokujący — SEAM podmieniany w testach (`monkeypatch.setattr` na
    tę funkcję w module `jev_gate`), żeby testy jednostkowe nigdy nie robiły
    prawdziwego połączenia sieciowego.

    Nigdy nie podnosi dla 4xx/5xx: `HTTPError` jest łapany i jego
    `(code, body)` zwracane jak normalna odpowiedź, więc wołający ma JEDNĄ
    ścieżkę kodu dla każdego wyniku HTTP (200, 4xx, 5xx).
    """
    req = urllib.request.Request(  # noqa: S310 - pinned gateway URL, kontrakt D-U3.1
        url, data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _skroc_powod(kod: int, cialo: bytes) -> str:
    try:
        tekst = cialo.decode("utf-8", errors="replace")
    except Exception:
        tekst = "<nieczytelne ciało>"
    return f"{kod} {tekst[:_MAX_POWOD_ECHO]}"


def _pusty_wynik(status: str, *, ms: float, zredagowano: int, powod: str) -> OdpowiedzJev:
    """Wspólny konstruktor dla statusów innych niż `"OK"` — wymusza, że
    WSZYSTKIE pola liczbowe zostają `None`, nigdy przypadkowo nie zostają
    ustawione przy błędzie."""
    return OdpowiedzJev(
        status=status,
        odpowiada=None,
        sensowne=None,
        ta_domena=None,
        jakosc=None,
        jakosc_conf=None,
        ms=ms,
        tok=None,
        zredagowano=zredagowano,
        powod=powod,
    )


async def zapytaj_jev(
    *,
    query: str,
    memories: str,
    gateway_url: str,
    api_key: str | None,
    model: str,
    timeout_ms: int,
    sekrety: list[str],
    klucz_env: str | None = None,
) -> OdpowiedzJev:
    """Woła bramę Jev z `(query, memories)`, zwraca `OdpowiedzJev`.

    - redaguje `query` i `memories` PRZED zbudowaniem ciała, sumuje liczniki;
    - ciało DOKŁADNIE `{"state", "model", "questions"}` — żadnych dodatkowych
      pól top-level (kontrakt: dałyby 400);
    - `status="OK"` tylko dla 200 z parsowalnymi `answers`;
    - 4xx/5xx ⇒ `"JEV_ODRZUCIL"`, `powod` = kod HTTP + pierwsze 200 znaków
      ciała;
    - timeout/`URLError`/błąd parsowania ⇒ `"JEV_NIEDOSTEPNY"` z nazwanym
      `powod`;
    - brak klucza ⇒ NATYCHMIAST `"JEV_NIEDOSTEPNY"`, `powod="brak klucza"`,
      BEZ wywołania sieciowego (brak sekretu to błąd twardy, nie tryb
      permisywny);
    - do `powod`/logów NIGDY nie trafia wartość klucza.
    """
    if not api_key:
        # NAZWA zmiennej nie jest sekretem — mówi, KTÓREGO klucza brak (kanał bez klucza w SSOT).
        powod = "brak klucza" if klucz_env is None else f"brak klucza: {klucz_env}"
        return _pusty_wynik("JEV_NIEDOSTEPNY", ms=0.0, zredagowano=0, powod=powod)

    query_red, n_query = redaguj(query, sekrety)
    memories_red, n_memories = redaguj(memories, sekrety)
    zredagowano = n_query + n_memories

    body = json.dumps(
        {
            "state": {"query": query_red, "memories": memories_red},
            "model": model,
            "questions": PYTANIA,
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    start = time.perf_counter()
    try:
        kod, cialo = await asyncio.to_thread(
            _blocking_post, gateway_url, body, headers, timeout_ms / 1000.0
        )
    except Exception as exc:  # timeout, URLError, connection refused, ...
        ms = (time.perf_counter() - start) * 1000.0
        return _pusty_wynik(
            "JEV_NIEDOSTEPNY",
            ms=ms,
            zredagowano=zredagowano,
            powod=f"{type(exc).__name__}: {exc}",
        )
    ms = (time.perf_counter() - start) * 1000.0

    if kod >= 400:
        return _pusty_wynik(
            "JEV_ODRZUCIL", ms=ms, zredagowano=zredagowano, powod=_skroc_powod(kod, cialo)
        )

    try:
        dane = json.loads(cialo.decode("utf-8"))
        answers = dane["answers"]
        odpowiada = float(answers["odpowiada"]["noul"])
        sensowne = float(answers["sensowne"]["noul"])
        ta_domena = float(answers["ta_domena"]["noul"])
        jakosc = int(answers["jakosc"]["score"])
        jakosc_conf_raw = answers["jakosc"].get("confidence")
        jakosc_conf = float(jakosc_conf_raw) if jakosc_conf_raw is not None else None
        tok_raw = dane.get("usage", {}).get("input_tokens")
        tok = int(tok_raw) if tok_raw is not None else None
    except Exception as exc:
        return _pusty_wynik(
            "JEV_NIEDOSTEPNY",
            ms=ms,
            zredagowano=zredagowano,
            powod=f"unparsable response: {type(exc).__name__}: {exc}",
        )

    return OdpowiedzJev(
        status="OK",
        odpowiada=odpowiada,
        sensowne=sensowne,
        ta_domena=ta_domena,
        jakosc=jakosc,
        jakosc_conf=jakosc_conf,
        ms=ms,
        tok=tok,
        zredagowano=zredagowano,
        powod=None,
    )
