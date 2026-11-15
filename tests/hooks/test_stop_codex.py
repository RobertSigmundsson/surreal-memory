"""Hook Stop a rollout sesji Codexa (codex-parytet U6, zakres A; PROJEKT §4.4).

Codex przekazuje hookowi Stop `transcript_path` = `$CODEX_HOME/sessions/**/rollout-*.jsonl` (domyślnie `~/.codex`). Hook Stop
przyjmuje dziś tylko ścieżki pod `~/.claude` i `zcode-hook-*` — rollout Codexa byłby pominięty (licznik `stop_transkrypt_pominiety`).
Fikstura `fixtures/rollout-codex-code-mode.jsonl` = KOPIA rolloutu ze stanowiska (`~/dane/codex-parytet/st-codex/cli-CM`, model
`gpt-6-luna` w trybie `code_mode_only`, binarka 0.167.1): zawiera wiersze `developer`, `world_state`, `custom_tool_call*`,
`additional_tools`, `token_usage_record` — czytnik ma je pominąć, a żadnego typu z fikstury nie uznać za nierozpoznany.
Jedyna atrapa: `capture_text` (zapis do bazy) — rejestrator tekstu, żeby test nie pisał do SurrealDB.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest

from surreal_memory.hooks import liczniki, stop

FIKSTURA = Path(__file__).parent / "fixtures" / "rollout-codex-code-mode.jsonl"
TEKST = "Decision: chose the rollout reader for Codex because the Stop hook must capture decisions from every harness."
TEKST_PL = "Zdecydowałem: zapis faktu-znacznika idzie przez smem CLI, bo kanał MCP czeka na klucz w oknie wdrożenia."


@pytest.fixture
def stanowisko(tmp_path, monkeypatch):
    home, dane = tmp_path / "home", tmp_path / "dane"
    for d in (home / ".claude/projects/p", home / ".codex/sessions/2026/11/12", dane):
        d.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("SURREAL_MEMORY_DIR", str(dane))
    zapisane: list[str] = []

    async def rejestrator(
        text, project_name=None, **_kw
    ):  # atrapa zapisu do bazy (nazwana w docstringu modułu)
        zapisane.append(text)
        return {"saved": 1}

    monkeypatch.setattr(stop, "capture_text", rejestrator)
    return {
        "home": home,
        "dane": dane,
        "zapisane": zapisane,
        "sesje": home / ".codex/sessions/2026/11/12",
    }


def _wiersz(typ: str, payload: dict) -> str:
    return (
        json.dumps({"timestamp": "2026-11-12T00:00:00.000Z", "type": typ, "payload": payload})
        + "\n"
    )


def _wiadomosc(rola: str, tekst: str, blok: str | None = None) -> str:
    blok = blok or ("output_text" if rola == "assistant" else "input_text")
    return _wiersz(
        "response_item",
        {"type": "message", "role": rola, "content": [{"type": blok, "text": tekst}]},
    )


def _rollout(sciezka: Path, *wiersze: str, z_fikstura: bool = False) -> Path:
    sciezka.parent.mkdir(parents=True, exist_ok=True)
    if z_fikstura:
        shutil.copyfile(FIKSTURA, sciezka)
        with open(sciezka, "a", encoding="utf-8") as f:
            f.writelines(wiersze)
    else:
        sciezka.write_text("".join(wiersze), encoding="utf-8")
    return sciezka


def _bieg(monkeypatch, sciezka: Path | str) -> None:
    monkeypatch.setattr(
        "sys.stdin",
        io.StringIO(
            json.dumps(
                {
                    "hook_event_name": "Stop",
                    "session_id": "s",
                    "transcript_path": str(sciezka),
                    "cwd": "/tmp",
                    "model": "gpt-6-luna",
                }
            )
        ),
    )
    monkeypatch.setattr("sys.argv", ["smem-hook-stop"])
    kod: object = None
    try:
        stop.main()
    except SystemExit as e:
        kod = e.code
    assert kod in (0, None)  # Stop nigdy nie blokuje zakończenia sesji


def _liczniki(dane: Path) -> dict[str, int]:
    p = dane / liczniki.LICZNIKI_PLIK
    if not p.exists():
        return {}
    wiersze = json.loads(p.read_text())
    return {k: v for doba in wiersze.values() for k, v in doba.items()}


# ---- ścieżka rolloutu: przyjęta i policzona -------------------------------------------------------------------------------


def test_rollout_pod_domyslnym_codex_home_przyjety_i_policzony(stanowisko, monkeypatch):
    p = _rollout(
        stanowisko["sesje"] / "rollout-2026-11-12T03-00-00-aaaa.jsonl", _wiadomosc("user", TEKST)
    )
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_codex": 1}


def test_rollout_pod_codex_home_ze_srodowiska(stanowisko, monkeypatch, tmp_path):
    ch = tmp_path / "stanowisko-home"
    monkeypatch.setenv("CODEX_HOME", str(ch))
    p = _rollout(ch / "sessions/2026/11/12/rollout-x.jsonl", _wiadomosc("user", TEKST))
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_codex": 1}


def test_rollout_w_podkatalogu_kanarek_przyjety(stanowisko, monkeypatch):
    """PROJEKT §6.5: kanarek wdrożenia leży w `sessions/kanarek/` — głębokość katalogów dowolna."""
    p = _rollout(
        stanowisko["home"] / ".codex/sessions/kanarek/rollout-kanarek.jsonl",
        _wiadomosc("user", TEKST),
    )
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_codex": 1}


def test_rollout_prawdziwej_sesji_z_fikstury_z_dopisana_odpowiedzia(stanowisko, monkeypatch):
    """Kopia rolloutu stanowiska + dopisany wiersz użytkownika: zapisany jest TYLKO tekst wiadomości (nic z developer/narzędzi)."""
    p = _rollout(
        stanowisko["sesje"] / "rollout-2026-11-12T03-01-00-bbbb.jsonl",
        _wiadomosc("user", TEKST_PL),
        z_fikstura=True,
    )
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST_PL]
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_codex": 1}


# ---- czytnik ogona ----------------------------------------------------------------------------------------------------


def test_czytnik_pomija_developer_world_state_narzedzia_i_kontekst_srodowiska(stanowisko):
    p = _rollout(stanowisko["sesje"] / "rollout-r.jsonl", z_fikstura=True)
    tekst = stop.read_rollout_tail(str(p))
    # w fikturze: wiadomość użytkownika „wykonaj" (7 zn. ≤ progu 20 jak w CC) i odpowiedź „koniec" (asystent bez znaczników pamięci)
    assert tekst == ""
    p2 = _rollout(
        stanowisko["sesje"] / "rollout-r2.jsonl", _wiadomosc("user", TEKST_PL), z_fikstura=True
    )
    tekst = stop.read_rollout_tail(str(p2))
    assert tekst == TEKST_PL
    for obce in (
        "You are Codex",
        "FORBID-marker",
        "touch ok-marker",
        "<environment_context>",
        "Script completed",
        "skills_instructions",
    ):
        assert obce not in tekst


def test_czytnik_asystent_tylko_ze_znacznikami_pamieci(stanowisko):
    p = _rollout(
        stanowisko["sesje"] / "rollout-a.jsonl",
        _wiadomosc(
            "assistant", "Zwykła odpowiedź bez niczego wartego zapamiętania w tej wiadomości."
        ),
        _wiadomosc(
            "assistant",
            "We decided to keep the dispatcher shared because the Codex adapter reuses the same rules.",
        ),
        _wiadomosc(
            "user", "Użytkownik opisuje swoją preferencję: zawsze polskie komunikaty w logach."
        ),
    )
    tekst = stop.read_rollout_tail(str(p))
    assert "We decided to keep the dispatcher shared" in tekst
    assert "Zwykła odpowiedź" not in tekst
    assert "zawsze polskie komunikaty" in tekst


def test_czytnik_pomija_krotkie_i_role_inne_niz_user_assistant(stanowisko):
    p = _rollout(
        stanowisko["sesje"] / "rollout-k.jsonl",
        _wiadomosc(
            "developer",
            "Instrukcja deweloperska, która nie jest rozmową z użytkownikiem — pomijana.",
        ),
        _wiadomosc("user", "krótka"),
        _wiadomosc(
            "system",
            "Wiadomość roli system — też pomijana, choć długa i pełna treści do zapisania.",
        ),
    )
    assert stop.read_rollout_tail(str(p)) == ""


def test_czytnik_ogon_ogranicza_liczbe_wierszy(stanowisko):
    wiersze = [
        _wiadomosc(
            "user", f"Wiadomość numer {i:04d} użytkownika z dłuższą treścią do zapamiętania."
        )
        for i in range(40)
    ]
    p = _rollout(stanowisko["sesje"] / "rollout-o.jsonl", *wiersze)
    tekst = stop.read_rollout_tail(str(p), max_lines=5)
    assert "numer 0039" in tekst and "numer 0035" in tekst
    assert "numer 0034" not in tekst


def test_czytnik_brak_pliku_i_uszkodzony_wiersz(stanowisko):
    assert stop.read_rollout_tail(str(stanowisko["sesje"] / "nie-ma.jsonl")) == ""
    p = _rollout(
        stanowisko["sesje"] / "rollout-u.jsonl", "to nie jest json\n", _wiadomosc("user", TEKST)
    )
    assert stop.read_rollout_tail(str(p)) == TEKST


# ---- nierozpoznany `type` jest liczony (format niestabilny) ----------------------------------------------------------------


def test_fikstura_prawdziwej_sesji_nie_ma_nierozpoznanych_typow(stanowisko):
    """Kontrola zerowa licznika: kopia prawdziwego rolloutu → 0 nierozpoznanych (inaczej licznik byłby szumem)."""
    p = _rollout(stanowisko["sesje"] / "rollout-z.jsonl", z_fikstura=True)
    stop.read_rollout_tail(str(p))
    assert _liczniki(stanowisko["dane"]) == {}


def test_nierozpoznany_typ_wiersza_i_ladunku_jest_liczony(stanowisko):
    p = _rollout(
        stanowisko["sesje"] / "rollout-n.jsonl",
        _wiersz("nowy_typ_wiersza_codexa", {"x": 1}),
        _wiersz("response_item", {"type": "nowy_typ_ladunku", "role": "user", "content": []}),
        _wiadomosc("user", TEKST),
    )
    assert stop.read_rollout_tail(str(p)) == TEKST  # nierozpoznane nie blokują reszty
    assert _liczniki(stanowisko["dane"]) == {"stop_rollout_nierozpoznane": 2}


def test_nierozpoznane_liczone_tez_przy_biegu_hooka(stanowisko, monkeypatch):
    p = _rollout(
        stanowisko["sesje"] / "rollout-nb.jsonl", _wiersz("nowy_typ", {}), _wiadomosc("user", TEKST)
    )
    _bieg(monkeypatch, p)
    assert _liczniki(stanowisko["dane"]) == {
        "stop_transkrypt_codex": 1,
        "stop_rollout_nierozpoznane": 1,
    }


# ---- ścieżki obce: pominięte i policzone ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "budowa",
    [
        "poza-sessions",
        "zla-nazwa",
        "ucieczka",
        "dowiazanie",
        "inny-codex-home",
        "poza-home",
        "zla-koncowka",
    ],
)
def test_inna_sciezka_pominieta_i_policzona(stanowisko, monkeypatch, tmp_path, budowa):
    home = stanowisko["home"]
    if budowa == "poza-sessions":  # np. historia/konfiguracja w `~/.codex`, nie sesja
        p = _rollout(home / ".codex/log/rollout-x.jsonl", _wiadomosc("user", TEKST))
    elif budowa == "zla-nazwa":
        p = _rollout(stanowisko["sesje"] / "inny-plik.jsonl", _wiadomosc("user", TEKST))
    elif budowa == "zla-koncowka":
        p = _rollout(stanowisko["sesje"] / "rollout-x.json", _wiadomosc("user", TEKST))
    elif budowa == "ucieczka":
        cel = _rollout(home / "rollout-ucieczka.jsonl", _wiadomosc("user", TEKST))
        p = stanowisko["sesje"] / ".." / ".." / ".." / ".." / ".." / "rollout-ucieczka.jsonl"
        assert p.resolve() == cel.resolve()
    elif budowa == "dowiazanie":
        cel = _rollout(home / "sekret" / "rollout-sekret.jsonl", _wiadomosc("user", TEKST))
        p = stanowisko["sesje"] / "rollout-dowiazanie.jsonl"
        p.symlink_to(cel)
    elif budowa == "inny-codex-home":  # CODEX_HOME wskazuje gdzie indziej niż plik
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "inny"))
        p = _rollout(stanowisko["sesje"] / "rollout-x.jsonl", _wiadomosc("user", TEKST))
    else:
        p = _rollout(
            tmp_path / "gdzies" / "sessions" / "rollout-x.jsonl", _wiadomosc("user", TEKST)
        )
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == []
    assert _liczniki(stanowisko["dane"]) == {"stop_transkrypt_pominiety": 1}


def _wpis_czlowieka_cc(tekst: str) -> dict[str, object]:
    """Wpis wypowiedzi człowieka w formacie transkryptu Claude Code (W-3, smem-higiena-autocapture: czytnik CC przepuszcza
    wyłącznie `type="user"` z `message.role="user"`; wpis bez `type` w tym kształcie nie jest wpisem CC). Przed W-3 ten
    plik był wpisem bez `type` — oczekiwanie zmienione Z ZAMYSŁU, wynik testu (zapis `[TEKST]`, brak licznika) ten sam."""
    return {
        "type": "user",
        "isSidechain": False,
        "origin": {"kind": "human"},
        "message": {"role": "user", "content": [{"type": "text", "text": tekst}]},
    }


# ---- ścieżki .claude i ZCode bez zmian (test złotego pliku) ------------------------------------------------------------------


def test_zloty_plik_claude_code_bez_zmian_i_bez_licznika(stanowisko, monkeypatch):
    p = stanowisko["home"] / ".claude/projects/p/sesja.jsonl"
    p.write_text(json.dumps(_wpis_czlowieka_cc(TEKST)) + "\n")
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {}


def test_plik_rollout_pod_claude_idzie_sciezka_claude_code_nie_codexa(stanowisko, monkeypatch):
    """Kolejność gałęzi: `.claude` pierwsza — plik o nazwie rollout-*.jsonl pod `~/.claude` czytany czytnikiem CC, bez licznika Codexa."""
    p = stanowisko["home"] / ".claude/projects/p/rollout-udaje.jsonl"
    p.write_text(json.dumps(_wpis_czlowieka_cc(TEKST)) + "\n")
    _bieg(monkeypatch, p)
    assert stanowisko["zapisane"] == [TEKST]
    assert _liczniki(stanowisko["dane"]) == {}
