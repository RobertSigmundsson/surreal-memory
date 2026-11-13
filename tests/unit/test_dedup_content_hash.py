"""W-6 (smem-higiena-autocapture): dedup przy zapisie znajduje DOKŁADNY duplikat po ``content_hash``.

Wada (zmierzona na produkcji): pula kandydatów brała pierwsze słowo treści i ``find_neurons(content_contains=…,
limit ≤ 30)``. Gdy w bazie jest > 30 kotwic z tym samym pierwszym słowem (``TODO:``, ``Decision:``, ścieżka pliku),
kotwica identycznej treści wypada z puli — drugi identyczny zapis tworzy DRUGĄ kotwicę. ``content_hash`` ma indeks
(``idx_neuron_hash``), ale pula z niego nie korzystała.

Poprawka (PROJEKT §2.6): ``find_neurons_by_content_hash`` + kandydaci po ``content_hash`` PRZED pulą po słowie;
``DedupResult.exact`` (odległość 0 i ta sama treść po normalizacji); ``CreateAnchorStep`` przy ``exact`` kotwiczy nową
fibrę na istniejącej kotwicy BEZ aliasu. Prawie-duplikat (odległość 1-7) — jak dotąd alias.

MUTANT (musi zaczerwienić ten plik): ``DedupPipeline._get_candidates`` bez zapytania po ``content_hash`` (stara pula) —
testy ``test_drugi_identyczny_zapis_*`` i ``test_pula_po_slowie_*`` padają. Mutant jest też wykonywany na kodzie
(``sed``) w ``qa/logs/U5-W6-mutanty.log``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.core.brain import Brain, BrainConfig
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.engine.dedup.config import DedupConfig
from surreal_memory.engine.dedup.pipeline import DedupPipeline
from surreal_memory.engine.encoder import MemoryEncoder
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.store import SurrealDBStorage
from surreal_memory.utils.simhash import simhash
from surreal_memory.utils.timeutils import utcnow

MAX_CANDIDATES = 30
N_FILLERS = MAX_CANDIDATES + 5

# (opis, treść). Pierwsze słowo (po obcięciu interpunkcji) jest wspólne z kotwicami-wypełniaczami.
TRESCI: list[tuple[str, str]] = [
    ("todo", "TODO: przepisać skrypt kopii zapasowej tak, żeby sprawdzał sumę kontrolną archiwum"),
    (
        "decision",
        "Decision: konfigurację bramki trzymamy w jednym pliku, bo dwa pliki się rozjeżdżały",
    ),
    ("sciezka", "/home/robert/notatki/plan-okna.md zawiera harmonogram okna konserwacyjnego bazy"),
    (
        "zdanie",
        "Kubernetes nie jest potrzebny na tym hoście, wystarczy podman z jednostkami systemd",
    ),
]


def _first_term(content: str) -> str:
    return content.split()[0]


async def _storage() -> InMemoryStorage:
    storage = InMemoryStorage()
    brain = Brain.create(name="w6", config=BrainConfig(), owner_id="test")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


async def _fillers(storage: InMemoryStorage, term: str) -> None:
    """> 30 kotwic z tym samym pierwszym słowem; ich id sortują się PRZED id kotwicy testowej (UUID)."""
    for i in range(N_FILLERS):
        await storage.add_neuron(
            Neuron(
                id=f"0000fill-0000-0000-0000-{i:012d}",
                type=NeuronType.CONCEPT,
                content=f"{term} wypełniacz numer {i} o zupełnie innej sprawie {i * 7919}",
                metadata={"is_anchor": True},
                content_hash=simhash(
                    f"{term} wypełniacz numer {i} o zupełnie innej sprawie {i * 7919}"
                ),
            )
        )


def _encoder(storage: InMemoryStorage) -> MemoryEncoder:
    brain_cfg = BrainConfig()
    pipeline = DedupPipeline(
        DedupConfig(enabled=True, max_candidates=MAX_CANDIDATES),
        storage,  # type: ignore[arg-type]
    )
    return MemoryEncoder(storage, brain_cfg, dedup_pipeline=pipeline)


async def _neurons_with_hash(storage: InMemoryStorage, content: str) -> list[Neuron]:
    h = simhash(content)
    return [n for n in storage._neurons[storage._get_brain_id()].values() if n.content_hash == h]


# --- find_neurons_by_content_hash (InMemory) -------------------------------------------------------------------------


class TestFindNeuronsByContentHashInMemory:
    async def test_zwraca_tylko_neurony_o_tym_hashu_od_najstarszego(self) -> None:
        storage = await _storage()
        base = utcnow()
        a = Neuron(id="a", type=NeuronType.CONCEPT, content="x", content_hash=42, created_at=base)
        b = Neuron(
            id="b",
            type=NeuronType.CONCEPT,
            content="y",
            content_hash=42,
            created_at=base - timedelta(seconds=5),
        )
        c = Neuron(id="c", type=NeuronType.CONCEPT, content="z", content_hash=43, created_at=base)
        for n in (a, b, c):
            await storage.add_neuron(n)

        found = await storage.find_neurons_by_content_hash(42, limit=10)

        assert [n.id for n in found] == [
            "b",
            "a",
        ]  # kontrola pozytywna: dwa trafienia, najstarszy pierwszy

    async def test_limit_i_brak_trafien(self) -> None:
        storage = await _storage()
        for i in range(5):
            await storage.add_neuron(
                Neuron(id=f"n{i}", type=NeuronType.CONCEPT, content=f"c{i}", content_hash=7)
            )
        assert len(await storage.find_neurons_by_content_hash(7, limit=3)) == 3
        assert await storage.find_neurons_by_content_hash(8, limit=3) == []  # kontrola zerowa

    async def test_izolacja_mozgow(self) -> None:
        storage = await _storage()
        await storage.add_neuron(
            Neuron(id="mine", type=NeuronType.CONCEPT, content="c", content_hash=9)
        )
        other = Brain.create(name="inny", config=BrainConfig(), owner_id="test")
        await storage.save_brain(other)
        storage.set_brain(other.id)
        assert await storage.find_neurons_by_content_hash(9, limit=5) == []


# --- find_neurons_by_content_hash (SurrealDB, zapytanie) -------------------------------------------------------------


class TestFindNeuronsByContentHashSurreal:
    async def test_zapytanie_indeksowane_literal_brain_i_parametry(self) -> None:
        store = SurrealDBStorage.__new__(SurrealDBStorage)
        seen: list[tuple[str, dict[str, Any]]] = []

        async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
            seen.append((sql, params))
            return [
                {
                    "id": "neuron:abc",
                    "type": "CONCEPT",
                    "content": "treść",
                    "metadata": {"is_anchor": True},
                    "content_hash": 7087715606235720604,
                }
            ]

        store._query = query  # type: ignore[method-assign]
        store._get_brain_id = lambda: "default"  # type: ignore[method-assign]

        found = await store.find_neurons_by_content_hash(7087715606235720604, limit=12)

        assert [n.content_hash for n in found] == [7087715606235720604]
        sql, params = seen[0]
        assert (
            'brain_id = "default"' in sql
        )  # literal, nie parametr (planista używa indeksu tylko dla literału)
        assert "content_hash = $ch" in sql
        assert "ORDER BY created_at ASC" in sql
        assert "LIMIT 12" in sql
        assert "SELECT * OMIT embedding_vec FROM neuron" in sql
        assert params == {"ch": 7087715606235720604}

    async def test_limit_jest_liczba_calkowita(self) -> None:
        store = SurrealDBStorage.__new__(SurrealDBStorage)
        seen: list[str] = []

        async def query(sql: str, **params: Any) -> list[dict[str, Any]]:
            seen.append(sql)
            return []

        store._query = query  # type: ignore[method-assign]
        store._get_brain_id = lambda: "default"  # type: ignore[method-assign]
        await store.find_neurons_by_content_hash(1, limit=5)
        assert seen[0].endswith("LIMIT 5")


# --- pula kandydatów i DedupResult.exact -----------------------------------------------------------------------------


def _anchor(content: str, nid: str = "anchor-1", stored_hash: int | None = None) -> Neuron:
    return Neuron(
        id=nid,
        type=NeuronType.CONCEPT,
        content=content,
        metadata={"is_anchor": True},
        content_hash=simhash(content) if stored_hash is None else stored_hash,
    )


class TestCandidatePool:
    async def test_pula_po_slowie_nie_widzi_kotwicy_po_30_wypelniaczach(self) -> None:
        """Kontrola przyrządu: scenariusz naprawdę zabija starą pulę (kotwica poza top-30 po pierwszym słowie)."""
        content = TRESCI[0][1]
        storage = await _storage()
        await _fillers(storage, _first_term(content))
        target = Neuron.create(
            type=NeuronType.CONCEPT,
            content=content,
            metadata={"is_anchor": True},
            content_hash=simhash(content),
        )
        await storage.add_neuron(target)

        pool = await storage.find_neurons(content_contains="todo", limit=MAX_CANDIDATES)

        assert len(pool) == MAX_CANDIDATES
        assert target.id not in {n.id for n in pool}

    async def test_kandydat_po_content_hash_jest_przed_pula_po_slowie(self) -> None:
        content = TRESCI[0][1]
        storage = await _storage()
        await _fillers(storage, _first_term(content))
        target = Neuron.create(
            type=NeuronType.CONCEPT,
            content=content,
            metadata={"is_anchor": True},
            content_hash=simhash(content),
        )
        await storage.add_neuron(target)
        pipeline = DedupPipeline(DedupConfig(enabled=True, max_candidates=MAX_CANDIDATES), storage)  # type: ignore[arg-type]

        candidates = await pipeline._get_candidates(content, simhash(content))

        assert candidates[0].id == target.id
        assert len({n.id for n in candidates}) == len(candidates)  # scalenie bez duplikatów

    async def test_filtr_kandydatow_bez_zmian_alias_i_placeholder_odpadaja(self) -> None:
        from surreal_memory.core.constants import GRAPH_ONLY_PLACEHOLDER

        content = "Powtórzona treść wspomnienia o jednoznacznym brzmieniu"
        h = simhash(content)
        storage = await _storage()
        alias = Neuron(
            id="alias",
            type=NeuronType.CONCEPT,
            content=content,
            metadata={"is_anchor": False, "_dedup_alias_of": "x"},
            content_hash=h,
        )
        tomb = Neuron(
            id="tomb",
            type=NeuronType.CONCEPT,
            content=GRAPH_ONLY_PLACEHOLDER,
            metadata={"is_anchor": True},
            content_hash=h,
        )
        real = Neuron(
            id="real",
            type=NeuronType.CONCEPT,
            content=content,
            metadata={"is_anchor": True},
            content_hash=h,
        )
        for n in (alias, tomb, real):
            await storage.add_neuron(n)
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)  # type: ignore[arg-type]

        assert [n.id for n in await pipeline._get_candidates(content, h)] == ["real"]

    async def test_hash_zero_nie_odpytuje_indeksu(self) -> None:
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        storage.find_neurons_by_content_hash = AsyncMock(return_value=[])
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)

        await pipeline._get_candidates("treść bez hasha", 0)

        storage.find_neurons_by_content_hash.assert_not_called()


class TestExactFlag:
    async def test_dokladny_duplikat_ma_exact(self) -> None:
        content = "Wybraliśmy PostgreSQL do płatności, bo potrzebujemy ACID"
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        storage.find_neurons_by_content_hash = AsyncMock(return_value=[_anchor(content)])
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)

        result = await pipeline.check_duplicate(content)

        assert result.is_duplicate and result.tier == 1 and result.exact is True
        assert result.existing_neuron_id == "anchor-1"

    async def test_normalizacja_wielkosc_liter_i_biale_znaki(self) -> None:
        stored = "Wybraliśmy PostgreSQL do płatności, bo potrzebujemy ACID"
        incoming = "wybraliśmy  postgresql do płatności,\n bo potrzebujemy acid  "
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        # ten sam hash (simhash normalizuje wielkość liter i białe znaki) — treść po normalizacji równa
        storage.find_neurons_by_content_hash = AsyncMock(
            return_value=[_anchor(stored, stored_hash=simhash(incoming))]
        )
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)

        result = await pipeline.check_duplicate(incoming)

        assert result.is_duplicate and result.exact is True

    async def test_kolizja_hasha_inna_tresc_nie_jest_exact(self) -> None:
        stored = "Zupełnie inna treść, której hash został sztucznie podstawiony"
        incoming = "Wybraliśmy PostgreSQL do płatności, bo potrzebujemy ACID"
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        storage.find_neurons_by_content_hash = AsyncMock(
            return_value=[_anchor(stored, stored_hash=simhash(incoming))]
        )
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)

        result = await pipeline.check_duplicate(incoming)

        assert (
            result.is_duplicate and result.exact is False
        )  # odległość 0, ale treść różna ⇒ alias jak dotąd

    async def test_exact_preferuje_dokladny_przed_kolizja(self) -> None:
        incoming = "Wybraliśmy PostgreSQL do płatności, bo potrzebujemy ACID"
        kolizja = _anchor(
            "Inna treść z podstawionym hashem", nid="kolizja", stored_hash=simhash(incoming)
        )
        dokladny = _anchor(incoming, nid="dokladny")
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[])
        storage.find_neurons_by_content_hash = AsyncMock(return_value=[kolizja, dokladny])
        pipeline = DedupPipeline(DedupConfig(enabled=True), storage)

        result = await pipeline.check_duplicate(incoming)

        assert result.exact is True and result.existing_neuron_id == "dokladny"

    async def test_prawie_duplikat_nie_jest_exact(self) -> None:
        stored = "We decided to use PostgreSQL for the database layer of payments and refunds in the billing service"
        incoming = stored + "."
        storage = AsyncMock()
        storage.find_neurons = AsyncMock(return_value=[_anchor(stored)])
        storage.find_neurons_by_content_hash = AsyncMock(return_value=[])
        pipeline = DedupPipeline(DedupConfig(enabled=True, simhash_threshold=10), storage)

        result = await pipeline.check_duplicate(incoming)

        assert result.is_duplicate and result.tier == 1 and result.exact is False


# --- CreateAnchorStep ------------------------------------------------------------------------------------------------


class TestCreateAnchorStepExact:
    @staticmethod
    async def _ctx_and_storage(exact: bool) -> tuple[Any, InMemoryStorage, Neuron]:
        from surreal_memory.engine.pipeline import PipelineContext

        storage = await _storage()
        canonical = Neuron.create(
            type=NeuronType.CONCEPT, content="kanoniczne wspomnienie", metadata={"is_anchor": True}
        )
        await storage.add_neuron(canonical)
        ctx = PipelineContext(
            content="kanoniczne wspomnienie",
            timestamp=utcnow(),
            metadata={},
            tags=set(),
            language="pl",
        )
        ctx.effective_metadata = {"_dedup_reused_anchor": canonical}
        if exact:
            ctx.effective_metadata["_dedup_exact"] = True
        return ctx, storage, canonical

    async def test_exact_kotwiczy_na_istniejacej_kotwicy_bez_aliasu(self) -> None:
        from surreal_memory.engine.pipeline_steps import CreateAnchorStep

        ctx, storage, canonical = await self._ctx_and_storage(exact=True)

        ctx = await CreateAnchorStep().execute(ctx, storage, BrainConfig())

        assert ctx.anchor_neuron is not None and ctx.anchor_neuron.id == canonical.id
        assert ctx.neurons_created == []
        assert ctx.synapses_created == []
        assert len(storage._neurons[storage._get_brain_id()]) == 1
        assert (
            "_dedup_exact" not in ctx.effective_metadata
        )  # znacznik nie wycieka do metadanych fibry

    async def test_bez_exact_alias_jak_dotad(self) -> None:
        from surreal_memory.engine.pipeline_steps import CreateAnchorStep

        ctx, storage, canonical = await self._ctx_and_storage(exact=False)

        ctx = await CreateAnchorStep().execute(ctx, storage, BrainConfig())

        assert ctx.anchor_neuron is not None and ctx.anchor_neuron.id != canonical.id
        assert ctx.anchor_neuron.metadata["_dedup_alias_of"] == canonical.id
        assert ctx.anchor_neuron.metadata["is_anchor"] is False


# --- cały koder: KA.11 -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("nazwa", "content"), TRESCI, ids=[t[0] for t in TRESCI])
async def test_drugi_identyczny_zapis_nie_tworzy_drugiej_kotwicy(nazwa: str, content: str) -> None:
    storage = await _storage()
    await _fillers(storage, _first_term(content))
    encoder = _encoder(storage)

    first = await encoder.encode(content)

    # kontrola pozytywna: po PIERWSZYM zapisie dokładnie 1 neuron o tym hashu jest kotwicą fibry
    after_first = await _neurons_with_hash(storage, content)
    assert [n.id for n in after_first if n.metadata.get("is_anchor")] == [
        first.fiber.anchor_neuron_id
    ]

    second = await encoder.encode(content)

    after_second = await _neurons_with_hash(storage, content)
    anchors_by_role = {first.fiber.anchor_neuron_id, second.fiber.anchor_neuron_id}
    assert len(anchors_by_role) == 1, "dwie fibry — dwie różne kotwice tej samej treści"
    assert [n for n in after_second if n.metadata.get("is_anchor")] == after_first
    assert len(after_second) == len(after_first), (
        "drugi zapis nie dodał żadnego neuronu o tym hashu (alias)"
    )
    assert second.fiber.id != first.fiber.id  # dwa zapisy = dwa wspomnienia (druga fibra istnieje)
    assert not any("_dedup_alias_of" in n.metadata for n in after_second)


async def test_pula_po_slowie_rozne_wypelniacze_nie_zmieniaja_wyniku() -> None:
    """Ten sam zapis dwukrotnie bez wypełniaczy (pula po słowie trafia) także daje jedną kotwicę."""
    content = TRESCI[3][1]
    storage = await _storage()
    encoder = _encoder(storage)

    first = await encoder.encode(content)
    second = await encoder.encode(content)

    assert first.fiber.anchor_neuron_id == second.fiber.anchor_neuron_id
    assert len(await _neurons_with_hash(storage, content)) == 1


async def test_prawie_duplikat_nadal_tworzy_alias() -> None:
    storage = await _storage()
    encoder = _encoder(storage)
    stored = "We decided to use PostgreSQL for the database layer of payments and refunds in the billing service"
    near = stored + "."  # odległość SimHash 2 (zmierzone), treść różna po normalizacji
    first = await encoder.encode(stored)
    second = await encoder.encode(near)

    assert second.fiber.anchor_neuron_id != first.fiber.anchor_neuron_id
    alias = await storage.get_neuron(second.fiber.anchor_neuron_id)
    assert alias is not None
    assert alias.metadata["_dedup_alias_of"] == first.fiber.anchor_neuron_id
    assert alias.metadata["is_anchor"] is False


async def test_trzeci_zapis_tez_kotwiczy_na_pierwszej_kotwicy() -> None:
    content = TRESCI[1][1]
    storage = await _storage()
    await _fillers(storage, _first_term(content))
    encoder = _encoder(storage)

    first = await encoder.encode(content)
    second = await encoder.encode(content)
    third = await encoder.encode(content)

    assert {
        first.fiber.anchor_neuron_id,
        second.fiber.anchor_neuron_id,
        third.fiber.anchor_neuron_id,
    } == {first.fiber.anchor_neuron_id}
    assert len(await _neurons_with_hash(storage, content)) == 1


# --- dwie fibry na jednej kotwicy: retrieval, konsolidacja, zapomnienie (PROJEKT §2.6, OPCJA H2) -----------------------


async def _two_writes(content: str) -> tuple[InMemoryStorage, Any, Any]:
    storage = await _storage()
    await _fillers(storage, _first_term(content))
    encoder = _encoder(storage)
    first = await encoder.encode(content)
    second = await encoder.encode(content)
    assert (
        first.fiber.anchor_neuron_id == second.fiber.anchor_neuron_id
    )  # kontrola: scenariusz „dwie fibry, jedna kotwica”
    return storage, first, second


async def test_fibra_exact_ma_kotwice_wsrod_czlonkow_i_zadnego_nowego_neuronu_kotwicy() -> None:
    content = TRESCI[0][1]
    storage, first, second = await _two_writes(content)

    assert second.fiber.anchor_neuron_id in second.fiber.neuron_ids
    assert second.fiber.anchor_neuron_id not in {n.id for n in second.neurons_created}
    # nowe neurony drugiego zapisu nie są kotwicami (żadnej drugiej kotwicy tej treści)
    assert not any(n.metadata.get("is_anchor") for n in second.neurons_created)
    assert second.fiber.id != first.fiber.id
    # drugi zapis nie dubluje krawędzi kotwica -> cel (ta sama para i typ) na współdzielonej kotwicy
    pary = [
        (s.target_id, s.type)
        for s in storage._synapses[storage._get_brain_id()].values()
        if s.source_id == first.fiber.anchor_neuron_id
    ]
    assert pary, "kontrola: kotwica ma krawędzie wychodzące"
    assert len(pary) == len(set(pary))


async def test_retrieval_toleruje_dwie_fibry_na_jednej_kotwicy() -> None:
    from surreal_memory.engine.retrieval import ReflexPipeline

    content = TRESCI[3][1]
    storage, _first, _second = await _two_writes(content)
    brain = await storage.get_brain(storage._get_brain_id())
    assert brain is not None

    result = await ReflexPipeline(storage, brain.config).query(
        query="Kubernetes podman systemd", max_tokens=500
    )

    assert result.neurons_activated > 0
    assert "podman" in result.context.lower()


async def test_konsolidacja_dedup_i_merge_toleruje_dwie_fibry_na_jednej_kotwicy() -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    content = TRESCI[0][1]
    storage, first, second = await _two_writes(content)

    report = await ConsolidationEngine(storage).run(
        [ConsolidationStrategy.DEDUP, ConsolidationStrategy.MERGE]
    )

    assert report.extra["consolidation_status"] == "completed"
    assert (
        report.extra["dedup_anchors_scanned"] > 0
    )  # kontrola: strategia DEDUP naprawdę przeszła po kotwicach
    assert await storage.get_neuron(first.fiber.anchor_neuron_id) is not None
    assert await storage.get_fiber(first.fiber.id) is not None


async def test_zapomnienie_jednej_fibry_zostawia_wspolna_kotwice_i_druga_fibre() -> None:
    content = TRESCI[1][1]
    storage, first, second = await _two_writes(content)
    anchor = first.fiber.anchor_neuron_id

    await storage.delete_memory_hard(second.fiber.id)

    assert await storage.get_fiber(second.fiber.id) is None
    assert await storage.get_fiber(first.fiber.id) is not None
    assert await storage.get_neuron(anchor) is not None  # wspólny neuron ZOSTAJE


async def test_usuniecie_wspolnej_kotwicy_kasuje_oba_wspomnienia() -> None:
    """Zamierzone (PROJEKT §2.6): ta sama treść, jedna kotwica — usunięcie kotwicy usuwa obie fibry."""
    content = TRESCI[2][1]
    storage, first, second = await _two_writes(content)

    await storage.delete_neuron(first.fiber.anchor_neuron_id)

    assert await storage.get_fiber(first.fiber.id) is None
    assert await storage.get_fiber(second.fiber.id) is None
