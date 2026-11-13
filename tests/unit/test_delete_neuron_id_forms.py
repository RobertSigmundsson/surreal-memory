"""W-1 (smem-higiena-autocapture): ``delete_neuron`` honours every id form.

Fibers store neuron ids with dashes (``3872d6a8-...``), neuron RECORDS carry
underscores (``neuron:3872d6a8_...``) and callers hand ``delete_neuron`` any of
four spellings (dashes, underscores, ``neuron:<underscores>``,
``neuron:<dashes>`` in angle brackets). The cascade in
``_shed_neuron_from_fibers`` used ``neuron_ids CONTAINS <raw input>`` and raw
string comparisons, so every spelling except the one the fiber happens to hold
missed the fiber: the neuron was gone, the fiber kept claiming it (invariant I2).

The fake below reproduces SurrealQL's exact-string membership semantics for the
three fiber queries the cascade may issue, so a raw-form lookup misses exactly
as it does on the real database. The live twin of this file is
``test_delete_neuron_id_forms_live.py``.

MUTANT (must turn this file red): in ``_shed_neuron_from_fibers`` compare raw
strings again -- ``m != neuron_id`` / ``anchor == neuron_id`` and
``CONTAINS $nid`` instead of the form set ``$f`` with ``CONTAINSANY``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from surreal_memory.storage.surrealdb.store import SurrealDBStorage

DASHED = "3872d6a8-1234-4abc-8def-0123456789ab"
UNDERSCORED = _to_surreal_id(DASHED)
OTHER = "aaaaaaaa-1111-4222-8333-444444444444"
FORMS = [
    DASHED,
    UNDERSCORED,
    f"neuron:{UNDERSCORED}",
    f"neuron:⟨{DASHED}⟩",
]


class _FakeFiberDb:
    """Stands in for ``_query`` / the SDK connection of a store holding fibers."""

    def __init__(self, fibers: dict[str, dict[str, Any]]) -> None:
        self.fibers = fibers  # fiber id part -> {neuron_ids, anchor_neuron_id, metadata}
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((sql, params))
        if "FROM fiber" not in sql:
            return []
        hits: list[dict[str, Any]] = []
        for fid, row in self.fibers.items():
            members = list(row["neuron_ids"])
            anchor = row["anchor_neuron_id"]
            if "CONTAINSANY" in sql:
                match = any(f in members for f in params["f"])
            elif "anchor_neuron_id IN" in sql:
                match = anchor in params["f"]
            elif "CONTAINS $nid" in sql:
                match = params["nid"] in members
            else:
                match = False
            if match:
                hits.append({"id": f"fiber:{fid}", **row})
        return hits


def _store(fibers: dict[str, dict[str, Any]]) -> tuple[SurrealDBStorage, _FakeFiberDb, AsyncMock]:
    db = _FakeFiberDb(fibers)
    conn = AsyncMock()
    store = SurrealDBStorage.__new__(SurrealDBStorage)
    store._query = db.query  # type: ignore[method-assign]
    store._ensure_conn = lambda: conn  # type: ignore[method-assign]
    store._get_brain_id = lambda: "default"  # type: ignore[method-assign]
    store._record_change_internal = AsyncMock()  # type: ignore[method-assign]
    return store, db, conn


def _member_fiber() -> dict[str, dict[str, Any]]:
    return {
        "f1": {
            "neuron_ids": [OTHER, DASHED],
            "anchor_neuron_id": OTHER,
            "metadata": {},
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("form", FORMS)
async def test_member_is_shed_for_every_id_form(form: str) -> None:
    store, _db, conn = _store(_member_fiber())
    assert await store.delete_neuron(form) is True
    merges = [c for c in conn.merge.await_args_list if str(c.args[0]).startswith("fiber:")]
    assert len(merges) == 1, f"form {form!r}: fiber not found by the cascade"
    assert merges[0].args[1]["neuron_ids"] == [OTHER], "only the deleted member is shed"
    assert "metadata" not in merges[0].args[1], "a member delete must not tombstone the fiber"


@pytest.mark.asyncio
@pytest.mark.parametrize("form", FORMS)
async def test_anchor_is_found_for_every_id_form(form: str) -> None:
    fibers = {"f1": {"neuron_ids": [DASHED, OTHER], "anchor_neuron_id": DASHED, "metadata": {}}}
    store, _db, conn = _store(fibers)
    assert await store.delete_neuron(form) is True
    merges = [c for c in conn.merge.await_args_list if str(c.args[0]).startswith("fiber:")]
    assert len(merges) == 1, f"form {form!r}: anchor fiber not found by the cascade"
    assert merges[0].args[1]["neuron_ids"] == [OTHER]
    assert merges[0].args[1]["metadata"].get("_anchor_deleted")


@pytest.mark.asyncio
async def test_fiber_whose_anchor_is_not_in_member_list_is_still_found() -> None:
    """Anchor missing from ``neuron_ids`` is reachable only via ``anchor_neuron_id IN``."""
    fibers = {"f1": {"neuron_ids": [OTHER], "anchor_neuron_id": DASHED, "metadata": {}}}
    store, _db, conn = _store(fibers)
    assert await store.delete_neuron(f"neuron:{UNDERSCORED}") is True
    merges = [c for c in conn.merge.await_args_list if str(c.args[0]).startswith("fiber:")]
    assert len(merges) == 1
    assert merges[0].args[1]["metadata"].get("_anchor_deleted")


@pytest.mark.asyncio
async def test_query_parameter_carries_the_dashed_form_for_a_record_input() -> None:
    store, db, _conn = _store(_member_fiber())
    await store.delete_neuron(f"neuron:{UNDERSCORED}")
    forms_sent = [p["f"] for sql, p in db.queries if "FROM fiber" in sql]
    assert forms_sent, "the cascade must query fibers with a form set `$f`"
    for sent in forms_sent:
        assert DASHED in sent, "fibers hold dashed ids; the set must contain that form"
        assert UNDERSCORED in sent


@pytest.mark.asyncio
async def test_non_uuid_id_is_matched_in_its_own_spelling() -> None:
    """``_to_public_id`` is lossy (``concept_pin`` -> ``concept-pin``): the id as stored
    must still be matched, and a different id must stay untouched."""
    fibers = {
        "f1": {
            "neuron_ids": [OTHER, "concept_pin"],
            "anchor_neuron_id": OTHER,
            "metadata": {},
        }
    }
    store, _db, conn = _store(fibers)
    assert await store.delete_neuron("neuron:concept_pin") is True
    merges = [c for c in conn.merge.await_args_list if str(c.args[0]).startswith("fiber:")]
    assert len(merges) == 1
    assert merges[0].args[1]["neuron_ids"] == [OTHER]


@pytest.mark.asyncio
async def test_unrelated_fiber_is_left_alone() -> None:
    """Zero control: a fiber that does not list the neuron is not merged."""
    fibers = {"f2": {"neuron_ids": [OTHER], "anchor_neuron_id": OTHER, "metadata": {}}}
    store, _db, conn = _store(fibers)
    assert await store.delete_neuron(DASHED) is True
    assert [c for c in conn.merge.await_args_list if str(c.args[0]).startswith("fiber:")] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("form", FORMS)
async def test_the_neuron_record_itself_is_deleted_for_every_id_form(form: str) -> None:
    """``neuron:⟨<dashes>⟩`` folded blindly addresses ``neuron:_<...>_`` -- no row."""
    store, _db, conn = _store(_member_fiber())
    assert await store.delete_neuron(form) is True
    deleted = [str(c.args[0]) for c in conn.delete.await_args_list]
    assert f"neuron:{UNDERSCORED}" in deleted, f"form {form!r} deleted {deleted}"


def test_live_helper_refuses_production_ports_and_accepts_the_pool() -> None:
    """Instrument control for the live twin: an assertion, not a skip, on 8000/8001."""
    from tests.unit._higiena_live import refuse_production

    for url in ("ws://127.0.0.1:8000/rpc", "http://localhost:8001"):
        with pytest.raises(AssertionError, match="production"):
            refuse_production(url)
    refuse_production("ws://127.0.0.1:18689/rpc")
