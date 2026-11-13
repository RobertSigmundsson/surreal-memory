"""W-1 live twin of ``test_delete_neuron_id_forms.py`` -- every id form, real SurrealDB.

STANOWISKO: kopia bazy (pula 18680-18689), wlasna przestrzen nazw ``higiena_test``;
skipped without ``SURREALDB_URL``, asserts (does not skip) on a production port.

Fibers store dashed ids, neuron records carry underscores. For each of the four
spellings a caller may pass -- ``<dashes>``, ``<underscores>``,
``neuron:<underscores>``, ``neuron:⟨<dashes>⟩`` -- deleting a MEMBER must take it
off ``fiber.neuron_ids`` and deleting the ANCHOR must reach the fiber too.

MUTANT (must turn this file red): compare raw strings in
``_shed_neuron_from_fibers`` (``m != neuron_id``, ``CONTAINS $nid``): every form but
the dashed one leaves a dead member on the fiber.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from surreal_memory.storage.surrealdb._ids import _to_surreal_id
from tests.unit._higiena_live import (
    add_memory,
    close_live_store,
    open_live_store,
    requires_live,
)

pytestmark = requires_live

_FORM_NAMES = ("dashes", "underscores", "record_underscores", "record_quoted_dashes")


def _spell(dashed: str, form: str) -> str:
    return {
        "dashes": dashed,
        "underscores": _to_surreal_id(dashed),
        "record_underscores": f"neuron:{_to_surreal_id(dashed)}",
        "record_quoted_dashes": f"neuron:⟨{dashed}⟩",
    }[form]


@pytest.fixture
async def store() -> Any:
    st, brain = await open_live_store("w1-id-forms-live")
    try:
        yield st
    finally:
        await close_live_store(st, brain)


@pytest.mark.asyncio
@pytest.mark.parametrize("form", _FORM_NAMES)
async def test_member_leaves_the_fiber_for_every_id_form(store: Any, form: str) -> None:
    anchor, member, fid = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=anchor, member_ids=[member], fiber_id=fid)

    assert await store.delete_neuron(_spell(member, form)) is True

    assert await store.get_neuron(member) is None, "the neuron row itself must be gone"
    fiber = await store.get_fiber(fid)
    assert fiber is not None, "a fiber that lost a member keeps living"
    assert member not in fiber.neuron_ids, f"form {form}: dead member still listed"
    assert fiber.anchor_neuron_id == anchor
    assert "_anchor_deleted" not in fiber.metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("form", _FORM_NAMES)
async def test_anchor_delete_reaches_the_fiber_for_every_id_form(store: Any, form: str) -> None:
    anchor, member, fid = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=anchor, member_ids=[member], fiber_id=fid)

    assert await store.delete_neuron(_spell(anchor, form)) is True

    assert await store.get_neuron(anchor) is None
    # CHANGED FROM INTENT (W-2): the fiber found through the anchor is deleted with its
    # memory (it used to get an `_anchor_deleted` tombstone).
    assert await store.get_fiber(fid) is None, f"form {form}: anchor delete missed the fiber"


@pytest.mark.asyncio
async def test_other_fibers_are_not_touched(store: Any) -> None:
    """Zero control: a neuron listed by no other fiber leaves the neighbour intact."""
    a1, m1, f1 = str(uuid4()), str(uuid4()), str(uuid4())
    a2, m2, f2 = str(uuid4()), str(uuid4()), str(uuid4())
    await add_memory(store, anchor_id=a1, member_ids=[m1], fiber_id=f1)
    await add_memory(store, anchor_id=a2, member_ids=[m2], fiber_id=f2)

    assert await store.delete_neuron(f"neuron:{_to_surreal_id(m1)}") is True

    other = await store.get_fiber(f2)
    assert other is not None and other.neuron_ids == {a2, m2}
    assert not other.metadata.get("_anchor_deleted")
