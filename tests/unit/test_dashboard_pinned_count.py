"""GET /api/dashboard/stats must report how many fibers are pinned.

The dashboard showed neuron, synapse and fiber counts but never the pinned
(permanent KB) total, so the one number that says how much of the brain is
protected from decay and pruning was unavailable to every HTTP client — an API
tester run against the live dashboard found the production brain's 320 pinned
fibers exposed by no endpoint at all.

These run against a real ``InMemoryStorage``, not a mocked storage: the whole
risk being covered is that the route reports a number the storage layer does
not actually hold, and a mock configured by this test would assert nothing but
its own return value. Only the process-global bits the route reads (config,
brain listing, the multi-second diagnostics pass) are substituted.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.server.dependencies import get_storage
from surreal_memory.server.routes.dashboard_api import router
from surreal_memory.storage.memory_pinning import _MAX_LIST_LIMIT
from surreal_memory.storage.memory_store import InMemoryStorage


def _make_client(storage: object) -> httpx.AsyncClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_storage] = lambda: storage
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost")


async def _brain_storage(name: str) -> InMemoryStorage:
    """A real store whose brain id IS ``name``, so ``storage_for_scope`` reuses it."""
    storage = InMemoryStorage()
    await storage.save_brain(Brain.create(name=name))
    storage.set_brain(name)
    return storage


async def _add_fibers(storage: InMemoryStorage, count: int, *, pinned: bool) -> list[str]:
    ids: list[str] = []
    for i in range(count):
        neuron = Neuron.create(type=NeuronType.ENTITY, content=f"n{i}")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
        await storage.add_fiber(fiber)
        ids.append(fiber.id)
    if pinned and ids:
        await storage.pin_fibers(ids, pinned=True)
    return ids


@pytest.fixture()
def patched_env(monkeypatch: pytest.MonkeyPatch):
    """Point the route's process-global reads at one named brain.

    Diagnostics is stubbed because ``get_stats`` runs a full
    ``DiagnosticsEngine.analyze`` for the active brain (seconds on a real
    brain) purely to fill ``grade``/``purity`` — neither is under test here.
    """

    def _apply(brain_name: str) -> None:
        monkeypatch.setattr(
            "surreal_memory.unified_config.get_config",
            lambda: SimpleNamespace(current_brain=brain_name, storage_backend="memory"),
        )

        async def _brains() -> list[str]:
            return [brain_name]

        monkeypatch.setattr("surreal_memory.unified_config.list_available_brains", _brains)

        class _Diagnostics:
            def __init__(self, _storage: object) -> None: ...

            async def analyze(self, _name: str) -> SimpleNamespace:
                return SimpleNamespace(grade="A", purity_score=90.0)

        monkeypatch.setattr("surreal_memory.engine.diagnostics.DiagnosticsEngine", _Diagnostics)

    return _apply


class TestPinnedCountIsReported:
    async def test_stats_reports_the_pinned_count(self, patched_env) -> None:
        brain = "pinned-count-reports"
        patched_env(brain)
        storage = await _brain_storage(brain)
        await _add_fibers(storage, 3, pinned=True)
        await _add_fibers(storage, 5, pinned=False)

        async with _make_client(storage) as client:
            response = await client.get("/api/dashboard/stats")

        assert response.status_code == 200
        body = response.json()
        # Non-zero on purpose: a counter wired to a constant 0 would pass any
        # assertion made against an empty brain.
        assert body["total_pinned_fibers"] == 3
        assert body["brains"][0]["pinned_fiber_count"] == 3
        # The pinned count is a subset of, not a replacement for, the fiber count.
        assert body["brains"][0]["fiber_count"] == 8
        assert body["total_fibers"] == 8

    async def test_field_is_present_when_nothing_is_pinned(self, patched_env) -> None:
        """Absent-vs-zero: clients must not have to guess which one they got."""
        brain = "pinned-count-empty"
        patched_env(brain)
        storage = await _brain_storage(brain)
        await _add_fibers(storage, 2, pinned=False)

        async with _make_client(storage) as client:
            body = (await client.get("/api/dashboard/stats")).json()

        assert "total_pinned_fibers" in body
        assert body["total_pinned_fibers"] == 0
        assert "pinned_fiber_count" in body["brains"][0]
        assert body["brains"][0]["pinned_fiber_count"] == 0

    async def test_count_is_not_measured_from_the_capped_listing(self, patched_env) -> None:
        """The regression this endpoint field exists to avoid.

        ``list_pinned_fibers`` caps at ``_MAX_LIST_LIMIT`` (200), so a route
        that reported ``len(...)`` would answer 200 for the production brain's
        320 and look entirely healthy doing it. Sized one over the cap.
        """
        brain = "pinned-count-over-cap"
        patched_env(brain)
        storage = await _brain_storage(brain)
        over_cap = _MAX_LIST_LIMIT + 1
        await _add_fibers(storage, over_cap, pinned=True)

        async with _make_client(storage) as client:
            body = (await client.get("/api/dashboard/stats")).json()

        assert body["total_pinned_fibers"] == over_cap
        assert body["brains"][0]["pinned_fiber_count"] == over_cap
        # Guard the guard: if the cap ever goes away this test stops meaning
        # anything, and should be resized rather than silently passing.
        assert len(await storage.list_pinned_fibers(limit=over_cap)) == _MAX_LIST_LIMIT


class TestPinnedCountFailureIsolation:
    async def test_a_failing_count_does_not_zero_the_other_numbers(
        self, patched_env, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One broken metric must not take the whole overview down with it.

        The counts share a handler with the pinned lookup, so an unguarded
        raise would drop the brain to the placeholder summary and report 0
        neurons for a brain that holds them.
        """
        brain = "pinned-count-explodes"
        patched_env(brain)
        storage = await _brain_storage(brain)
        await _add_fibers(storage, 4, pinned=False)

        async def _boom() -> int:
            raise RuntimeError("storage says no")

        storage.count_pinned_fibers = _boom  # type: ignore[method-assign]

        with caplog.at_level("WARNING"):
            async with _make_client(storage) as client:
                response = await client.get("/api/dashboard/stats")

        assert response.status_code == 200
        body = response.json()
        assert body["brains"][0]["fiber_count"] == 4
        assert body["brains"][0]["pinned_fiber_count"] == 0
        # Silence would make a broken counter indistinguishable from an
        # unpinned brain, which is the failure mode worth being loud about.
        assert any("Pinned-fiber count failed" in r.message for r in caplog.records)


class TestPinnedCountInTheOpenApiContract:
    async def test_schema_declares_both_fields(self, patched_env) -> None:
        brain = "pinned-count-schema"
        patched_env(brain)
        storage = await _brain_storage(brain)

        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_storage] = lambda: storage
        schemas = app.openapi()["components"]["schemas"]

        # Each new field must be described exactly like the count it sits
        # beside, so a client reading the spec cannot tell them apart by shape
        # and treat the pinned total as some other kind of number.
        stats_props = schemas["DashboardStats"]["properties"]
        assert stats_props["total_pinned_fibers"] == {
            **stats_props["total_fibers"],
            "title": "Total Pinned Fibers",
        }

        brain_props = schemas["BrainSummary"]["properties"]
        assert brain_props["pinned_fiber_count"] == {
            **brain_props["fiber_count"],
            "title": "Pinned Fiber Count",
        }

        # Both carry a default, so neither is listed as required — and neither
        # is any other count. Adding them cannot break a client that was
        # already parsing this schema.
        assert "required" not in schemas["DashboardStats"]
        assert schemas["BrainSummary"]["required"] == ["id", "name"]
