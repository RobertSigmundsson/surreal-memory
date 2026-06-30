"""Tests for TypedMemory storage functionality."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from surreal_memory.core.brain import Brain
from surreal_memory.core.fiber import Fiber
from surreal_memory.core.memory_types import (
    MemoryType,
    Priority,
    TypedMemory,
)
from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.storage.surrealdb.typed_memory import SurrealDBTypedMemoryMixin
from surreal_memory.utils.timeutils import utcnow


@pytest.fixture
async def storage() -> InMemoryStorage:
    """Create storage with a test brain."""
    storage = InMemoryStorage()
    brain = Brain.create(name="test_brain")
    await storage.save_brain(brain)
    storage.set_brain(brain.id)
    return storage


@pytest.fixture
async def storage_with_fiber(storage: InMemoryStorage) -> tuple[InMemoryStorage, Fiber]:
    """Create storage with a fiber."""
    # Create anchor neuron
    neuron = Neuron.create(
        type=NeuronType.CONCEPT,
        content="Test memory content",
    )
    await storage.add_neuron(neuron)

    # Create fiber
    fiber = Fiber.create(
        neuron_ids={neuron.id},
        synapse_ids=set(),
        anchor_neuron_id=neuron.id,
        summary="Test fiber",
    )
    await storage.add_fiber(fiber)

    return storage, fiber


class TestTypedMemoryStorage:
    """Tests for TypedMemory CRUD operations."""

    @pytest.mark.asyncio
    async def test_add_typed_memory(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test adding a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            priority=Priority.NORMAL,
        )
        result = await storage.add_typed_memory(typed_mem)

        assert result == fiber.id

    @pytest.mark.asyncio
    async def test_add_typed_memory_requires_fiber(self, storage: InMemoryStorage) -> None:
        """Test that adding typed memory requires existing fiber."""
        typed_mem = TypedMemory.create(
            fiber_id="nonexistent-fiber",
            memory_type=MemoryType.FACT,
        )

        with pytest.raises(ValueError, match="does not exist"):
            await storage.add_typed_memory(typed_mem)

    @pytest.mark.asyncio
    async def test_get_typed_memory(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test getting a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
        )
        await storage.add_typed_memory(typed_mem)

        result = await storage.get_typed_memory(fiber.id)

        assert result is not None
        assert result.memory_type == MemoryType.DECISION
        assert result.priority == Priority.HIGH

    @pytest.mark.asyncio
    async def test_get_nonexistent_typed_memory(self, storage: InMemoryStorage) -> None:
        """Test getting a nonexistent typed memory."""
        result = await storage.get_typed_memory("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_find_typed_memories_by_type(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test finding typed memories by type."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
        )
        await storage.add_typed_memory(typed_mem)

        # Find TODOs
        results = await storage.find_typed_memories(memory_type=MemoryType.TODO)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.TODO

        # Find DECISIONs (should be empty)
        results = await storage.find_typed_memories(memory_type=MemoryType.DECISION)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_typed_memories_by_priority(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test finding typed memories by minimum priority."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
            priority=Priority.HIGH,
        )
        await storage.add_typed_memory(typed_mem)

        # Find with min_priority=NORMAL (should find)
        results = await storage.find_typed_memories(min_priority=Priority.NORMAL)
        assert len(results) == 1

        # Find with min_priority=CRITICAL (should not find)
        results = await storage.find_typed_memories(min_priority=Priority.CRITICAL)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_typed_memories_excludes_expired(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test that find excludes expired memories by default."""
        storage, fiber = storage_with_fiber

        # Create expired memory
        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.CONTEXT,
            expires_at=utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        # Default excludes expired
        results = await storage.find_typed_memories()
        assert len(results) == 0

        # Include expired
        results = await storage.find_typed_memories(include_expired=True)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_update_typed_memory(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test updating a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            priority=Priority.LOW,
        )
        await storage.add_typed_memory(typed_mem)

        # Update priority
        updated = typed_mem.with_priority(Priority.CRITICAL)
        await storage.update_typed_memory(updated)

        result = await storage.get_typed_memory(fiber.id)
        assert result is not None
        assert result.priority == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_delete_typed_memory(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test deleting a typed memory."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.FACT,
        )
        await storage.add_typed_memory(typed_mem)

        # Delete
        result = await storage.delete_typed_memory(fiber.id)
        assert result is True

        # Verify deleted
        assert await storage.get_typed_memory(fiber.id) is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_typed_memory(self, storage: InMemoryStorage) -> None:
        """Test deleting nonexistent typed memory."""
        result = await storage.delete_typed_memory("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_expired_memories(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test getting expired memories."""
        storage, fiber = storage_with_fiber

        # Create expired memory
        typed_mem = TypedMemory(
            fiber_id=fiber.id,
            memory_type=MemoryType.TODO,
            expires_at=utcnow() - timedelta(days=1),
        )
        await storage.add_typed_memory(typed_mem)

        expired = await storage.get_expired_memories()
        assert len(expired) == 1
        assert expired[0].fiber_id == fiber.id


class TestTypedMemoryExportImport:
    """Tests for TypedMemory export/import."""

    @pytest.mark.asyncio
    async def test_export_includes_typed_memories(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test that export includes typed memories."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.DECISION,
            priority=Priority.HIGH,
            source="test",
        )
        await storage.add_typed_memory(typed_mem)

        snapshot = await storage.export_brain(storage._current_brain_id)

        # Check metadata contains typed_memories
        assert "typed_memories" in snapshot.metadata
        tm_data = snapshot.metadata["typed_memories"]
        assert len(tm_data) == 1
        assert tm_data[0]["memory_type"] == "decision"
        assert tm_data[0]["priority"] == Priority.HIGH.value

    @pytest.mark.asyncio
    async def test_import_restores_typed_memories(
        self, storage_with_fiber: tuple[InMemoryStorage, Fiber]
    ) -> None:
        """Test that import restores typed memories."""
        storage, fiber = storage_with_fiber

        typed_mem = TypedMemory.create(
            fiber_id=fiber.id,
            memory_type=MemoryType.INSIGHT,
            priority=Priority.NORMAL,
        )
        await storage.add_typed_memory(typed_mem)

        # Export
        snapshot = await storage.export_brain(storage._current_brain_id)

        # Create new storage and import
        new_storage = InMemoryStorage()
        await new_storage.import_brain(snapshot, "imported_brain")
        new_storage.set_brain("imported_brain")

        # Verify typed memory was restored
        result = await new_storage.get_typed_memory(fiber.id)
        assert result is not None
        assert result.memory_type == MemoryType.INSIGHT
        assert result.priority == Priority.NORMAL


class TestCanonicalFiberId:
    """Tests for _to_canonical_fiber_id — the get_typed_memory id normalizer.

    typed_memory.fiber_id is stored as the canonical hyphenated uuid, but callers
    pass several record-id shapes (bare underscored, table-prefixed). The lookup
    must normalize every shape to the canonical form (inverse of _to_surreal_id).
    """

    def test_hyphenated_is_unchanged(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        uuid = "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
        assert _to_canonical_fiber_id(uuid) == uuid

    def test_underscored_maps_to_hyphenated(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        assert (
            _to_canonical_fiber_id("63840762_cdfe_49c5_849e_eb0dc2f6b79f")
            == "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
        )

    def test_fiber_prefixed_is_stripped_and_normalized(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        assert (
            _to_canonical_fiber_id("fiber:63840762_cdfe_49c5_849e_eb0dc2f6b79f")
            == "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
        )

    def test_typed_memory_prefixed_is_stripped_and_normalized(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        assert (
            _to_canonical_fiber_id("typed_memory:63840762_cdfe_49c5_849e_eb0dc2f6b79f")
            == "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
        )

    def test_idempotent(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        once = _to_canonical_fiber_id("fiber:63840762_cdfe_49c5_849e_eb0dc2f6b79f")
        assert _to_canonical_fiber_id(once) == once

    def test_inverse_of_to_surreal_id(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import (
            _to_canonical_fiber_id,
            _to_surreal_id,
        )

        uuid = "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
        # _to_surreal_id builds the record-id form; canonicalizing the (prefixed) record id round-trips.
        assert _to_canonical_fiber_id(f"fiber:{_to_surreal_id(uuid)}") == uuid

    def test_empty_string(self) -> None:
        from surreal_memory.storage.surrealdb.typed_memory import _to_canonical_fiber_id

        assert _to_canonical_fiber_id("") == ""


_SIB_CANON = "63840762-cdfe-49c5-849e-eb0dc2f6b79f"
_SIB_UNDER = "63840762_cdfe_49c5_849e_eb0dc2f6b79f"
_SIB_PREF = f"fiber:{_SIB_UNDER}"


class _CapturingTypedMemoryStore(SurrealDBTypedMemoryMixin):
    """Minimal fake of the SurrealDB mixin: records the fiber_id reaching the SQL
    layer and the record-id write/delete target, so the sibling-method
    normalization (commit f4abd57) is locked at the unit level (the live
    behavioural harness lives outside the test suite)."""

    def __init__(self) -> None:
        self.query_fiber_ids: list[str] = []
        self.merge_targets: list[str] = []
        self.delete_targets: list[str] = []

    def _ensure_conn(self) -> Any:
        return self

    def _get_brain_id(self) -> str:
        return "test-brain"

    async def _query(self, sql: str, **params: Any) -> list[dict[str, Any]]:
        if "fiber_id" in params:
            self.query_fiber_ids.append(params["fiber_id"])
        return [
            {
                "id": "typed_memory:x",
                "fiber_id": params.get("fiber_id", ""),
                "memory_type": "fact",
                "priority": "5",
                "metadata": {},
                "tags": [],
            }
        ]

    async def merge(self, target: str, data: dict[str, Any]) -> None:
        self.merge_targets.append(target)

    async def delete(self, target: str) -> None:
        self.delete_targets.append(target)


class TestSiblingFiberIdNormalization:
    """The sibling write/delete lookups normalize every caller id-form to the
    canonical hyphenated uuid AND build the record-id write/delete target from it
    (commit f4abd57) — the inverse of _to_surreal_id, idempotent on canonical."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("form", [_SIB_CANON, _SIB_UNDER, _SIB_PREF])
    async def test_delete_typed_memory_normalizes(self, form: str) -> None:
        store = _CapturingTypedMemoryStore()
        assert await store.delete_typed_memory(form) is True
        assert store.query_fiber_ids == [_SIB_CANON]
        assert store.delete_targets == [f"typed_memory:{_SIB_UNDER}"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("form", [_SIB_CANON, _SIB_UNDER, _SIB_PREF])
    async def test_update_typed_memory_source_normalizes(self, form: str) -> None:
        store = _CapturingTypedMemoryStore()
        assert await store.update_typed_memory_source(form, "source:abc") is True
        assert store.query_fiber_ids == [_SIB_CANON]
        assert store.merge_targets == [f"typed_memory:{_SIB_UNDER}"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("form", [_SIB_CANON, _SIB_UNDER, _SIB_PREF])
    async def test_promote_memory_type_normalizes(self, form: str) -> None:
        store = _CapturingTypedMemoryStore()
        assert await store.promote_memory_type(form, MemoryType.INSIGHT) is True
        assert store.query_fiber_ids == [_SIB_CANON]
        assert store.merge_targets == [f"typed_memory:{_SIB_UNDER}"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("form", [_SIB_CANON, _SIB_UNDER, _SIB_PREF])
    async def test_get_expiring_memories_for_fibers_normalizes(self, form: str) -> None:
        store = _CapturingTypedMemoryStore()
        await store.get_expiring_memories_for_fibers([form])
        assert store.query_fiber_ids == [_SIB_CANON]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("form", [_SIB_CANON, _SIB_UNDER, _SIB_PREF])
    async def test_update_typed_memory_normalizes(self, form: str) -> None:
        store = _CapturingTypedMemoryStore()
        tm = TypedMemory.create(
            fiber_id=form, memory_type=MemoryType.INSIGHT, priority=Priority.HIGH
        )
        await store.update_typed_memory(tm)
        assert store.query_fiber_ids == [_SIB_CANON]
        assert store.merge_targets == [f"typed_memory:{_SIB_UNDER}"]
