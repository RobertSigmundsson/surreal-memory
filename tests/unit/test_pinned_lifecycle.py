"""Tests for pinned (KB) memory lifecycle bypass."""

from __future__ import annotations

from dataclasses import replace as dc_replace
from datetime import timedelta

import pytest

from surreal_memory.core.fiber import Fiber
from surreal_memory.core.neuron import Neuron, NeuronState, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.utils.timeutils import utcnow


class TestFiberPinned:
    """Test Fiber pinned field."""

    def test_default_not_pinned(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert fiber.pinned is False

    def test_create_pinned_via_replace(self) -> None:
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        pinned = dc_replace(fiber, pinned=True)
        assert pinned.pinned is True
        # Original unchanged (immutable)
        assert fiber.pinned is False

    def test_pinned_field_exists(self) -> None:
        """Pinned field is part of the dataclass."""
        fiber = Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
        )
        assert hasattr(fiber, "pinned")


class TestPinnedDecayBypass:
    """Test that pinned neurons skip decay."""

    @pytest.fixture()
    def decay_manager(self):
        from surreal_memory.engine.lifecycle import DecayManager

        return DecayManager(decay_rate=0.1, prune_threshold=0.01, min_age_days=0)

    @pytest.fixture()
    def old_time(self):
        return utcnow() - timedelta(days=30)

    async def test_pinned_neurons_skip_decay(self, decay_manager, old_time, tmp_path):
        """Pinned neurons should not have their activation reduced."""
        from surreal_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()

        # Create brain
        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create a neuron that belongs to a pinned fiber
        neuron = Neuron.create(type=NeuronType.ENTITY, content="React hooks")
        await storage.add_neuron(neuron)

        # Create initial neuron state with old activation
        state = NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            access_frequency=1,
            last_activated=old_time,
            decay_rate=0.1,
        )
        await storage.update_neuron_state(state)

        # Create a pinned fiber containing this neuron
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True)
        await storage.add_fiber(pinned_fiber)

        # Apply decay
        await decay_manager.apply_decay(storage)

        # Neuron should NOT be decayed (belongs to pinned fiber)
        updated_state = await storage.get_neuron_state(neuron.id)
        assert updated_state is not None
        assert updated_state.activation_level == 1.0

        await storage.close()

    async def test_unpinned_neurons_still_decay(self, decay_manager, old_time, tmp_path):
        """Non-pinned neurons should decay normally."""
        from surreal_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()

        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        neuron = Neuron.create(type=NeuronType.ENTITY, content="Temporary note")
        await storage.add_neuron(neuron)

        state = NeuronState(
            neuron_id=neuron.id,
            activation_level=1.0,
            access_frequency=1,
            last_activated=old_time,
            decay_rate=0.1,
        )
        await storage.update_neuron_state(state)

        # Create a NON-pinned fiber
        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        await storage.add_fiber(fiber)
        assert fiber.pinned is False

        await decay_manager.apply_decay(storage)

        # Neuron SHOULD be decayed
        updated_state = await storage.get_neuron_state(neuron.id)
        assert updated_state is not None
        assert updated_state.activation_level < 1.0

        await storage.close()


class TestPinnedCompressionBypass:
    """Test that pinned fibers skip compression."""

    async def test_pinned_fiber_skips_compression(self, tmp_path):
        """Pinned fiber should remain at tier 0."""
        from surreal_memory.engine.compression import CompressionEngine
        from surreal_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()

        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create an old neuron + pinned fiber
        old_time = utcnow() - timedelta(days=200)
        neuron = Neuron.create(type=NeuronType.ENTITY, content="Permanent KB content")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True, created_at=old_time)
        await storage.add_fiber(pinned_fiber)

        # Run compression
        engine = CompressionEngine(storage)
        report = await engine.run()

        # Pinned fiber should be skipped
        assert report.fibers_skipped >= 1

        # Verify tier unchanged
        updated = await storage.get_fiber(pinned_fiber.id)
        assert updated is not None
        assert updated.compression_tier == 0

        await storage.close()


class TestPinnedPruneBypass:
    """Test that synapses to pinned neurons skip pruning."""

    async def test_pinned_synapse_skips_prune(self, tmp_path):
        """Synapses connected to pinned neurons should not be pruned."""
        from surreal_memory.engine.consolidation import (
            ConsolidationEngine,
            ConsolidationStrategy,
        )
        from surreal_memory.storage.memory_store import InMemoryStorage

        storage = InMemoryStorage()

        from surreal_memory.core.brain import Brain

        brain = Brain.create(name="test-brain")
        await storage.save_brain(brain)
        storage.set_brain(brain.id)

        # Create two neurons
        n1 = Neuron.create(type=NeuronType.ENTITY, content="Pinned KB concept")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="Related concept")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        # Create a weak, old synapse (normally would be pruned)
        old_time = utcnow() - timedelta(days=30)
        synapse = Synapse.create(
            source_id=n1.id,
            target_id=n2.id,
            type=SynapseType.RELATED_TO,
            weight=0.01,  # Below prune threshold
        )
        synapse = dc_replace(synapse, created_at=old_time, last_activated=old_time)
        await storage.add_synapse(synapse)

        # Pin the fiber containing n1
        fiber = Fiber.create(
            neuron_ids={n1.id},
            synapse_ids={synapse.id},
            anchor_neuron_id=n1.id,
        )
        pinned_fiber = dc_replace(fiber, pinned=True)
        await storage.add_fiber(pinned_fiber)

        # Run pruning
        engine = ConsolidationEngine(storage)
        await engine.run(strategies=[ConsolidationStrategy.PRUNE])

        # Synapse should NOT be pruned (connected to pinned neuron)
        remaining = await storage.get_synapse(synapse.id)
        assert remaining is not None, "Synapse connected to pinned neuron should survive pruning"

        await storage.close()


class TestPinUnpin:
    """Pin/unpin/list against every backend that stores memories.

    These used to run against SQLiteStorage only. That is exactly how the gap
    hid: ``pin_fibers``, ``get_pinned_neuron_ids`` and ``list_pinned_fibers``
    existed on no other backend, every caller probed for them with ``hasattr``,
    and so on SurrealDB — the production backend since 2.0.0 — decay and prune
    silently deleted pinned memories while ``smem_pin`` refused to run at all.
    Parametrizing over backends is what keeps that from being reintroduced.
    """

    async def test_pin_and_unpin_round_trip(self, pin_storage) -> None:
        storage = pin_storage
        neuron = Neuron.create(type=NeuronType.ENTITY, content="test")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
        )
        await storage.add_fiber(fiber)
        assert fiber.pinned is False

        assert await storage.pin_fibers([fiber.id], pinned=True) == 1
        updated = await storage.get_fiber(fiber.id)
        assert updated is not None
        assert updated.pinned is True

        assert await storage.pin_fibers([fiber.id], pinned=False) == 1
        updated = await storage.get_fiber(fiber.id)
        assert updated is not None
        assert updated.pinned is False

    async def test_pin_empty_list_is_a_noop(self, pin_storage) -> None:
        assert await pin_storage.pin_fibers([], pinned=True) == 0

    async def test_pin_unknown_fiber_updates_nothing(self, pin_storage) -> None:
        assert await pin_storage.pin_fibers(["does-not-exist"], pinned=True) == 0

    async def test_repinning_still_counts_the_fiber(self, pin_storage) -> None:
        """The count is "fibers matched", not "flags flipped".

        SQLite reports the UPDATE rowcount, which ignores the prior value, and
        smem_pin echoes it straight back to the user as "pinned N fiber(s)".
        Every backend must agree, or the same call reports differently
        depending on which one is configured.
        """
        storage = pin_storage
        neuron = Neuron.create(type=NeuronType.ENTITY, content="test")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
        await storage.add_fiber(fiber)

        assert await storage.pin_fibers([fiber.id], pinned=True) == 1
        assert await storage.pin_fibers([fiber.id], pinned=True) == 1

        updated = await storage.get_fiber(fiber.id)
        assert updated is not None
        assert updated.pinned is True

    async def test_get_pinned_neuron_ids(self, pin_storage) -> None:
        storage = pin_storage
        n1 = Neuron.create(type=NeuronType.ENTITY, content="pinned")
        n2 = Neuron.create(type=NeuronType.ENTITY, content="not pinned")
        await storage.add_neuron(n1)
        await storage.add_neuron(n2)

        f1 = Fiber.create(neuron_ids={n1.id}, synapse_ids=set(), anchor_neuron_id=n1.id)
        f1 = dc_replace(f1, pinned=True)
        await storage.add_fiber(f1)

        f2 = Fiber.create(neuron_ids={n2.id}, synapse_ids=set(), anchor_neuron_id=n2.id)
        await storage.add_fiber(f2)

        pinned_ids = await storage.get_pinned_neuron_ids()
        assert n1.id in pinned_ids
        assert n2.id not in pinned_ids

    async def test_list_pinned_fibers(self, pin_storage) -> None:
        """Regression: this query selected fibers.type / fibers.priority, columns
        that do not exist, so smem_pin(action="list") raised OperationalError on
        SQLite and was unimplemented everywhere else — broken on every backend.
        """
        storage = pin_storage
        neuron = Neuron.create(type=NeuronType.ENTITY, content="kb fact")
        await storage.add_neuron(neuron)

        fiber = Fiber.create(
            neuron_ids={neuron.id},
            synapse_ids=set(),
            anchor_neuron_id=neuron.id,
            summary="a pinned summary",
        )
        await storage.add_fiber(fiber)
        await storage.pin_fibers([fiber.id], pinned=True)

        listed = await storage.list_pinned_fibers()
        assert len(listed) == 1
        entry = listed[0]
        assert entry["fiber_id"] == fiber.id
        assert entry["summary"] == "a pinned summary"
        # No typed memory attached, so type/priority fall back to documented defaults.
        assert entry["type"] == "unknown"
        assert entry["priority"] == 5
        assert entry["tags"] == []
        assert entry["created_at"]

    async def test_list_pinned_fibers_excludes_unpinned(self, pin_storage) -> None:
        storage = pin_storage
        neuron = Neuron.create(type=NeuronType.ENTITY, content="ordinary")
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
        await storage.add_fiber(fiber)

        assert await storage.list_pinned_fibers() == []


class TestCountPinnedFibers:
    """How many fibers are pinned — the number the dashboard reports.

    Separate from ``list_pinned_fibers`` because that one is capped at
    ``_MAX_LIST_LIMIT`` (200) so ``smem_pin(action="list")`` cannot drag an
    unbounded set through the MCP layer. Measuring the capped list is therefore
    a counter that stops counting at 200 and says nothing about it — and the
    production brain was already holding 320 when this was added, so that
    shortcut would have under-reported by 120 while looking perfectly healthy.
    """

    async def _pinned_fiber(self, storage, content: str, *, pinned: bool):
        neuron = Neuron.create(type=NeuronType.ENTITY, content=content)
        await storage.add_neuron(neuron)
        fiber = Fiber.create(neuron_ids={neuron.id}, synapse_ids=set(), anchor_neuron_id=neuron.id)
        await storage.add_fiber(fiber)
        if pinned:
            await storage.pin_fibers([fiber.id], pinned=True)
        return fiber

    async def test_empty_brain_counts_zero(self, pin_storage) -> None:
        assert await pin_storage.count_pinned_fibers() == 0

    async def test_counts_only_pinned_fibers(self, pin_storage) -> None:
        await self._pinned_fiber(pin_storage, "kb one", pinned=True)
        await self._pinned_fiber(pin_storage, "kb two", pinned=True)
        await self._pinned_fiber(pin_storage, "ordinary", pinned=False)

        assert await pin_storage.count_pinned_fibers() == 2

    async def test_unpinning_lowers_the_count(self, pin_storage) -> None:
        fiber = await self._pinned_fiber(pin_storage, "kb fact", pinned=True)
        assert await pin_storage.count_pinned_fibers() == 1

        await pin_storage.pin_fibers([fiber.id], pinned=False)
        assert await pin_storage.count_pinned_fibers() == 0

    async def test_counts_past_the_list_limit(self, pin_storage) -> None:
        """The regression this method exists for.

        ``len(await list_pinned_fibers(limit=...))`` saturates at
        ``_MAX_LIST_LIMIT``; the count must not. Sized just over the cap so a
        capped implementation reports 200 here instead of 201.
        """
        from surreal_memory.storage.memory_pinning import _MAX_LIST_LIMIT

        over_cap = _MAX_LIST_LIMIT + 1
        for i in range(over_cap):
            await self._pinned_fiber(pin_storage, f"kb fact {i}", pinned=True)

        assert await pin_storage.count_pinned_fibers() == over_cap
        # The list really is capped — this is what makes measuring it wrong.
        assert len(await pin_storage.list_pinned_fibers(limit=over_cap)) == _MAX_LIST_LIMIT

    async def test_count_is_scoped_to_the_current_brain(self, pin_storage) -> None:
        """A pinned fiber in another brain must not be counted."""
        from surreal_memory.core.brain import Brain

        await self._pinned_fiber(pin_storage, "brain one kb", pinned=True)
        assert await pin_storage.count_pinned_fibers() == 1

        other = Brain.create(name="pin-test-brain-other")
        await pin_storage.save_brain(other)
        pin_storage.set_brain(other.id)

        assert await pin_storage.count_pinned_fibers() == 0
