"""Decay reads the ``_last_decayed`` bookmark, so each run charges only its own slice.

Every assertion here drives the real ``DecayManager.apply_decay`` against the real
``InMemoryStorage``. Nothing re-implements the elapsed-time arithmetic locally: a test
that computed ``days_elapsed`` itself would keep passing even if lifecycle.py never
read the bookmark at all, which is exactly how the bookmark shipped dead the first time.
Expected weights are written as closed-form Ebbinghaus values so a regression shows up
as a wrong number rather than as agreement with a copy of the bug.
"""

from __future__ import annotations

import math
from dataclasses import replace as dc_replace
from datetime import datetime, timedelta

import pytest

from surreal_memory.core.neuron import Neuron, NeuronType
from surreal_memory.core.synapse import Synapse, SynapseType
from surreal_memory.engine.lifecycle import DecayManager
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.utils.timeutils import utcnow

RATE = 0.1


def _manager(min_age_days: float = 0.0, prune_threshold: float = 0.0) -> DecayManager:
    """Decay manager with the age gate open unless a test is about the gate itself."""
    return DecayManager(decay_rate=RATE, prune_threshold=prune_threshold, min_age_days=min_age_days)


async def _seed(
    now: datetime,
    *,
    idle_days: float = 10.0,
    weight: float = 1.0,
    created_days_ago: float | None = None,
) -> tuple[InMemoryStorage, str]:
    """A brain holding one synapse last used ``idle_days`` ago and never decayed.

    Shaped like every row already in production: a real timestamp pair and no bookmark
    in metadata, so the fallback path is what the first run exercises.
    """
    storage = InMemoryStorage()
    storage.set_brain("decay-bookmark-test")

    source = Neuron.create(type=NeuronType.ENTITY, content="source")
    target = Neuron.create(type=NeuronType.ENTITY, content="target")
    await storage.add_neuron(source)
    await storage.add_neuron(target)

    stamp = now - timedelta(days=idle_days)
    born = now - timedelta(days=idle_days if created_days_ago is None else created_days_ago)
    synapse = dc_replace(
        Synapse.create(
            source_id=source.id,
            target_id=target.id,
            type=SynapseType.RELATED_TO,
            weight=weight,
        ),
        created_at=born,
        last_activated=stamp,
    )
    await storage.add_synapse(synapse)
    return storage, synapse.id


async def _weight(storage: InMemoryStorage, synapse_id: str) -> float:
    stored = await storage.get_synapse(synapse_id)
    assert stored is not None
    return stored.weight


class TestBookmarkIsRead:
    """The elapsed time charged comes from the bookmark, not from creation."""

    async def test_first_run_on_a_pre_bookmark_row_decays_from_last_activated(self) -> None:
        """Existing rows carry no bookmark, so run one must behave exactly as before."""
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=7)

        await _manager().apply_decay(storage, reference_time=now)

        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 7))

    async def test_row_activated_before_it_was_created_keeps_the_old_base(self) -> None:
        """Imports and merges can leave last_activated older than created_at.

        The fallback stays on last_activated for those rows, as today. Taking the newest
        of the three stamps instead would charge from the fresher created_at and fade
        such a memory more slowly — a rate change this fix has no business making.
        """
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10, created_days_ago=0.0)

        await _manager().apply_decay(storage, reference_time=now)

        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 10))

    async def test_run_records_the_reference_time_it_charged_up_to(self) -> None:
        """The bookmark has to name the same instant the charge stopped at."""
        now = utcnow()
        storage, synapse_id = await _seed(now)

        await _manager().apply_decay(storage, reference_time=now)

        stored = await storage.get_synapse(synapse_id)
        assert stored is not None
        assert stored.last_decayed == now

    async def test_second_run_charges_only_the_new_interval(self) -> None:
        """The regression: run two must not re-apply the eleven days run one already did."""
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        manager = _manager()

        await manager.apply_decay(storage, reference_time=now)
        await manager.apply_decay(storage, reference_time=now + timedelta(days=1))

        incremental = math.exp(-RATE * 10) * math.exp(-RATE * 1)
        compounding = math.exp(-RATE * 10) * math.exp(-RATE * 11)  # the pre-fix result
        assert await _weight(storage, synapse_id) == pytest.approx(incremental)
        assert incremental > compounding  # guards the constants above from drifting equal

    async def test_repeat_run_at_the_same_reference_time_changes_nothing(self) -> None:
        """Nothing unbilled left means no weight change and no row rewritten."""
        now = utcnow()
        storage, synapse_id = await _seed(now)
        manager = _manager()

        await manager.apply_decay(storage, reference_time=now)
        settled = await _weight(storage, synapse_id)
        report = await manager.apply_decay(storage, reference_time=now)

        assert report.synapses_decayed == 0
        assert await _weight(storage, synapse_id) == settled

    async def test_use_since_the_last_decay_is_not_charged(self) -> None:
        """A synapse fired after its last decay must not pay for the time it was in use."""
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        manager = _manager()

        await manager.apply_decay(storage, reference_time=now)
        after_first = await _weight(storage, synapse_id)

        fired = await storage.get_synapse(synapse_id)
        assert fired is not None
        await storage.update_synapse(dc_replace(fired, last_activated=now + timedelta(days=2)))
        await manager.apply_decay(storage, reference_time=now + timedelta(days=3))

        # One day (now+3 back to the firing at now+2), not three back to the bookmark.
        assert await _weight(storage, synapse_id) == pytest.approx(after_first * math.exp(-RATE))


class TestTotalDecayIsPreserved:
    """Incremental decay must land on the same curve as one pass from creation."""

    async def test_ten_daily_runs_equal_a_single_ten_day_run(self) -> None:
        """exp(-r*d) chained over a partition of a window telescopes to exp(-r*window).

        Ten runs of one day each and one run of ten days must produce the identical
        weight — otherwise "incremental" would mean "memories fade at a different rate".
        """
        now = utcnow()
        start = now - timedelta(days=10)
        incremental, incremental_id = await _seed(now, idle_days=10)
        one_shot, one_shot_id = await _seed(now, idle_days=10)
        manager = _manager()

        for day in range(1, 11):
            await manager.apply_decay(incremental, reference_time=start + timedelta(days=day))
        await manager.apply_decay(one_shot, reference_time=now)

        assert await _weight(incremental, incremental_id) == pytest.approx(
            await _weight(one_shot, one_shot_id)
        )
        assert await _weight(incremental, incremental_id) == pytest.approx(math.exp(-RATE * 10))

    async def test_uneven_run_cadence_lands_on_the_same_curve(self) -> None:
        """The partition need not be regular — only the window endpoints matter."""
        now = utcnow()
        start = now - timedelta(days=12)
        storage, synapse_id = await _seed(now, idle_days=12)
        manager = _manager()

        for offset in (0.5, 3.0, 3.25, 12.0):
            await manager.apply_decay(storage, reference_time=start + timedelta(days=offset))

        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 12))


class TestAgeGateStillHolds:
    """min_age_days keeps measuring idleness, not time since the last decay pass."""

    async def test_recently_used_synapse_is_still_skipped(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=0.5)

        report = await _manager(min_age_days=1.0).apply_decay(storage, reference_time=now)

        assert report.synapses_decayed == 0
        assert await _weight(storage, synapse_id) == 1.0
        stored = await storage.get_synapse(synapse_id)
        assert stored is not None
        assert stored.last_decayed is None

    async def test_runs_closer_together_than_min_age_days_still_decay(self) -> None:
        """Gating on the bookmark would freeze decay forever on a sub-daily schedule.

        The synapse is thirty days idle; two runs twelve hours apart must both charge,
        because the gate asks how long the memory went unused, not how long ago the
        previous pass ran.
        """
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=30)
        manager = _manager(min_age_days=1.0)

        await manager.apply_decay(storage, reference_time=now)
        after_first = await _weight(storage, synapse_id)
        await manager.apply_decay(storage, reference_time=now + timedelta(hours=12))
        after_second = await _weight(storage, synapse_id)

        assert after_second < after_first
        assert after_second == pytest.approx(after_first * math.exp(-RATE * 0.5))


class TestPrunePath:
    """The zero-out branch bookmarks too, so a pruned row settles instead of churning."""

    async def test_pruned_synapse_is_bookmarked_and_left_alone(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        manager = DecayManager(decay_rate=1.0, prune_threshold=0.5, min_age_days=0.0)

        first = await manager.apply_decay(storage, reference_time=now)
        stored = await storage.get_synapse(synapse_id)
        assert stored is not None
        assert first.synapses_pruned == 1
        assert stored.weight == 0.0
        assert stored.last_decayed == now

        second = await manager.apply_decay(storage, reference_time=now + timedelta(days=1))
        assert second.synapses_decayed == 0


class TestDryRun:
    """A dry run must not leave a bookmark, or it would silently skip the real pass."""

    async def test_dry_run_leaves_no_bookmark_and_no_weight_change(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        manager = _manager()

        report = await manager.apply_decay(storage, reference_time=now, dry_run=True)
        stored = await storage.get_synapse(synapse_id)
        assert stored is not None
        assert report.synapses_decayed == 1
        assert stored.weight == 1.0
        assert stored.last_decayed is None

        await manager.apply_decay(storage, reference_time=now)
        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 10))


# ---------------------------------------------------------------------------
# C2 / P1 — decay epoch (``not_before``): the first pass after a pause must not bill it
# ---------------------------------------------------------------------------

from surreal_memory.core.brain import Brain  # noqa: E402
from surreal_memory.core.neuron import NeuronState  # noqa: E402
from surreal_memory.engine.lifecycle import DecayRefusedError  # noqa: E402


async def _put_neuron_state(
    storage: InMemoryStorage,
    now: datetime,
    *,
    idle_days: float,
    level: float = 1.0,
    rate: float = RATE,
    activated: bool = True,
) -> str:
    neuron = Neuron.create(type=NeuronType.ENTITY, content="epoch-neuron")
    await storage.add_neuron(neuron)
    stamp = now - timedelta(days=idle_days)
    await storage.update_neuron_state(
        NeuronState(
            neuron_id=neuron.id,
            activation_level=level,
            decay_rate=rate,
            last_activated=stamp if activated else None,
            created_at=stamp,
        )
    )
    return neuron.id


class TestDecayEpoch:
    async def test_a_no_epoch_is_identical_to_today(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        report = await _manager().apply_decay(storage, reference_time=now)
        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 10))
        assert report.not_before is None

    async def test_b_newer_epoch_charges_from_the_epoch(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        epoch = now - timedelta(days=2)
        manager = DecayManager(
            decay_rate=RATE, prune_threshold=0.0, min_age_days=0.0, not_before=epoch
        )
        await manager.apply_decay(storage, reference_time=now)
        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 2))

    async def test_c_epoch_equal_reference_time_skips_everything(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        nid = await _put_neuron_state(storage, now, idle_days=10)
        manager = DecayManager(
            decay_rate=RATE, prune_threshold=0.0, min_age_days=0.0, not_before=now
        )
        report = await manager.apply_decay(storage, reference_time=now)
        assert report.synapses_decayed == 0
        assert report.synapses_skipped_bookmark == 1
        assert report.neurons_decayed == 0
        assert await _weight(storage, synapse_id) == 1.0
        state = await storage.get_neuron_state(nid)
        assert state is not None and state.activation_level == 1.0

    async def test_d_bookmark_newer_than_epoch_wins(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        epoch = now - timedelta(days=5)
        manager = DecayManager(
            decay_rate=RATE, prune_threshold=0.0, min_age_days=0.0, not_before=epoch
        )
        await manager.apply_decay(
            storage, reference_time=now - timedelta(days=3)
        )  # bookmark = now-3d
        after_first = await _weight(storage, synapse_id)
        await manager.apply_decay(storage, reference_time=now)
        # Second run charges 3 days (bookmark now-3d), not 5 (epoch).
        assert await _weight(storage, synapse_id) == pytest.approx(
            after_first * math.exp(-RATE * 3)
        )

    async def test_e_use_after_epoch_charges_from_last_activated(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=1)  # used 1 day ago, after the epoch
        epoch = now - timedelta(days=5)
        manager = DecayManager(
            decay_rate=RATE, prune_threshold=0.0, min_age_days=0.0, not_before=epoch
        )
        await manager.apply_decay(storage, reference_time=now)
        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 1))

    async def test_f_neurons_follow_the_epoch(self) -> None:
        now = utcnow()
        storage = InMemoryStorage()
        storage.set_brain("epoch-neurons")
        nid = await _put_neuron_state(storage, now, idle_days=30)
        old = await _put_neuron_state(storage, now, idle_days=30)
        epoch = now - timedelta(days=4)
        manager = DecayManager(
            decay_rate=RATE, prune_threshold=0.0, min_age_days=1.0, not_before=epoch
        )
        await manager.apply_decay(storage, reference_time=now)
        state = await storage.get_neuron_state(nid)
        assert state is not None
        assert state.activation_level == pytest.approx(math.exp(-RATE * 4))
        # same epoch, second neuron identical
        other = await storage.get_neuron_state(old)
        assert other is not None
        assert other.activation_level == pytest.approx(math.exp(-RATE * 4))

    async def test_g_reads_brain_metadata_and_parameter_wins(self) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        brain_id = storage.brain_id
        assert brain_id is not None
        brain = Brain(
            id=brain_id,
            name=brain_id,
            metadata={"decay_not_before": (now - timedelta(days=2)).isoformat()},
        )
        await storage.save_brain(brain)

        report = await _manager().apply_decay(storage, reference_time=now)
        assert await _weight(storage, synapse_id) == pytest.approx(math.exp(-RATE * 2))
        assert report.not_before == now - timedelta(days=2)

        # Parameter overrides the brain value (and a fresh synapse isolates the check).
        storage2, sid2 = await _seed(now, idle_days=10)
        await storage2.save_brain(
            Brain(
                id=storage2.brain_id or "",
                name=storage2.brain_id or "",
                metadata={"decay_not_before": (now - timedelta(days=2)).isoformat()},
            )
        )
        manager = DecayManager(
            decay_rate=RATE,
            prune_threshold=0.0,
            min_age_days=0.0,
            not_before=now - timedelta(days=6),
        )
        await manager.apply_decay(storage2, reference_time=now)
        assert await _weight(storage2, sid2) == pytest.approx(math.exp(-RATE * 6))

    @pytest.mark.parametrize("bad", ["not-a-date", 12345, "", ["2026-11-03"]])
    async def test_h_bad_value_refuses_with_zero_writes(self, bad: object) -> None:
        now = utcnow()
        storage, synapse_id = await _seed(now, idle_days=10)
        nid = await _put_neuron_state(storage, now, idle_days=10)
        brain_id = storage.brain_id or ""
        await storage.save_brain(
            Brain(id=brain_id, name=brain_id, metadata={"decay_not_before": bad})
        )

        for dry in (False, True):
            with pytest.raises(ValueError):
                await _manager().apply_decay(storage, reference_time=now, dry_run=dry)
        with pytest.raises(DecayRefusedError):
            await _manager().apply_decay(storage, reference_time=now)

        assert await _weight(storage, synapse_id) == 1.0
        state = await storage.get_neuron_state(nid)
        assert state is not None and state.activation_level == 1.0


class TestDecayEpochCli:
    """`smem decay --set-not-before` stores a verified epoch and runs no pass."""

    def _wire(self, monkeypatch: pytest.MonkeyPatch, storage: InMemoryStorage) -> None:
        import asyncio

        from surreal_memory.cli.commands import tools

        async def _fake_storage(*_a: object, **_k: object) -> InMemoryStorage:
            return storage

        monkeypatch.setattr(tools, "get_storage", _fake_storage)
        monkeypatch.setattr(tools, "resolve_brain", lambda *_a, **_k: storage.brain_id)
        monkeypatch.setattr(tools, "run_async", lambda coro: asyncio.run(coro))

    def _call(self, **kw: object) -> None:
        from surreal_memory.cli.commands import tools

        args: dict[str, object] = {
            "brain": None,
            "dry_run": False,
            "prune_threshold": 0.01,
            "not_before": None,
            "set_not_before": None,
        }
        args.update(kw)
        tools.decay(**args)  # type: ignore[arg-type]

    def test_set_not_before_now_is_stored_verified_and_runs_no_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        storage = InMemoryStorage()
        storage.set_brain("cli-epoch")
        asyncio.run(storage.save_brain(Brain(id="cli-epoch", name="cli-epoch")))
        asyncio.run(_seed_into(storage))
        self._wire(monkeypatch, storage)

        self._call(set_not_before="now")

        brain = asyncio.run(storage.get_brain("cli-epoch"))
        assert brain is not None
        stored = brain.metadata["decay_not_before"]
        assert isinstance(stored, str)
        assert abs((utcnow() - datetime.fromisoformat(stored)).total_seconds()) < 60
        assert not asyncio.run(storage.get_all_synapses())[0].last_decayed  # no pass ran

    def test_bad_set_not_before_exits_nonzero_and_stores_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import typer

        storage = InMemoryStorage()
        storage.set_brain("cli-epoch")
        asyncio.run(storage.save_brain(Brain(id="cli-epoch", name="cli-epoch")))
        self._wire(monkeypatch, storage)

        with pytest.raises(typer.Exit) as excinfo:
            self._call(set_not_before="yesterday-ish")
        assert excinfo.value.exit_code != 0
        brain = asyncio.run(storage.get_brain("cli-epoch"))
        assert brain is not None and "decay_not_before" not in brain.metadata

    def test_stored_bad_epoch_makes_the_pass_exit_nonzero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        import typer

        storage = InMemoryStorage()
        storage.set_brain("cli-epoch")
        asyncio.run(
            storage.save_brain(
                Brain(id="cli-epoch", name="cli-epoch", metadata={"decay_not_before": "junk"})
            )
        )
        self._wire(monkeypatch, storage)
        with pytest.raises(typer.Exit) as excinfo:
            self._call()
        assert excinfo.value.exit_code != 0


async def _seed_into(storage: InMemoryStorage) -> None:
    source = Neuron.create(type=NeuronType.ENTITY, content="s")
    target = Neuron.create(type=NeuronType.ENTITY, content="t")
    await storage.add_neuron(source)
    await storage.add_neuron(target)
    await storage.add_synapse(
        dc_replace(
            Synapse.create(source_id=source.id, target_id=target.id, type=SynapseType.RELATED_TO),
            last_activated=utcnow() - timedelta(days=9),
        )
    )
