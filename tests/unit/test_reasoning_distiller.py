"""Tests for engine/reasoning_distiller.py — heuristic distillation.

Runs against InMemoryStorage with the embedding provider forced OFF (the
`no_embedder` fixture), exercising the fail-soft keyword-classification +
move-set-Jaccard-clustering path (D5). Covers move segmentation, keyword
classification, clustering/pattern materialization, idempotency, mark-processed,
coverage math, and the LEARN_REASONING consolidation handler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from surreal_memory.engine.reasoning_distiller import (
    DistillResult,
    _classify_by_keywords,
    _cluster,
    distill_reasoning_patterns,
    reasoning_coverage,
    segment_moves,
)
from surreal_memory.storage.memory_store import InMemoryStorage
from surreal_memory.unified_config import ReasoningTrainingConfig, UnifiedConfig

BRAIN = "b1"


@pytest.fixture
def no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the fail-soft (no embedding provider) path — keyword + Jaccard."""
    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._get_embedder", lambda *_a, **_k: None
    )


def _ucfg(tmp_path: Path, **rt_kw: object) -> UnifiedConfig:
    base: dict[str, object] = {
        "mining_enabled": True,
        "min_cluster_support": 2,
        "min_patterns_per_category": 1,
        "min_confidence": 0.2,
        # Generous per-model targets so the existing fixtures (which create at
        # most a couple of patterns) distill freely; individual tests override.
        "pattern_targets": {"claude-fable-5": 100, "claude-sonnet-5": 100},
    }
    base.update(rt_kw)
    return UnifiedConfig(
        data_dir=tmp_path / ".surrealmemory",
        current_brain="default",
        reasoning_training=ReasoningTrainingConfig(**base),  # type: ignore[arg-type]
    )


async def _seed(storage: InMemoryStorage, model: str, contents: list[str]) -> None:
    traces = [
        {
            "trace_hash": f"{model}-{i}",
            "model": model,
            "session_id": "s",
            "project": "p",
            "task_context": "",
            "content": c,
            "content_chars": len(c),
            "created_at": "2026-03-01T00:00:00",
        }
        for i, c in enumerate(contents)
    ]
    await storage.insert_reasoning_traces(BRAIN, traces)


# 3 debugging traces sharing the move-set {restate-goal, gather-evidence, verify}.
_DEBUG_TRACES = [
    "I need to fix this. Let me check the error traceback. I verify the bug is gone.",
    "I need to look at this. Let me check the exception. Verify the crash is fixed.",
    "I need to resolve it. Let me check the failing traceback. I verify the error.",
]

# Three 2-trace clusters with DISTINCT move-sets (→ distinct signatures), ordered
# so a small per-model batch fetches one cluster at a time.
_THREE_CLUSTERS = [
    "I need to fix the bug. Let me check the error traceback. I verify it.",
    "I need to fix the bug. Let me check the error traceback. I verify it.",
    "What if the edge case is null? Actually, wait, let me reconsider the boundary.",
    "What if the edge case is null? Actually, wait, let me reconsider the boundary.",
    "My plan: step 1 decompose the problem. Compare option A versus option B.",
    "My plan: step 1 decompose the problem. Compare option A versus option B.",
]


def test_segment_moves_detects_closed_vocab() -> None:
    moves = segment_moves("I need to fix it. Let me check the logs. Then I verify the result.")
    assert "restate-goal" in moves
    assert "gather-evidence" in moves
    assert "verify" in moves
    assert segment_moves("") == []


def test_classify_by_keywords_and_other() -> None:
    cats = ReasoningTrainingConfig().categories
    assert _classify_by_keywords("hit an error with a traceback", cats) == "debugging"
    assert _classify_by_keywords("let me refactor and clean up this module", cats) == "refactoring"
    assert _classify_by_keywords("zzz qwerty plugh foobar nothing here", cats) == "other"


async def test_distill_creates_pattern_fiber(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert isinstance(result, DistillResult)
    assert result.patterns_learned == 1
    assert result.traces_processed == 3

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 1
    assert fibers[0].pinned is True  # KB-grade: lifecycle must never decay/prune patterns
    md = fibers[0].metadata
    assert md["_reasoning_pattern"] is True
    assert md["_source_model"] == "claude-fable-5"
    assert md["_reasoning_category"] == "debugging"
    assert md["_reasoning_frequency"] == 3
    assert md["_reasoning_confidence"] == 1.0
    assert md["_reasoning_signature"]
    # CONCEPT neuron for the category + EFFECTIVE_FOR synapse exist.
    cat_neuron = await storage.find_neurons(content_exact="reasoning_category:debugging", limit=1)
    assert cat_neuron


async def test_distill_honors_mining_models_filter(tmp_path: Path, no_embedder: None) -> None:
    # Pre-existing unprocessed traces for two models; a mining_models glob must
    # restrict distillation to the matching model only (HIGH-2 regression: the
    # POST /mine models= override must actually gate distillation, not just ingest).
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await _seed(storage, "claude-sonnet-5", _DEBUG_TRACES)

    cfg = _ucfg(tmp_path, mining_models=("claude-fable-5",))
    await distill_reasoning_patterns(storage, BRAIN, cfg)

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert {f.metadata["_source_model"] for f in fibers} == {"claude-fable-5"}
    # sonnet's traces are untouched (not consumed by the restricted run).
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, limit=100)
    assert {t["model"] for t in remaining} == {"claude-sonnet-5"}


async def test_distill_is_idempotent(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    first = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert first.patterns_learned == 1
    # Second pass: traces already processed → nothing new.
    second = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert second.patterns_learned == 0
    assert second.traces_processed == 0
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 1


async def test_distill_marks_traces_processed(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert remaining == []


async def test_below_min_cluster_support_no_pattern(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES[:1])  # only 1 trace, support=2
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert result.patterns_learned == 0
    # The trace is still marked processed (considered, just not clustered).
    assert result.traces_processed == 1


async def test_other_category_not_distilled(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", ["qwerty plugh foobar", "zzz nothing", "blah blah"])
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert result.patterns_learned == 0


async def test_reasoning_coverage_math(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path)
    await distill_reasoning_patterns(storage, BRAIN, cfg)

    cov = await reasoning_coverage(storage, "claude-fable-5", cfg)
    assert cov["by_category"]["debugging"] == 1
    assert cov["covered"]["debugging"] is True
    assert cov["coverage_percent"] == round(1 / len(cfg.reasoning_training.categories) * 100, 1)
    # A different model has no coverage.
    cov_other = await reasoning_coverage(storage, "claude-sonnet-5", cfg)
    assert cov_other["coverage_percent"] == 0.0


async def test_coverage_respects_min_patterns_bar(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    # Require 5 patterns/category — the single pattern is below the bar.
    strict = _ucfg(tmp_path, min_patterns_per_category=5)
    cov = await reasoning_coverage(storage, "claude-fable-5", strict)
    assert cov["by_category"]["debugging"] == 1  # still counted
    assert cov["covered"]["debugging"] is False  # but below the coverage bar
    assert cov["coverage_percent"] == 0.0


class _FakeEmbedder:
    """Deterministic one-hot-by-category embedder for the embedding code path."""

    async def embed(self, text: str) -> list[float]:
        return _fake_vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_fake_vec(t) for t in texts]


def _fake_vec(text: str) -> list[float]:
    cats = ReasoningTrainingConfig().categories
    cat = _classify_by_keywords(text, cats)
    vec = [0.0] * (len(cats) + 1)
    vec[cats.index(cat) if cat in cats else len(cats)] = 1.0
    return vec


async def test_distill_uses_embedding_path(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # Explicit embedder -> exercises _seed_centroids / _embed_texts /
    # _classify_by_vector / cosine clustering / medoid (no live provider needed).
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path), embedder=_FakeEmbedder()
    )
    assert result.patterns_learned == 1
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert fibers[0].metadata["_reasoning_category"] == "debugging"


def test_backtrack_and_hypothesize_moves_match() -> None:
    moves = segment_moves("Actually, I was wrong. My hypothesis is that the bug is here.")
    assert "backtrack" in moves
    assert "hypothesize" in moves


class _RaisingEmbedder:
    async def embed(self, text: str) -> list[float]:
        raise RuntimeError("provider down")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("provider down")


async def test_fail_soft_when_embedder_raises(tmp_path: Path) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # An embedder that raises must not break distillation — it falls back to
    # keyword classification + move-set clustering (D5).
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path), embedder=_RaisingEmbedder()
    )
    assert result.patterns_learned == 1


async def test_cap_marks_only_consumed_traces(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    contents = (
        ["I need to fix the error bug. Let me check the traceback."] * 2
        + ["I need to refactor and clean up. Let me check the code."] * 2
        + ["I need to design the module architecture boundary. Let me check it."] * 2
        + ["I need to read the documentation docs. Let me check them."] * 2
    )
    await _seed(storage, "claude-fable-5", contents)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 2})
    )
    assert result.patterns_learned == 2
    # Only the 2 fully-clustered categories are marked processed; the 2 categories
    # the budget never reached keep their traces for the next run (no silent loss).
    assert result.traces_processed == 4
    remaining = await storage.get_unprocessed_reasoning_traces(
        BRAIN, model="claude-fable-5", limit=100
    )
    assert len(remaining) == 4


async def test_distinct_clusters_same_top_moves_not_dropped(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    a = "I need to fix the bug. Let me check the error. I verify it."
    b = (
        "I need to fix the bug. Let me check the error. I verify. "
        "What if the edge case? Actually, wait. Instead of option 1."
    )
    await _seed(storage, "claude-fable-5", [a, a, b, b])
    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    # Two distinct clusters (Jaccard 0.5) sharing the same top-3-moves title must
    # NOT collide on signature — both materialize.
    assert result.patterns_learned == 2
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 2
    assert len({f.metadata["_reasoning_signature"] for f in fibers}) == 2


# ── Per-model pattern targets (sliders) ──────────────────────────────────────


async def test_target_zero_skips_model_and_leaves_traces(tmp_path: Path, no_embedder: None) -> None:
    # With no target set for the model (default 0), distillation is skipped and
    # its traces stay UNPROCESSED — a preliminary Mine only detects models.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, pattern_targets={})
    result = await distill_reasoning_patterns(storage, BRAIN, cfg, drain=True)
    assert result.patterns_learned == 0
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert len(remaining) == len(_DEBUG_TRACES)  # untouched, not marked processed


async def test_existing_patterns_count_toward_budget_and_raising_distills_remainder(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)
    # First run: target 1 → only the first cluster distills (budget 1); the rest
    # stay unprocessed.
    run1 = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 1}), drain=True
    )
    assert run1.patterns_learned == 1
    # Raise the target to 3: 1 already exists → budget 2 → the 2 remaining
    # clusters distill (raising a target picks up exactly the remainder).
    run2 = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 3}), drain=True
    )
    assert run2.patterns_learned == 2
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 3


async def test_drain_clears_backlog_across_multiple_fetches(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A tiny per-model batch forces the drain loop to fetch more than twice.
    monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._BATCH_PER_MODEL", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)  # 3 clusters, batch=2
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100}), drain=True
    )
    assert result.patterns_learned == 3  # all three clusters drained (3 fetches)
    remaining = await storage.get_unprocessed_reasoning_traces(BRAIN, model="claude-fable-5")
    assert remaining == []


async def test_background_run_processes_one_batch_per_model(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # drain=False (background consolidation) does ONE batch per model per run
    # even with a large budget and a bigger backlog.
    monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._BATCH_PER_MODEL", 2)
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _THREE_CLUSTERS)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100})
    )
    assert result.patterns_learned == 1  # only the first batch this run
    remaining = await storage.get_unprocessed_reasoning_traces(
        BRAIN, model="claude-fable-5", limit=100
    )
    assert len(remaining) == 4


async def test_drain_terminates_when_a_batch_makes_no_progress(
    tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If a batch consumes nothing (pathological no-forward-progress), the drain
    # loop must break rather than re-fetch the same traces forever.
    async def _no_progress(*_a: object, **_k: object) -> tuple[int, int, list[object]]:
        return 0, 0, []  # (created, merged, consumed)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller._process_model_batch", _no_progress
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    result = await distill_reasoning_patterns(
        storage, BRAIN, _ucfg(tmp_path, pattern_targets={"claude-fable-5": 100}), drain=True
    )
    assert result.patterns_learned == 0  # terminated without hanging


# ── Consolidation LEARN_REASONING handler ────────────────────────────────────


async def test_consolidation_learn_reasoning_runs(tmp_path: Path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _ucfg(tmp_path, mining_enabled=True)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))

    async def _fake_distill(storage, brain_id, config, **k):
        return DistillResult(patterns_learned=4, traces_processed=12)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", _fake_distill
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    report = await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])
    assert report.reasoning_patterns_learned == 4


async def test_consolidation_learn_reasoning_guard_disabled(tmp_path: Path, monkeypatch) -> None:
    from surreal_memory.engine.consolidation import ConsolidationEngine, ConsolidationStrategy

    fake = _ucfg(tmp_path, mining_enabled=False)
    monkeypatch.setattr(UnifiedConfig, "load", staticmethod(lambda config_path=None: fake))
    calls = {"n": 0}

    async def _spy(*a, **k):  # pragma: no cover - must NOT run
        calls["n"] += 1
        return DistillResult(patterns_learned=9)

    monkeypatch.setattr(
        "surreal_memory.engine.reasoning_distiller.distill_reasoning_patterns", _spy
    )
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    report = await ConsolidationEngine(storage).run([ConsolidationStrategy.LEARN_REASONING])
    assert report.reasoning_patterns_learned == 0
    assert calls["n"] == 0


class _SpyNamer:
    """Stands in for a PatternNamer, counting renames and releases."""

    def __init__(self) -> None:
        self.renamed = 0
        self.released = 0
        self.acquired = 0

    async def acquire(self) -> None:
        self.acquired += 1

    async def rename(self, pattern: dict, cluster_traces: list[dict]) -> dict:
        self.renamed += 1
        return {**pattern, "title": f"llm-named-{self.renamed}"}

    async def release(self) -> None:
        self.released += 1


class TestLLMNamingIsWiredIn:
    """distill_use_llm has to reach the patterns, and let go of the model after."""

    async def test_patterns_carry_the_llm_title(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert result.patterns_learned >= 1
        assert spy.renamed >= 1
        fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
        titles = [str(f.metadata.get("_reasoning_title", "")) for f in fibers]
        assert any(t.startswith("llm-named") for t in titles)

    async def test_the_model_is_released_when_the_run_finishes(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert spy.acquired == 1
        assert spy.released == 1

    async def test_the_model_is_released_even_when_the_run_blows_up(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crash mid-run must not strand a multi-gigabyte model in VRAM."""
        spy = _SpyNamer()
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: spy
        )

        async def _boom(*_a: object, **_k: object) -> tuple[int, list[object]]:
            raise RuntimeError("storage went away")

        monkeypatch.setattr("surreal_memory.engine.reasoning_distiller._process_model_batch", _boom)
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        with pytest.raises(RuntimeError):
            await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        assert spy.acquired == 1
        assert spy.released == 1

    async def test_no_namer_means_the_heuristic_title_survives(
        self, tmp_path: Path, no_embedder: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "surreal_memory.engine.reasoning_distiller.build_namer", lambda _rt: None
        )
        storage = InMemoryStorage()
        storage.set_brain(BRAIN)
        await _seed(storage, "claude-fable-5", _DEBUG_TRACES)

        await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path), drain=True)

        fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
        assert fibers
        titles = [str(f.metadata.get("_reasoning_title", "")) for f in fibers]
        assert not any("llm-named" in t for t in titles)


# ── reprocess: rebuilding patterns from an already-processed backlog ───────────


async def test_reset_lets_distillation_revisit_a_drained_backlog(
    tmp_path: Path, no_embedder: None
) -> None:
    # The distiller marks EVERY consumed trace processed, so a brain that lost
    # its pattern fibers has no way back — re-mining finds nothing unprocessed.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    # Long retention: this test is about the reset, not the post-run prune
    # (pinned separately below).
    cfg = _ucfg(tmp_path, retention_days=100_000)
    first = await distill_reasoning_patterns(storage, BRAIN, cfg)
    assert first.patterns_learned == 1
    assert await storage.get_unprocessed_reasoning_traces(BRAIN) == []

    # Simulate the loss: patterns gone, traces still staged but all processed.
    for fiber in await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100):
        await storage.delete_fiber(str(fiber.id))
    stuck = await distill_reasoning_patterns(storage, BRAIN, cfg)
    assert (stuck.patterns_learned, stuck.traces_processed) == (0, 0)

    await storage.reset_reasoning_traces_processed(BRAIN)
    rebuilt = await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert rebuilt.patterns_learned == 1
    cov = await reasoning_coverage(storage, "claude-fable-5", cfg)
    assert cov["covered"]["debugging"] is True


async def test_reset_then_redistill_does_not_duplicate_patterns(
    tmp_path: Path, no_embedder: None
) -> None:
    # Re-opening the backlog must be safe to repeat: pattern signatures make the
    # second pass a no-op rather than a duplicate-maker.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, retention_days=100_000)
    await distill_reasoning_patterns(storage, BRAIN, cfg)
    before = len(await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100))

    await storage.reset_reasoning_traces_processed(BRAIN)
    second = await distill_reasoning_patterns(storage, BRAIN, cfg)

    after = len(await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100))
    assert after == before
    assert second.patterns_learned == 0


async def test_reset_cannot_recover_traces_the_retention_prune_removed(
    tmp_path: Path, no_embedder: None
) -> None:
    # A distill run prunes processed traces past retention, so reprocess can only
    # rebuild from what is still staged; anything older needs a backfill re-ingest
    # from the transcripts. Seeded traces are older than the 1-day retention here.
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _DEBUG_TRACES)
    cfg = _ucfg(tmp_path, retention_days=1)

    await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert await storage.get_reasoning_trace_models(BRAIN) == []
    assert await storage.reset_reasoning_traces_processed(BRAIN) == 0


# ── embedder selection: config wins over the environment probe ────────────────


class TestEndpointIsLoopback:
    """The check that decides whether reasoning text may leave this machine.

    A prefix or substring test passes hostnames an attacker can register, so
    these cases are the point of the helper, not decoration.
    """

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://127.0.0.1:11435/v1",
            "http://localhost:11435/v1",
            "http://[::1]:11435/v1",
            "http://127.1.2.3:8080/v1",
        ],
    )
    def test_accepts_real_loopback(self, endpoint: str) -> None:
        from surreal_memory.engine.reasoning_distiller import _endpoint_is_loopback

        assert _endpoint_is_loopback(endpoint) is True

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://127.0.0.1.attacker.example/v1",  # starts with "127."
            "https://localhost.evil.example/v1",  # starts with "localhost"
            "https://not-127.0.0.1.example/v1",  # contains the literal
            "https://api.openai.com/v1",
            "",
            "   ",
        ],
    )
    def test_refuses_everything_else(self, endpoint: str) -> None:
        from surreal_memory.engine.reasoning_distiller import _endpoint_is_loopback

        assert _endpoint_is_loopback(endpoint) is False


class TestGetEmbedderUsesConfig:
    """Selection logic had no coverage at all, which is why the defect survived.

    The `no_embedder` fixture blanks `_get_embedder` for most of this module, so
    nothing ever exercised the real function.
    """

    @staticmethod
    def _cfg(tmp_path: Path, **kw: object) -> UnifiedConfig:
        from surreal_memory.unified_config import EmbeddingSettings

        base: dict[str, object] = {
            "enabled": True,
            "provider": "openai",
            "model": "bge-m3-FP16",
            "dimension": 1024,
            "endpoint": "http://127.0.0.1:11435/v1",
        }
        base.update(kw)
        return UnifiedConfig(
            data_dir=tmp_path / ".surrealmemory",
            current_brain="default",
            embedding=EmbeddingSettings(**base),  # type: ignore[arg-type]
        )

    def test_configured_loopback_provider_is_built(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unrelated GEMINI_API_KEY used to win the environment probe and
        # shadow this configuration entirely.
        monkeypatch.setenv("GEMINI_API_KEY", "irrelevant-but-present")
        # The provider factory must NOT be consulted: it re-resolves the base
        # URL on its own, which is how a validated loopback endpoint ended up
        # pointing at a remote host.
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery._create_provider",
            lambda *_a, **_k: pytest.fail("provider factory must not be used"),
        )
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        # The real client is an optional extra; record the construction instead
        # of requiring it, so this runs everywhere.
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding",
            lambda model="", base_url=None: seen.update(base_url=base_url) or object(),
        )

        assert _get_embedder(self._cfg(tmp_path)) is not None
        assert seen["base_url"] == "http://127.0.0.1:11435/v1"

    def test_remote_endpoint_is_refused_not_silently_degraded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        cfg = self._cfg(tmp_path, endpoint="https://api.openai.com/v1")
        assert _get_embedder(cfg) is None

    def test_hostname_that_merely_looks_local_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        cfg = self._cfg(tmp_path, endpoint="https://127.0.0.1.attacker.example/v1")
        assert _get_embedder(cfg) is None

    def test_disabled_embedding_falls_back_to_the_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Fail-soft contract: embeddings off must not start reading config.
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery._auto_detect_provider",
            lambda: (_ for _ in ()).throw(RuntimeError("no provider")),
        )
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        assert _get_embedder(self._cfg(tmp_path, enabled=False)) is None

    def test_no_config_keeps_the_old_behaviour(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "surreal_memory.engine.semantic_discovery._auto_detect_provider",
            lambda: (_ for _ in ()).throw(RuntimeError("no provider")),
        )
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        assert _get_embedder() is None


class TestClusterCosineIsConfigurable:
    """The clustering threshold belongs to the embedder, not to the module.

    A constant tuned for one embedding model clusters nothing under another:
    the value that shipped sat above the 99th percentile of real bge-m3 trace
    similarity, so the embedding path produced strictly fewer patterns than the
    move-set fallback it is meant to supersede.
    """

    @staticmethod
    def _pair(cosine_like: float) -> list[list[float]]:
        """Two unit vectors whose cosine is exactly ``cosine_like``."""
        import math

        angle = math.acos(cosine_like)
        return [[1.0, 0.0], [math.cos(angle), math.sin(angle)]]

    def test_pair_clusters_below_threshold_and_splits_above_it(self) -> None:
        vectors = self._pair(0.80)
        moves = [[], []]

        lenient = _cluster(vectors, moves, 0.75)
        strict = _cluster(vectors, moves, 0.83)

        assert [len(c) for c in lenient] == [2], "0.80 similarity must cluster at 0.75"
        assert sorted(len(c) for c in strict) == [1, 1], "0.80 must not cluster at 0.83"

    def test_default_threshold_is_taken_from_config_not_the_module(self) -> None:
        rt = ReasoningTrainingConfig()

        assert rt.cluster_cosine == 0.75

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (0.9, 0.9),
            (5.0, 1.0),  # clamped
            (0.0, 0.05),  # floored: single-linkage collapses into one blob
            ("not-a-number", 0.75),  # falls back rather than raising
            (float("nan"), 0.75),  # NaN must not propagate into the floor
            (float("inf"), 1.0),
            (None, 0.75),
        ],
    )
    def test_loader_clamps_and_survives_junk(self, raw: object, expected: float) -> None:
        rt = ReasoningTrainingConfig.from_dict({"cluster_cosine": raw})

        assert rt.cluster_cosine == expected

    def test_survives_a_config_round_trip(self) -> None:
        original = ReasoningTrainingConfig(cluster_cosine=0.62)

        restored = ReasoningTrainingConfig.from_dict(original.to_dict())

        assert restored.cluster_cosine == 0.62


class TestValidatedEndpointIsTheUsedEndpoint:
    """The endpoint that clears the loopback gate must be the one connected to.

    Checking one value and connecting to another makes the gate decorative:
    reasoning traces are private user data, and "the check passed" has to imply
    "the data stayed on this machine".
    """

    @staticmethod
    def _record_openai(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Capture the kwargs the embedder is constructed with.

        The real client is an OPTIONAL extra, so building it would make these
        tests pass only where it happens to be installed — and these guard a
        security property, so they must run everywhere. The contract under test
        is this module's, not the SDK's: which base_url does it hand over.
        """
        seen: dict[str, object] = {}

        class _Recorder:
            def __init__(self, model: str = "", base_url: str | None = None) -> None:
                seen["model"] = model
                seen["base_url"] = base_url
                self._base_url = base_url

        monkeypatch.setattr(
            "surreal_memory.engine.embedding.openai_embedding.OpenAIEmbedding", _Recorder
        )
        return seen

    @staticmethod
    def _config(tmp_path: Path, provider: str, endpoint: str) -> UnifiedConfig:
        from surreal_memory.unified_config import EmbeddingSettings

        return UnifiedConfig(
            data_dir=tmp_path / ".surrealmemory",
            current_brain="default",
            embedding=EmbeddingSettings(
                enabled=True,
                provider=provider,
                model="bge-m3-FP16",
                dimension=1024,
                endpoint=endpoint,
            ),
        )

    def test_openrouter_cannot_reach_its_hardcoded_remote_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        seen = self._record_openai(monkeypatch)

        assert _get_embedder(self._config(tmp_path, "openrouter", "http://127.0.0.1:11435/v1"))

        assert seen["base_url"] == "http://127.0.0.1:11435/v1"
        assert "openrouter.ai" not in str(seen["base_url"])

    def test_config_only_loopback_endpoint_reaches_the_client(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A loopback endpoint set ONLY in config.toml must be used, not just checked."""
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)
        seen = self._record_openai(monkeypatch)

        assert _get_embedder(self._config(tmp_path, "openai", "http://127.0.0.1:11435/v1"))

        assert seen["base_url"] == "http://127.0.0.1:11435/v1"

    def test_ollama_is_refused_when_its_own_base_url_is_remote(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        # Isolate the OLLAMA_BASE_URL path: with no embedding endpoint set,
        # ollama's own base URL is what the client will open.
        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://attacker.example:11434")

        assert _get_embedder(self._config(tmp_path, "ollama", "")) is None

    def test_ollama_is_allowed_on_a_loopback_base_url(self, tmp_path: Path, monkeypatch) -> None:
        from surreal_memory.engine.reasoning_distiller import _get_embedder

        monkeypatch.delenv("SURREAL_MEMORY_EMBEDDING_ENDPOINT", raising=False)
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

        embedder = _get_embedder(self._config(tmp_path, "ollama", ""))

        assert embedder is not None
        assert "127.0.0.1" in (getattr(embedder, "_base_url", "") or "")


# ── Temporal move chain (audit defect 2) ─────────────────────────────────────
#
# `segment_moves` used to iterate the move dictionary, so it returned the SET of
# moves present, emitted in fixed vocabulary order, and `_build_pattern` rendered
# that with arrows. Every pattern therefore claimed a sequence that nothing had
# measured. These tests pin the temporal reading and the honest rendering.


class TestTemporalMoveChain:
    def test_sequence_follows_the_text_not_the_vocabulary(self) -> None:
        # "verify" precedes "backtrack" in _REASONING_MOVES, but this text does
        # the opposite: it backtracks first, then verifies.
        assert segment_moves("Wait, let me reconsider. Now I verify the fix.") == [
            "backtrack",
            "verify",
        ]

    def test_adjacent_repeats_collapse_to_one_step(self) -> None:
        # Three verification markers in a row are one verification move.
        moves = segment_moves("Verify this. Confirm that. Double-check it. I need to move on.")
        assert moves == ["verify", "restate-goal"]

    def test_a_later_return_to_a_move_is_kept(self) -> None:
        # Coming BACK to a move after a different one is part of the shape.
        assert segment_moves("Verify the output. Actually, wait. Verify it again.") == [
            "verify",
            "backtrack",
            "verify",
        ]

    def test_empty_text_has_no_moves(self) -> None:
        assert segment_moves("") == []


class TestStrategyRendering:
    """`_build_pattern` must not render a frequency list as if it were a chain."""

    @staticmethod
    def _pattern(cluster_moves: list[list[str]]) -> dict[str, object]:
        from surreal_memory.engine.reasoning_distiller import _build_pattern

        traces = [
            {"content": f"trace {i}", "trace_hash": f"h{i}"} for i in range(len(cluster_moves))
        ]
        return _build_pattern(traces, None, cluster_moves, "m", "debugging", len(traces))

    def test_shared_chain_renders_with_arrows(self) -> None:
        strategy = str(
            self._pattern([["backtrack", "verify"], ["backtrack", "verify"]])["strategy"]
        )
        assert strategy.startswith("Moves: backtrack -> verify")

    def test_no_shared_chain_renders_unordered_without_arrows(self) -> None:
        # Disjoint move sets: the LCS is empty, so there is no chain to show.
        strategy = str(self._pattern([["verify"], ["backtrack"]])["strategy"])
        assert strategy.startswith("Moves (unordered): ")
        assert "->" not in strategy

    def test_no_moves_at_all_says_so(self) -> None:
        strategy = str(self._pattern([[], []])["strategy"])
        assert strategy.startswith("Moves: (none detected)")
        assert "->" not in strategy

    def test_title_ranks_moves_by_how_many_traces_made_them(self) -> None:
        # One verbose trace repeats verify/backtrack; two quiet traces both
        # hypothesize. The title must follow the traces, not the repetitions.
        pattern = self._pattern(
            [
                ["verify", "backtrack", "verify", "backtrack", "verify"],
                ["hypothesize"],
                ["hypothesize"],
            ]
        )
        assert str(pattern["title"]).startswith("debugging: hypothesize")


# 3 debugging traces whose shared TEMPORAL order is backtrack -> verify, which
# is the reverse of the vocabulary order the old implementation would emit.
_BACKTRACK_THEN_VERIFY_TRACES = [
    "Wait, let me reconsider the traceback. Now I verify the exception is gone.",
    "Hold on, let me reconsider this traceback. I verify the exception no longer fires.",
    "Actually, let me reconsider that traceback. Verify the exception is handled.",
]


async def test_pattern_fiber_records_the_measured_chain(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    await _seed(storage, "claude-fable-5", _BACKTRACK_THEN_VERIFY_TRACES)

    result = await distill_reasoning_patterns(storage, BRAIN, _ucfg(tmp_path))
    assert result.patterns_learned == 1

    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    strategy = str(fibers[0].metadata["_reasoning_strategy"])
    assert strategy.startswith("Moves: backtrack -> verify")


# ── Merge gate: one strategy, one fiber (audit defect 1) ─────────────────────
#
# The pattern signature was sha256(model:category:trace_hashes) — addressed by
# the cluster's CONTENT — so every fresh batch of traces minted a "new" pattern
# even when it described the same strategy. Measured 2026-08-26: 47% of the
# opus-5 slots were duplicate titles, and injection served the same strategy
# twice in one 5-slot block.


async def _seed_batch(
    storage: InMemoryStorage, model: str, prefix: str, contents: list[str]
) -> None:
    """Seed traces under a distinct hash prefix, so batches never collide."""
    await storage.insert_reasoning_traces(
        BRAIN,
        [
            {
                "trace_hash": f"{prefix}-{i}",
                "model": model,
                "session_id": "s",
                "project": "p",
                "task_context": "",
                "content": c,
                "content_chars": len(c),
                "created_at": "2026-03-01T00:00:00",
            }
            for i, c in enumerate(contents)
        ],
    )


# Two disjoint trace sets that describe the SAME strategy: debugging, and the
# text moves backtrack -> verify in both.
_MERGE_BATCH_A = [
    "Wait, let me reconsider the traceback. Now I verify the exception is gone.",
    "Hold on, let me reconsider this traceback. I verify the exception stopped.",
    "Actually, let me reconsider that traceback. Verify the exception is handled.",
]
_MERGE_BATCH_B = [
    "Hold on, rethink this bug. Then validate the crash is gone for good.",
    "Wait, rethink the bug here. Then validate the crash no longer happens.",
    "Actually, rethink that bug. Then validate the crash is fixed properly.",
]
# Same category, same moves, OPPOSITE order — a different strategy, not a dupe.
_MERGE_BATCH_REVERSED = [
    "Verify the traceback first. Actually, let me reconsider the exception.",
    "Verify this traceback now. Wait, let me reconsider that exception.",
    "Verify that traceback again. Hold on, let me reconsider the exception.",
]


async def test_same_strategy_from_two_trace_sets_merges_into_one_fiber(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    cfg = _ucfg(tmp_path)

    await _seed_batch(storage, "claude-fable-5", "a", _MERGE_BATCH_A)
    first = await distill_reasoning_patterns(storage, BRAIN, cfg)
    assert first.patterns_learned == 1

    await _seed_batch(storage, "claude-fable-5", "b", _MERGE_BATCH_B)
    second = await distill_reasoning_patterns(storage, BRAIN, cfg)

    # THE defect, asserted first and using no new API, so that on pre-change
    # code this test fails for the RIGHT reason: the bank used to grow a second
    # fiber for a strategy it already held.
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 1
    md = fibers[0].metadata
    assert md["_reasoning_frequency"] == 6  # 3 + 3, summed
    # ...and it folded in rather than forking a near-duplicate.
    assert second.patterns_learned == 0
    assert second.patterns_merged == 1
    assert len(md["_reasoning_signatures"]) == 2  # both clusters accounted for
    assert md["_reasoning_chain"] == ["backtrack", "verify"]
    # Fields injection reads must survive a merge.
    for key in (
        "_reasoning_title",
        "_reasoning_description",
        "_reasoning_strategy",
        "_reasoning_confidence",
        "_source_model",
        "_reasoning_category",
    ):
        assert md.get(key), key


async def test_a_different_move_order_is_a_different_pattern(
    tmp_path: Path, no_embedder: None
) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    cfg = _ucfg(tmp_path)

    await _seed_batch(storage, "claude-fable-5", "a", _MERGE_BATCH_A)
    await distill_reasoning_patterns(storage, BRAIN, cfg)
    await _seed_batch(storage, "claude-fable-5", "r", _MERGE_BATCH_REVERSED)
    second = await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert second.patterns_learned == 1  # verify -> backtrack is its own strategy
    fibers = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(fibers) == 2
    assert {tuple(f.metadata["_reasoning_chain"]) for f in fibers} == {
        ("backtrack", "verify"),
        ("verify", "backtrack"),
    }


async def test_replaying_the_same_batch_changes_nothing(tmp_path: Path, no_embedder: None) -> None:
    storage = InMemoryStorage()
    storage.set_brain(BRAIN)
    cfg = _ucfg(tmp_path)

    await _seed_batch(storage, "claude-fable-5", "a", _MERGE_BATCH_A)
    await distill_reasoning_patterns(storage, BRAIN, cfg)
    before = (await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100))[0]

    # Re-open the same traces and distill again: same cluster, same signature.
    await storage.reset_reasoning_traces_processed(BRAIN)
    replay = await distill_reasoning_patterns(storage, BRAIN, cfg)

    assert (replay.patterns_learned, replay.patterns_merged) == (0, 0)
    after = await storage.find_fibers(metadata_key="_reasoning_pattern", limit=100)
    assert len(after) == 1
    assert after[0].metadata["_reasoning_frequency"] == before.metadata["_reasoning_frequency"]
    assert after[0].metadata["_reasoning_signatures"] == before.metadata["_reasoning_signatures"]


class TestMergeKey:
    def test_chain_identifies_the_pattern(self) -> None:
        from surreal_memory.engine.reasoning_distiller import merge_key

        a = merge_key("claude-fable-5", "debugging", ["backtrack", "verify"], "t1")
        b = merge_key("Claude-Fable-5", "Debugging", ["Backtrack", "Verify"], "t2")
        assert a == b  # case and title do not fork a pattern

    def test_reversed_chain_is_a_different_key(self) -> None:
        from surreal_memory.engine.reasoning_distiller import merge_key

        assert merge_key("m", "c", ["a", "b"], "t") != merge_key("m", "c", ["b", "a"], "t")

    def test_without_a_chain_the_title_identifies_the_pattern(self) -> None:
        from surreal_memory.engine.reasoning_distiller import merge_key

        assert merge_key("m", "c", [], "same title") == merge_key("m", "c", [], "Same   Title")
        assert merge_key("m", "c", [], "one") != merge_key("m", "c", [], "two")

    def test_chain_is_recovered_from_a_legacy_strategy_line(self) -> None:
        from surreal_memory.engine.reasoning_distiller import chain_from_strategy

        assert chain_from_strategy("Moves: a -> b -> c\nmedoid text") == ["a", "b", "c"]
        assert chain_from_strategy("Moves (unordered): a, b\ntext") == []
        assert chain_from_strategy("Moves: (none detected)\ntext") == []
        assert chain_from_strategy("Moves: \ntext") == []
        assert chain_from_strategy("") == []


class TestMergeArithmetic:
    @staticmethod
    def _fiber(freq: int, conf: float, sig: str = "s0") -> object:
        from surreal_memory.core.fiber import Fiber

        return Fiber.create(
            neuron_ids={"n1"},
            synapse_ids=set(),
            anchor_neuron_id="n1",
            summary="title",
            metadata={
                "_reasoning_pattern": True,
                "_reasoning_frequency": freq,
                "_reasoning_confidence": conf,
                "_reasoning_signature": sig,
                "_reasoning_signatures": [sig],
                "_reasoning_title": "title",
                "_reasoning_description": "old description",
                "_reasoning_strategy": "old strategy",
            },
        )

    @staticmethod
    def _incoming(freq: int, conf: float, sig: str = "s1") -> dict[str, object]:
        return {
            "model": "m",
            "category": "c",
            "title": "new title",
            "description": "new description",
            "strategy": "new strategy",
            "confidence": conf,
            "frequency": freq,
            "signature": sig,
            "chain": ["a", "b"],
        }

    def test_confidence_is_weighted_by_the_traces_behind_it(self) -> None:
        from surreal_memory.engine.reasoning_distiller import _merge_pattern_into

        merged = _merge_pattern_into(self._fiber(9, 0.3), self._incoming(1, 1.0), "s1")
        assert merged is not None
        # (0.3*9 + 1.0*1) / 10 = 0.37 — one tiny fully-clustered batch must not
        # pin the pattern at 1.0, because injection ranks by confidence x frequency.
        assert merged.metadata["_reasoning_confidence"] == 0.37
        assert merged.metadata["_reasoning_frequency"] == 10

    def test_the_better_supported_cluster_describes_the_pattern(self) -> None:
        from surreal_memory.engine.reasoning_distiller import _merge_pattern_into

        better = _merge_pattern_into(self._fiber(2, 0.2), self._incoming(2, 0.9), "s1")
        assert better is not None
        assert better.metadata["_reasoning_strategy"] == "new strategy"
        # Title/summary stay: they are the fiber's identity and its anchor neuron.
        assert better.metadata["_reasoning_title"] == "title"
        assert better.summary == "title"

        worse = _merge_pattern_into(self._fiber(2, 0.9), self._incoming(2, 0.2), "s1")
        assert worse is not None
        assert worse.metadata["_reasoning_strategy"] == "old strategy"

    def test_a_signature_already_carried_is_a_no_op(self) -> None:
        from surreal_memory.engine.reasoning_distiller import _merge_pattern_into

        assert (
            _merge_pattern_into(self._fiber(3, 0.5, "s0"), self._incoming(3, 0.5, "s0"), "s0")
            is None
        )
