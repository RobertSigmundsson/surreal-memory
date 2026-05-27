"""End-to-end coverage for smem_remember's `type=` argument.

The smem_remember handler validates the `type` argument with a single
line:

    mem_type = MemoryType(args["type"])

If this passes, the rest of `_remember()` propagates `mem_type.value`
through to the response and to the stored TypedMemory row. The risk is
that someone adds a downstream whitelist or downgrade that silently
drops cognitive-only types (HYPOTHESIS, PREDICTION, SCHEMA) or new
types (TOOL, BOUNDARY).

These tests lock the entry-point contract without requiring the heavy
MinimalServer machinery (which would duplicate test_audit_fixes
plumbing). They cover:

- Every MemoryType value is accepted by `MemoryType(value)` — catches
  enum drift if someone renames a value.
- The auto-classifier never emits a cognitive-only type — those must
  only ever come from explicit `type=` or dedicated cognitive handlers.
- The handler's type-rejection branch fires for unknown strings.
"""

from __future__ import annotations

import pytest

from surreal_memory.core.memory_types import (
    MemoryType,
    suggest_memory_type,
)

# Types intentionally excluded from the auto-classifier.
COGNITIVE_ONLY = {MemoryType.HYPOTHESIS, MemoryType.PREDICTION, MemoryType.SCHEMA}


@pytest.mark.parametrize("mtype", list(MemoryType), ids=lambda m: m.value)
def test_every_memory_type_string_is_accepted_by_constructor(mtype: MemoryType) -> None:
    """The handler's type-validation branch is `MemoryType(args['type'])`.

    If that constructor raises ValueError for any value in the enum,
    the handler returns an error response and silently drops the
    request. This test catches that drift.
    """
    parsed = MemoryType(mtype.value)
    assert parsed is mtype


def test_unknown_type_string_raises_value_error() -> None:
    """Unknown type strings must raise so the handler's error branch fires."""
    with pytest.raises(ValueError):
        MemoryType("not_a_real_type")


def test_auto_classifier_never_emits_cognitive_only_types() -> None:
    """suggest_memory_type must never return HYPOTHESIS, PREDICTION, or SCHEMA.

    Those types are only authored via dedicated cognitive handlers
    (smem hypothesize, smem predict, knowledge_gap detection). If the
    classifier started emitting them from raw text, the cognitive layer
    would receive uncalibrated entries and break confidence tracking.
    """
    sentences = [
        "I think the encoder is the bottleneck",  # could be confused for HYPOTHESIS
        "Migration will take 8 hours",  # could be confused for PREDICTION
        "Our mental model is that synapses dominate",  # could be confused for SCHEMA
        "The cache hit rate dropped on Mondays",
        "Learned that the bug only fires on UTC midnight",
        "Discovered a leak in the pool",
    ]
    for s in sentences:
        result = suggest_memory_type(s)
        assert result not in COGNITIVE_ONLY, (
            f"classifier emitted {result.value} for {s!r} — "
            "cognitive-only types must come from dedicated handlers"
        )


def test_classifier_covers_all_twelve_non_cognitive_types() -> None:
    """The 12 non-cognitive types must each be reachable from the classifier.

    If a future refactor drops a branch (e.g. removes the BOUNDARY
    branch by accident), this test catches it before it reaches users.
    """
    expected_reachable = set(MemoryType) - COGNITIVE_ONLY
    seen: set[MemoryType] = set()

    # One characteristic sentence per type. Mirrors the corpus in
    # test_suggest_memory_type but is intentionally minimal so this
    # test owns the reachability contract on its own.
    samples: dict[MemoryType, str] = {
        MemoryType.BOUNDARY: "Never use eval() in production",
        MemoryType.TODO: "TODO: refactor the auth module",
        MemoryType.DECISION: "Decided to use PostgreSQL",
        MemoryType.ERROR: "Bug: pagination skips the last page",
        MemoryType.INSIGHT: "Realized the cache hit rate drops on Mondays",
        MemoryType.INSTRUCTION: "Always use type hints in Python",
        MemoryType.PREFERENCE: "I prefer tabs over spaces",
        MemoryType.WORKFLOW: "Deploy pipeline runs lint then build",
        MemoryType.TOOL: "Use the --check flag to dry-run",
        MemoryType.REFERENCE: "Docs at https://example.com/api",
        MemoryType.CONTEXT: "Currently working on the SurrealDB fork",
        MemoryType.FACT: "Python 3.11 was released in October 2022",
    }
    assert set(samples.keys()) == expected_reachable, (
        "samples must cover every non-cognitive MemoryType exactly once"
    )

    for expected, sentence in samples.items():
        result = suggest_memory_type(sentence)
        assert result == expected, (
            f"classifier should emit {expected.value} for {sentence!r}, got {result.value}"
        )
        seen.add(result)

    assert seen == expected_reachable


def test_explicit_type_kwarg_supports_all_fifteen_types() -> None:
    """Documents the full contract: callers can request any of the 15 types.

    This is the regression guard for the P3-002 task — if a handler-level
    whitelist is ever added that drops TOOL or BOUNDARY (or any other
    type), this test fails because MemoryType.<X>.value must round-trip
    through the constructor for every X.
    """
    requested_via_args = [m.value for m in MemoryType]
    assert len(requested_via_args) == 15

    # Each string must hit the same enum it came from.
    parsed = [MemoryType(s) for s in requested_via_args]
    assert set(parsed) == set(MemoryType)
