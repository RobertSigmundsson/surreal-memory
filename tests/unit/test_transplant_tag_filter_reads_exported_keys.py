"""Unit pinning test: the transplant tag filter must read the keys exports actually write.

``_fiber_matches_tags`` keyed on ``fiber["tags"]``. The in-memory backend writes that
key; the SurrealDB backend never has. A tag-filtered transplant of a SurrealDB snapshot
therefore matched zero fibers and reported it as "no fiber carries that tag" — a silent
empty result, not an error.
"""

from __future__ import annotations

from surreal_memory.engine.brain_transplant import _fiber_matches_tags


class TestFiberMatchesTags:
    def test_matches_a_surrealdb_snapshot_that_has_only_the_halves(self) -> None:
        """SurrealDB export shape: auto_tags / agent_tags, no combined ``tags`` key."""
        fiber = {"id": "f1", "auto_tags": ["kb", "norms"], "agent_tags": ["reviewed"]}
        assert _fiber_matches_tags(fiber, frozenset({"kb"})) is True
        assert _fiber_matches_tags(fiber, frozenset({"reviewed"})) is True

    def test_still_matches_an_in_memory_snapshot_that_has_the_union_key(self) -> None:
        """In-memory export shape: the combined ``tags`` key must keep working."""
        fiber = {"id": "f2", "tags": ["kb"], "auto_tags": ["kb"], "agent_tags": []}
        assert _fiber_matches_tags(fiber, frozenset({"kb"})) is True

    def test_casing_is_ignored_on_both_sides(self) -> None:
        fiber = {"id": "f3", "auto_tags": ["KB"]}
        assert _fiber_matches_tags(fiber, frozenset({"kb"})) is True

    def test_still_says_no_when_the_tag_really_is_absent(self) -> None:
        """The filter must not become an always-true match — that is the other failure."""
        fiber = {"id": "f4", "auto_tags": ["norms"], "agent_tags": ["reviewed"]}
        assert _fiber_matches_tags(fiber, frozenset({"kb"})) is False
        assert _fiber_matches_tags({"id": "f5"}, frozenset({"kb"})) is False
