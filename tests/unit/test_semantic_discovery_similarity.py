"""Focused tests for the cosine-similarity seam used by semantic discovery.

The discovery rewrite ranks candidate pairs by cosine similarity over stored
embeddings (vectorised via numpy, with this pure-python function as the
reference / fallback). Guard the maths and the zero-norm edge case.
"""

from __future__ import annotations

import math

from surreal_memory.engine.semantic_discovery import _cosine_similarity


def test_identical_vectors_similarity_one() -> None:
    assert math.isclose(_cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0, rel_tol=1e-9)


def test_orthogonal_vectors_similarity_zero() -> None:
    assert math.isclose(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_opposite_vectors_similarity_minus_one() -> None:
    assert math.isclose(_cosine_similarity([1.0, 1.0], [-1.0, -1.0]), -1.0, rel_tol=1e-9)


def test_zero_norm_returns_zero() -> None:
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine_similarity([1.0, 1.0], [0.0, 0.0]) == 0.0


def test_scale_invariance() -> None:
    base = _cosine_similarity([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    scaled = _cosine_similarity([10.0, 20.0, 30.0], [4.0, 5.0, 6.0])
    assert math.isclose(base, scaled, rel_tol=1e-9)
