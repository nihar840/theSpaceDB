"""Tests for DistanceEngine."""

import tempfile
import numpy as np
import pytest

from spacedb._core.distance_engine import DistanceEngine


DIM = 32


@pytest.fixture
def de(tmp_dir):
    return DistanceEngine(tmp_dir, dim=DIM)


class TestDistanceEngine:

    def test_initial_distance_is_euclidean_like(self, de):
        """With W = I, distance should equal Euclidean distance."""
        v1 = np.zeros(DIM, dtype=np.float32)
        v2 = np.ones(DIM, dtype=np.float32)

        d = de.distance(v1, v2)
        expected = np.linalg.norm(v1.astype(np.float64) - v2.astype(np.float64))

        assert abs(d - expected) < 1e-6, f"Expected ~{expected}, got {d}"

    def test_reinforce_reduces_distance(self, de):
        """Reinforcing two vectors should reduce their distance."""
        v1 = np.random.randn(DIM).astype(np.float32)
        v2 = np.random.randn(DIM).astype(np.float32)

        d_before = de.distance(v1, v2)

        for _ in range(10):
            de.reinforce(v1, v2, strength=0.01)

        d_after = de.distance(v1, v2)
        assert d_after < d_before, (
            f"Distance should decrease after reinforcement: "
            f"before={d_before:.6f}, after={d_after:.6f}"
        )

    def test_decay_increases_distance(self, de):
        """Decaying two vectors should increase their distance."""
        v1 = np.random.randn(DIM).astype(np.float32)
        v2 = np.random.randn(DIM).astype(np.float32)

        d_before = de.distance(v1, v2)

        for _ in range(10):
            de.decay(v1, v2, strength=0.001)

        d_after = de.distance(v1, v2)
        assert d_after > d_before, (
            f"Distance should increase after decay: "
            f"before={d_before:.6f}, after={d_after:.6f}"
        )

    def test_w_matrix_persists(self, tmp_dir):
        """The W matrix survives a reload from disk."""
        v1 = np.random.randn(DIM).astype(np.float32)
        v2 = np.random.randn(DIM).astype(np.float32)

        de1 = DistanceEngine(tmp_dir, dim=DIM)
        for _ in range(5):
            de1.reinforce(v1, v2, strength=0.01)
        w_original = de1.W.copy()

        # Reload from same path
        de2 = DistanceEngine(tmp_dir, dim=DIM)

        assert np.allclose(de2.W, w_original, atol=1e-12), (
            "W matrix was not persisted correctly"
        )
