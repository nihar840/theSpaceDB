"""Tests for VectorStore."""

import tempfile
import numpy as np
import pytest

from spacedb._core.vector_store import VectorStore
from spacedb._core._exceptions import VectorDimensionError, BlockNotFoundError


DIM = 32


@pytest.fixture
def store(tmp_dir):
    return VectorStore(tmp_dir, dim=DIM)


class TestVectorStore:

    def test_put_get_roundtrip(self, store, random_vec):
        """Store a vector, retrieve it, and confirm values match."""
        vec = random_vec()
        store.put("block-1", vec)

        got = store.get("block-1")
        assert np.allclose(got, vec, atol=1e-7)

    def test_dimension_mismatch_raises(self, store):
        """Putting a vector with wrong dimension raises VectorDimensionError."""
        wrong = np.ones(DIM + 5, dtype=np.float32)
        with pytest.raises(VectorDimensionError):
            store.put("block-bad", wrong)

    def test_nan_vector_raises(self, store):
        """Putting a vector containing NaN raises ValueError."""
        vec = np.zeros(DIM, dtype=np.float32)
        vec[0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            store.put("block-nan", vec)

    def test_search_nearest(self, store):
        """Put 3 vectors, retrieve all, verify distance ordering."""
        base = np.zeros(DIM, dtype=np.float32)
        near = base.copy(); near[0] = 0.1
        mid  = base.copy(); mid[0] = 1.0
        far  = base.copy(); far[0] = 5.0

        store.put("base", base)
        store.put("near", near)
        store.put("mid", mid)
        store.put("far", far)

        # Retrieve all vectors and compute distances to base
        ids, matrix = store.get_all()
        dists = np.linalg.norm(matrix - base, axis=1)

        # Build (id, dist) pairs and sort
        ranked = sorted(zip(ids, dists), key=lambda x: x[1])
        ranked_ids = [r[0] for r in ranked]

        # base itself should be closest (dist 0), then near, mid, far
        assert ranked_ids[0] == "base"
        assert ranked_ids[1] == "near"
        assert ranked_ids[2] == "mid"
        assert ranked_ids[3] == "far"

    def test_persistence(self, tmp_dir, random_vec):
        """Vectors survive a store reload from the same directory."""
        vec = random_vec()

        store1 = VectorStore(tmp_dir, dim=DIM)
        store1.put("persist-1", vec)

        # Reload from same path
        store2 = VectorStore(tmp_dir, dim=DIM)
        got = store2.get("persist-1")

        assert np.allclose(got, vec, atol=1e-7)
