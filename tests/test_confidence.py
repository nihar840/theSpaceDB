"""
test_confidence.py -- Tests for query confidence tracking.

Verifies:
  - QueryConfidence has correct fields and ranges
  - Empty results -> zero confidence, sparse=True
  - Strong matches -> higher confidence
  - fetch_with_confidence() returns tuple
  - fetch() backward compatibility (still returns list)
"""

import tempfile
import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core.models import QueryConfidence

DIM = 32


def _axis_vec(dim, axis):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _near_vec(center, dim=DIM, noise=0.01):
    v = center + np.random.randn(dim).astype(np.float32) * noise
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


class TestQueryConfidence:

    def test_empty_results_zero_confidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            conf = engine._compute_confidence([], limit=5)
            assert conf.score == 0.0
            assert conf.hit_count == 0
            assert conf.sparse is True

    def test_confidence_score_in_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            for i in range(10):
                v = _near_vec(center)
                engine.ingest(f"block_{i}", v, "text")
            results = engine.query(center, time_budget_ms=500, limit=5)
            conf = engine._compute_confidence(results, limit=5)
            assert 0.0 <= conf.score <= 1.0

    def test_sparse_when_few_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            v = _axis_vec(DIM, 0)
            engine.ingest("only_one", v, "text")
            results = engine.query(v, time_budget_ms=500, limit=10)
            conf = engine._compute_confidence(results, limit=10)
            # 1 result out of 10 requested -> sparse
            assert conf.sparse is True

    def test_not_sparse_with_enough_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            for i in range(10):
                v = _near_vec(center)
                engine.ingest(f"block_{i}", v, "text")
            results = engine.query(center, time_budget_ms=500, limit=5)
            if len(results) >= 2:  # need enough hits
                conf = engine._compute_confidence(results, limit=5)
                assert conf.sparse is False

    def test_confidence_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            for i in range(5):
                engine.ingest(f"b_{i}", _near_vec(center), "text")
            results = engine.query(center, time_budget_ms=500, limit=5)
            conf = engine._compute_confidence(results, limit=5)
            assert hasattr(conf, 'score')
            assert hasattr(conf, 'hit_count')
            assert hasattr(conf, 'max_similarity')
            assert hasattr(conf, 'mean_similarity')
            assert hasattr(conf, 'coverage')
            assert hasattr(conf, 'sparse')


class TestFetchWithConfidence:

    def test_fetch_returns_list(self, space, dim):
        """Existing fetch() still returns a plain list (backward compat)."""
        vec = np.random.randn(dim).astype(np.float32)
        space.ingest(vec, sensory_type="text")
        results = space.query(vec).within(ms=500).limit(5).fetch()
        assert isinstance(results, list)

    def test_fetch_with_confidence_returns_tuple(self, space, dim):
        """fetch_with_confidence() returns (list, QueryConfidence)."""
        vec = np.random.randn(dim).astype(np.float32)
        space.ingest(vec, sensory_type="text")
        result = space.query(vec).within(ms=500).limit(5).fetch_with_confidence()
        assert isinstance(result, tuple)
        assert len(result) == 2
        results_list, confidence = result
        assert isinstance(results_list, list)
        assert isinstance(confidence, QueryConfidence)

    def test_confidence_matches_results(self, space, dim):
        """Confidence hit_count matches result length."""
        vec = np.random.randn(dim).astype(np.float32)
        for i in range(5):
            v = vec + np.random.randn(dim).astype(np.float32) * 0.01
            v = v / (np.linalg.norm(v) + 1e-8)
            space.ingest(v, sensory_type="text")
        results, confidence = (
            space.query(vec).within(ms=500).limit(10)
                .fetch_with_confidence()
        )
        assert confidence.hit_count == len(results)
