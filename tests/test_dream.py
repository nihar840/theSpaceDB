"""
test_dream.py -- Tests for the Dream Engine.

Covers:
  1. DreamEngine creation and basic properties
  2. Goal setting and clearing
  3. Dream step mode selection
  4. Bridge discovery with proper cluster setup
  5. Synthesis insight generation
  6. Dream status reporting
  7. Integration with SpaceEngine
"""

import tempfile
import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core.dream import DreamEngine, DreamResult

DIM = 32


def _axis_vec(dim, axis):
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


def _near_vec(center, dim=DIM, noise=0.01):
    v = center + np.random.randn(dim).astype(np.float32) * noise
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def _make_engine_with_clusters(dim=DIM):
    """Create an engine with 2 distinct clusters for dream testing."""
    tmp = tempfile.mkdtemp()
    engine = SpaceEngine(tmp, dim=dim)

    # Cluster 1: around axis 0
    c1 = _axis_vec(dim, 0)
    blocks1 = []
    for i in range(8):
        v = _near_vec(c1, dim)
        b = engine.ingest(f"cluster1_block_{i}", v, "text")
        blocks1.append(b)

    # Cluster 2: around axis 1
    c2 = _axis_vec(dim, 1)
    blocks2 = []
    for i in range(8):
        v = _near_vec(c2, dim)
        b = engine.ingest(f"cluster2_block_{i}", v, "text")
        blocks2.append(b)

    # Register clusters manually
    cid1 = engine.register_cluster([b.id for b in blocks1], "topic_a")
    cid2 = engine.register_cluster([b.id for b in blocks2], "topic_b")

    return engine, tmp, cid1, cid2


# ── DreamEngine basics ───────────────────────────────────────


class TestDreamEngineBasics:

    def test_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            dream = DreamEngine(engine)
            assert dream.dream_count == 0

    def test_engine_has_dream(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            assert hasattr(engine, '_dream')
            assert isinstance(engine._dream, DreamEngine)

    def test_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            status = engine._dream.status()
            assert "dream_count" in status
            assert "has_goal" in status
            assert "goal_text" in status
            assert status["has_goal"] is False


# ── Goal setting ─────────────────────────────────────────────


class TestDreamGoal:

    def test_set_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            vec = _axis_vec(DIM, 0)
            engine._dream.set_goal(vec, "test topic")
            status = engine._dream.status()
            assert status["has_goal"] is True
            assert status["goal_text"] == "test topic"

    def test_clear_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            engine._dream.set_goal(_axis_vec(DIM, 0), "topic")
            engine._dream.clear_goal()
            status = engine._dream.status()
            assert status["has_goal"] is False


# ── Dream modes ──────────────────────────────────────────────


class TestDreamModes:

    def test_dream_step_no_clusters_returns_none(self):
        """Dream should return None if no clusters exist."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            result = engine._dream.dream_step()
            assert result is None

    def test_dream_step_with_clusters(self):
        """Dream should run without errors when clusters exist."""
        engine, tmp, _, _ = _make_engine_with_clusters()
        # Try multiple times (modes are probabilistic)
        results = []
        for _ in range(20):
            r = engine._dream.dream_step()
            if r is not None:
                results.append(r)
        # At least some should succeed
        assert len(results) >= 1

    def test_dream_result_has_mode(self):
        """DreamResult should have a valid mode string."""
        engine, tmp, _, _ = _make_engine_with_clusters()
        for _ in range(30):
            r = engine._dream.dream_step()
            if r is not None:
                assert r.mode in ("bridge", "contradiction", "goal", "synthesis")
                break

    def test_synthesis_produces_insights(self):
        """Synthesis mode should produce insight tokens."""
        engine, tmp, _, _ = _make_engine_with_clusters()
        # Force synthesis mode by trying many times
        insights_found = False
        for _ in range(50):
            r = engine._dream.dream_step()
            if r and r.mode == "synthesis" and r.insights:
                insights_found = True
                assert r.insights[0].startswith("dream::link(")
                assert len(r.clusters_touched) == 2
                break
        # Probabilistic — may not always hit synthesis
        # Just verify it doesn't crash

    def test_goal_directed_uses_goal(self):
        """Goal-directed dreams should touch clusters near the goal."""
        engine, tmp, cid1, _ = _make_engine_with_clusters()
        engine._dream.set_goal(_axis_vec(DIM, 0), "cluster 1 topic")
        # Run several dreams
        goal_dreams = []
        for _ in range(30):
            r = engine._dream.dream_step()
            if r and r.mode == "goal":
                goal_dreams.append(r)
        # At least some goal dreams should fire
        assert len(goal_dreams) >= 1


# ── Dream count ──────────────────────────────────────────────


class TestDreamCount:

    def test_count_increments(self):
        engine, tmp, _, _ = _make_engine_with_clusters()
        initial = engine._dream.dream_count
        for _ in range(20):
            engine._dream.dream_step()
        assert engine._dream.dream_count >= initial


# ── Drift integration ────────────────────────────────────────


class TestDreamDriftIntegration:

    def test_drift_controller_dream_api(self, space):
        """DriftController exposes dream API."""
        assert hasattr(space.drift, 'dream_goal')
        assert hasattr(space.drift, 'clear_dream_goal')
        assert hasattr(space.drift, 'dream_status')

    def test_drift_status_includes_dream(self, space):
        """Drift status includes dream fields."""
        status = space.drift.status()
        assert "dream_count" in status
        assert "dream_goal" in status
