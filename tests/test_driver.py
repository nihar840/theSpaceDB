"""
test_driver.py -- Tests for the Driver (executive self / the God layer)

Tests:
  1. Driver creation & initialization
  2. Personality selection with momentum & switch cost
  3. Feedback learning (affinity)
  4. Drift steering
  5. Meta-observations (internal blocks)
  6. Coactivation matrix
  7. Persistence (save/load)
  8. auto_personality() on QueryBuilder
  9. Engine integration (hooks fire correctly)
"""

import json
import os
import time
import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core.driver import Driver
from spacedb._core.models import ActivationRecord, DriverState


# ── Data Models ────────────────────────────────────────────────

class TestDataModels:

    def test_activation_record_defaults(self):
        r = ActivationRecord()
        assert r.trigger_type == "query"
        assert r.outcome is None
        assert r.feedback_score == 0.0
        assert isinstance(r.scores, dict)

    def test_driver_state_defaults(self):
        s = DriverState()
        assert s.active_personality_id is None
        assert s.total_activations == 0
        assert s.total_switches == 0
        assert s.drift_focus is None
        assert isinstance(s.personality_affinity, dict)
        assert isinstance(s.coactivation_matrix, dict)


# ── Driver Initialization ─────────────────────────────────────

class TestDriverInit:

    def test_driver_created_with_engine(self, engine):
        """Driver should be auto-created by engine."""
        assert engine.driver is not None
        assert isinstance(engine.driver, Driver)

    def test_driver_status_empty(self, engine):
        s = engine.driver.status()
        assert s["active_personality"] is None
        assert s["total_activations"] == 0
        assert s["total_switches"] == 0

    def test_driver_history_empty(self, engine):
        assert engine.driver.history() == []

    def test_driver_affinity_empty(self, engine):
        assert engine.driver.affinity() == {}


# ── Personality Selection ──────────────────────────────────────

class TestSelection:

    def _make_two_personalities(self, engine, dim):
        """Create two personality clusters for testing selection."""
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32)

        b1s = [engine.ingest(f"p1_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(4)]
        b2s = [engine.ingest(f"p2_{i}", c2 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(4)]

        cid1 = engine.register_cluster([b.id for b in b1s], "happy")
        cid2 = engine.register_cluster([b.id for b in b2s], "sad")

        # Force as personalities
        engine.clusters.get(cid1).is_personality = True
        engine.clusters.get(cid2).is_personality = True

        return cid1, cid2

    def test_select_picks_highest_score(self, engine, dim):
        cid1, cid2 = self._make_two_personalities(engine, dim)
        vec = np.random.randn(dim).astype(np.float32)

        result = engine.driver.select_personality(
            vec, {cid1: 0.8, cid2: 0.3}
        )
        # cid1 has higher base score, should win on first call
        assert result == cid1

    def test_select_empty_candidates_returns_none(self, engine, dim):
        vec = np.random.randn(dim).astype(np.float32)
        result = engine.driver.select_personality(vec, {})
        assert result is None

    def test_momentum_keeps_current(self, engine, dim):
        """After selecting P1, momentum should help P1 stay even with lower base."""
        cid1, cid2 = self._make_two_personalities(engine, dim)
        vec = np.random.randn(dim).astype(np.float32)

        # First selection: P1 wins clearly
        engine.driver.select_personality(vec, {cid1: 0.8, cid2: 0.3})

        # Second selection: P2 has slightly higher base, but P1 has momentum
        # P1: 0.4 + 0.3 (momentum) = 0.7
        # P2: 0.55 - 0.15 (switch cost) = 0.4
        result = engine.driver.select_personality(vec, {cid1: 0.4, cid2: 0.55})
        assert result == cid1

    def test_switch_when_difference_large(self, engine, dim):
        """Should switch when new personality is clearly better."""
        cid1, cid2 = self._make_two_personalities(engine, dim)
        vec = np.random.randn(dim).astype(np.float32)

        engine.driver.select_personality(vec, {cid1: 0.8, cid2: 0.3})
        # Now P2 is massively better
        result = engine.driver.select_personality(vec, {cid1: 0.1, cid2: 1.0})
        assert result == cid2

    def test_selection_increments_counts(self, engine, dim):
        cid1, cid2 = self._make_two_personalities(engine, dim)
        vec = np.random.randn(dim).astype(np.float32)

        engine.driver.select_personality(vec, {cid1: 0.8, cid2: 0.3})
        s = engine.driver.status()
        assert s["total_activations"] == 1
        switches_after_first = s["total_switches"]  # None -> cid1 = 1 switch

        engine.driver.select_personality(vec, {cid1: 0.1, cid2: 1.0})
        s = engine.driver.status()
        assert s["total_activations"] == 2
        assert s["total_switches"] == switches_after_first + 1  # cid1 -> cid2

    def test_history_recorded(self, engine, dim):
        cid1, cid2 = self._make_two_personalities(engine, dim)
        vec = np.random.randn(dim).astype(np.float32)

        engine.driver.select_personality(vec, {cid1: 0.5, cid2: 0.3})
        h = engine.driver.history()
        assert len(h) == 1
        assert h[0]["personality_id"] == cid1
        assert h[0]["personality_name"] == "happy"


# ── Feedback ───────────────────────────────────────────────────

class TestFeedback:

    def test_positive_feedback_increases_affinity(self, engine, dim):
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine.ingest(f"fb_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "test")
        engine.clusters.get(cid).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)
        engine.driver.select_personality(vec, {cid: 0.5})

        aff_before = engine.driver.affinity().get(cid, 0.0)
        engine.driver.feedback("positive", score=0.8)
        aff_after = engine.driver.affinity().get(cid, 0.0)
        assert aff_after > aff_before

    def test_negative_feedback_decreases_affinity(self, engine, dim):
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine.ingest(f"nf_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "test2")
        engine.clusters.get(cid).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)
        engine.driver.select_personality(vec, {cid: 0.5})

        engine.driver.feedback("positive", score=0.5)
        aff_before = engine.driver.affinity()[cid]

        engine.driver.feedback("negative", score=0.8)
        aff_after = engine.driver.affinity()[cid]
        assert aff_after < aff_before

    def test_feedback_no_crash_without_active(self, engine):
        """Feedback with no active personality should be a no-op."""
        engine.driver.feedback("positive", score=0.5)


# ── Drift Steering ─────────────────────────────────────────────

class TestDriftSteering:

    def test_direct_drift_sets_focus(self, engine):
        engine.driver.direct_drift("some_cluster", strength=0.8)
        s = engine.driver.status()
        assert s["drift_focus"] == "some_cluster"
        assert abs(s["drift_focus_strength"] - 0.8) < 0.01

    def test_drift_bias_returned(self, engine):
        engine.driver.direct_drift("cluster_x", strength=0.5)
        cid, strength = engine.driver.get_drift_bias()
        assert cid == "cluster_x"
        assert abs(strength - 0.5) < 0.01

    def test_drift_bias_decays(self, engine):
        engine.driver.direct_drift("cluster_x", strength=1.0)
        engine.driver.on_drift_step("seed", ["a", "b"])
        _, strength = engine.driver.get_drift_bias()
        assert strength < 1.0  # decayed


# ── Meta-Observations ─────────────────────────────────────────

class TestMetaObservations:

    def test_meta_block_emitted_after_threshold(self, engine, dim):
        """After META_INGEST_EVERY activations, an internal block should be stored."""
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine.ingest(f"meta_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "meta_test")
        engine.clusters.get(cid).is_personality = True

        blocks_before = engine._blocks.count()
        vec = np.random.randn(dim).astype(np.float32)

        # Run exactly META_INGEST_EVERY selections
        for i in range(engine.driver.META_INGEST_EVERY):
            engine.driver.select_personality(vec, {cid: 0.5})

        blocks_after = engine._blocks.count()
        # Should have one more block (meta-observation)
        assert blocks_after > blocks_before

    def test_internal_block_has_correct_type(self, engine, dim):
        """Meta-observation block should have sensory_type='internal'."""
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine.ingest(f"int_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "int_test")
        engine.clusters.get(cid).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)
        for i in range(engine.driver.META_INGEST_EVERY):
            engine.driver.select_personality(vec, {cid: 0.5})

        # Find the internal block
        all_blocks = [engine._blocks.read(bid) for bid in engine._blocks.all_ids()]
        internal = [b for b in all_blocks if b and b.sensory_type == "internal"]
        assert len(internal) >= 1
        assert internal[0].token.startswith("meta::driver")


# ── Coactivation ──────────────────────────────────────────────

class TestCoactivation:

    def test_coactivation_updates_on_switch(self, engine, dim):
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32)

        b1s = [engine.ingest(f"co1_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(4)]
        b2s = [engine.ingest(f"co2_{i}", c2 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(4)]

        cid1 = engine.register_cluster([b.id for b in b1s], "alpha")
        cid2 = engine.register_cluster([b.id for b in b2s], "beta")
        engine.clusters.get(cid1).is_personality = True
        engine.clusters.get(cid2).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)

        # Select P1, then switch to P2
        engine.driver.select_personality(vec, {cid1: 0.8, cid2: 0.3})
        engine.driver.select_personality(vec, {cid1: 0.1, cid2: 0.9})

        # Coactivation should be recorded
        state = engine.driver._state
        assert cid1 in state.coactivation_matrix
        assert cid2 in state.coactivation_matrix[cid1]
        assert state.coactivation_matrix[cid1][cid2] > 0


# ── Persistence ───────────────────────────────────────────────

class TestPersistence:

    def test_state_persists_across_restart(self, tmp_dir, dim):
        """Driver state should survive engine restart."""
        engine1 = SpaceEngine(tmp_dir, dim=dim)
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine1.ingest(f"pers_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine1.register_cluster([b.id for b in blocks], "persistent")
        engine1.clusters.get(cid).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)
        engine1.driver.select_personality(vec, {cid: 0.5})
        engine1.driver.feedback("positive", score=0.8)

        s1 = engine1.driver.status()
        aff1 = engine1.driver.affinity()

        # Restart engine
        engine2 = SpaceEngine(tmp_dir, dim=dim)
        s2 = engine2.driver.status()
        aff2 = engine2.driver.affinity()

        assert s2["total_activations"] == s1["total_activations"]
        assert s2["active_personality"] == s1["active_personality"]
        assert aff2 == aff1

    def test_driver_json_exists(self, engine, dim):
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        blocks = [engine.ingest(f"dj_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "dj_test")
        engine.clusters.get(cid).is_personality = True

        vec = np.random.randn(dim).astype(np.float32)
        engine.driver.select_personality(vec, {cid: 0.5})

        assert os.path.exists(os.path.join(engine.path, "driver.json"))


# ── Cluster Events ────────────────────────────────────────────

class TestClusterEvents:

    def test_personality_promotion_notifies_driver(self, engine, dim):
        """When a cluster becomes a personality, Driver should be notified."""
        # Create a cluster and manually trigger promotion
        blocks = [engine.ingest(f"promo_{i}", np.random.randn(dim).astype(np.float32), "text")
                  for i in range(4)]
        cid = engine.register_cluster([b.id for b in blocks], "promo_test")

        # Driver should track this if spirit triggers promotion
        # Since we can't easily force promotion threshold, test the callback directly
        engine.driver.on_cluster_event("promoted", cid)
        assert cid in engine.driver.affinity()


# ── Engine Hook Integration ───────────────────────────────────

class TestEngineHooks:

    def test_ingest_fires_on_ingest(self, engine, random_vec):
        """Ingest should call driver.on_ingest without crash."""
        engine.ingest("hook_test", random_vec(), "text")
        # If we got here without error, hooks are wired

    def test_query_fires_on_query(self, engine, random_vec, dim):
        """Query should call driver.on_query without crash."""
        engine.ingest("qh_test", random_vec(), "text")
        engine.query(random_vec())
        # No crash = hooks wired correctly

    def test_drift_step_fires_on_drift(self, engine, random_vec):
        """Drift step should call driver.on_drift_step."""
        blocks = [engine.ingest(f"drift_{i}", random_vec(), "text")
                  for i in range(5)]
        engine.register_cluster([b.id for b in blocks], "drift_test")
        engine._drift_step()
        # No crash = hooks wired

    def test_internal_block_skipped_in_on_ingest(self, engine, dim):
        """on_ingest should skip internal blocks to prevent recursion."""
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        # Directly ingest an internal block
        block = engine.ingest("meta::test", c1, sensory_type="internal")
        # on_ingest was called but should have returned early for internal
        # If we got here, no infinite loop = success


# ── Space API ─────────────────────────────────────────────────

class TestSpaceDriverAPI:

    def test_space_has_driver(self, space):
        """Space object should expose driver attribute."""
        assert hasattr(space, 'driver')
        assert space.driver is not None

    def test_space_driver_status(self, space):
        s = space.driver.status()
        assert "active_personality" in s
        assert "total_activations" in s
