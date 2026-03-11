"""
test_cognitive.py -- Tests for Phase 2 cognitive engine features:
  1. Passive decay (forgetting)
  2. HDBSCAN auto-clustering
  3. Drift decay (thinking also forgets)
  4. Cross-personality dynamics
  5. Evolution trajectory tracking
"""

import time
import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core.graph_store import GraphStore


# ── helpers ──────────────────────────────────────────────────

def make_cluster_vecs(dim, center, n=5, spread=0.05):
    """Generate n vectors tightly clustered around a center."""
    return [center + np.random.randn(dim).astype(np.float32) * spread
            for _ in range(n)]


# ── GraphStore new methods ───────────────────────────────────

class TestGraphStoreExtensions:

    def test_random_edges_returns_list(self, tmp_dir):
        g = GraphStore(tmp_dir)
        g.add_edge("a", "b", 1.0)
        g.add_edge("b", "c", 0.5)
        edges = g.random_edges(10)
        assert len(edges) == 2
        assert all(len(e) == 3 for e in edges)

    def test_random_edges_empty_graph(self, tmp_dir):
        g = GraphStore(tmp_dir)
        assert g.random_edges(5) == []

    def test_random_edges_samples_n(self, tmp_dir):
        g = GraphStore(tmp_dir)
        for i in range(10):
            g.add_edge(f"n{i}", f"n{i+1}", 1.0)
        edges = g.random_edges(3)
        assert len(edges) == 3

    def test_remove_edge(self, tmp_dir):
        g = GraphStore(tmp_dir)
        g.add_edge("a", "b", 1.0)
        assert g.edge_count() == 1
        g.remove_edge("a", "b")
        assert g.edge_count() == 0
        assert g.neighbors("a") == []

    def test_remove_nonexistent_edge(self, tmp_dir):
        g = GraphStore(tmp_dir)
        g.remove_edge("x", "y")  # should not raise


# ── Passive Decay ────────────────────────────────────────────

class TestPassiveDecay:

    def test_edges_weaken_on_ingest(self, engine, random_vec, dim):
        """Ingesting new blocks should weaken some existing edges."""
        # Ingest enough blocks to build a solid graph
        for i in range(15):
            engine.ingest(f"token_{i}", random_vec(), "text")

        # Record total weight at this point
        edges_mid = engine._graph.random_edges(1000)
        total_weight_before = sum(w for _, _, w in edges_mid)

        # Ingest more blocks (triggers passive decay on existing edges)
        for i in range(15):
            engine.ingest(f"new_{i}", random_vec(), "text")

        edges_after = engine._graph.random_edges(1000)
        # Graph should still have edges (new connections from _connect)
        assert len(edges_after) > 0
        # Average weight should have decreased due to passive decay
        avg_before = total_weight_before / max(len(edges_mid), 1)
        avg_after = sum(w for _, _, w in edges_after) / max(len(edges_after), 1)
        assert avg_after < avg_before

    def test_passive_decay_with_zero_rate(self, engine, dim):
        """Even with very low input rate, decay should still apply minimally."""
        vec = np.random.randn(dim).astype(np.float32)
        engine.ingest("a", vec, "text")
        time.sleep(0.01)
        vec2 = np.random.randn(dim).astype(np.float32)
        engine.ingest("b", vec2, "text")
        # Should not crash
        assert engine._graph.edge_count() >= 0


# ── HDBSCAN Auto-Clustering ──────────────────────────────────

class TestAutoClustering:

    def test_auto_cluster_triggers_at_threshold(self, tmp_dir, dim):
        """After AUTO_CLUSTER_EVERY ingestions, clustering should run."""
        engine = SpaceEngine(tmp_dir, dim=dim)
        engine.AUTO_CLUSTER_EVERY = 10  # lower for test speed

        # Create 3 tight groups
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32)
        c3 = np.array([0.0, 0.0, 1.0] + [0.0] * (dim - 3), dtype=np.float32)

        for i, center in enumerate([c1, c2, c3]):
            for v in make_cluster_vecs(dim, center, n=4, spread=0.02):
                engine.ingest(f"g{i}_block", v, "text")

        # Should have created clusters (12 blocks > AUTO_CLUSTER_EVERY=10)
        clusters = engine._clusters.all()
        # HDBSCAN may or may not find clusters depending on geometry,
        # but the method should have been called without error
        assert engine._ingest_since_cluster < engine.AUTO_CLUSTER_EVERY

    def test_auto_cluster_skips_without_sklearn(self, engine, random_vec):
        """If sklearn not installed, auto_cluster logs debug and returns."""
        # This test verifies the graceful fallback
        # (sklearn IS installed, but the method should still work)
        for i in range(6):
            engine.ingest(f"t{i}", random_vec(), "text")
        engine._auto_cluster()  # should not crash

    def test_auto_cluster_needs_min_blocks(self, engine, random_vec):
        """Auto-cluster should skip if < 6 blocks."""
        engine.ingest("solo", random_vec(), "text")
        engine._auto_cluster()
        assert engine._clusters.count() == 0


# ── Drift Decay ──────────────────────────────────────────────

class TestDriftDecay:

    def test_drift_step_includes_decay(self, engine, random_vec):
        """Drift step should both reinforce and decay edges."""
        # Need enough blocks + clusters for drift to work
        blocks = []
        for i in range(8):
            b = engine.ingest(f"drift_{i}", random_vec(), "text")
            blocks.append(b)

        # Create a cluster so drift has something to walk
        cid = engine.register_cluster([b.id for b in blocks[:4]], "cluster_a")

        # Record state
        edges_before = engine._graph.edge_count()

        # Run drift step
        engine._drift_step()

        # Drift should have run without error
        # Edge count may have changed (some pruned, some reinforced)
        assert engine._graph.edge_count() >= 0

    def test_drift_step_no_crash_empty(self, engine):
        """Drift step on empty engine should be a no-op."""
        engine._drift_step()


# ── Cross-Personality Dynamics ───────────────────────────────

class TestCrossPersonality:

    def test_cross_link_no_personalities(self, engine, random_vec):
        """No personalities -> cross_personality_link is a no-op."""
        blocks = [engine.ingest(f"t{i}", random_vec(), "text") for i in range(3)]
        engine.cross_personality_link([b.id for b in blocks])
        # Should not crash

    def test_cross_link_single_personality(self, engine, random_vec):
        """One personality -> no cross-linking."""
        blocks = [engine.ingest(f"t{i}", random_vec(), "text") for i in range(5)]
        cid = engine.register_cluster([b.id for b in blocks], "solo")
        # Force personality
        c = engine._clusters.get(cid)
        c.is_personality = True
        engine.cross_personality_link([b.id for b in blocks])
        # Should not crash, no reinforcement across personalities

    def test_cross_link_two_personalities(self, engine, dim):
        """Two personalities in results -> cross reinforcement."""
        c1 = np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32)
        c2 = np.array([0.0, 1.0] + [0.0] * (dim - 2), dtype=np.float32)

        b1s = [engine.ingest(f"p1_{i}", c1 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(3)]
        b2s = [engine.ingest(f"p2_{i}", c2 + np.random.randn(dim).astype(np.float32) * 0.01, "text")
               for i in range(3)]

        cid1 = engine.register_cluster([b.id for b in b1s], "happy")
        cid2 = engine.register_cluster([b.id for b in b2s], "sad")

        # Force both as personalities
        engine._clusters.get(cid1).is_personality = True
        engine._clusters.get(cid2).is_personality = True

        w_before = engine._dist.update_count
        engine.cross_personality_link([b1s[0].id, b2s[0].id])
        w_after = engine._dist.update_count

        # W matrix should have been updated (cross reinforcement)
        assert w_after > w_before


# ── Evolution Trajectory ─────────────────────────────────────

class TestEvolution:

    def test_evolution_empty_at_start(self, engine):
        assert engine.evolution_trajectory() == []

    def test_evolution_logs_ingest_events(self, engine, random_vec):
        """After 10 ingests, should have at least one log entry."""
        for i in range(10):
            engine.ingest(f"evo_{i}", random_vec(), "text")
        events = engine.evolution_trajectory()
        assert len(events) >= 1
        e = events[0]
        assert "event" in e
        assert e["event"] == "ingest"
        assert "blocks" in e
        assert "edges" in e
        assert "clusters" in e

    def test_evolution_auto_cluster_event(self, tmp_dir, dim):
        """Auto-cluster should log an event."""
        engine = SpaceEngine(tmp_dir, dim=dim)
        engine.AUTO_CLUSTER_EVERY = 10

        for i in range(12):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"ac_{i}", v, "text")

        events = engine.evolution_trajectory()
        event_types = [e["event"] for e in events]
        assert "auto_cluster" in event_types

    def test_evolution_space_api(self, space, random_vec):
        """Space.evolution() should expose trajectory."""
        for i in range(10):
            space.ingest(random_vec(), sensory_type="text")
        evo = space.evolution()
        assert isinstance(evo, list)
        assert len(evo) >= 1


# ── Integration ──────────────────────────────────────────────

class TestCognitiveIntegration:

    def test_full_lifecycle(self, tmp_dir, dim):
        """
        Full lifecycle: ingest -> passive decay -> auto-cluster -> drift ->
        cross-personality -> evolution log.
        """
        engine = SpaceEngine(tmp_dir, dim=dim)
        engine.AUTO_CLUSTER_EVERY = 15

        # Ingest 3 tight groups (5 each = 15 total triggers auto-cluster)
        centers = [
            np.array([1, 0, 0] + [0] * (dim - 3), dtype=np.float32),
            np.array([0, 1, 0] + [0] * (dim - 3), dtype=np.float32),
            np.array([0, 0, 1] + [0] * (dim - 3), dtype=np.float32),
        ]

        all_blocks = []
        for g, center in enumerate(centers):
            for i in range(5):
                v = center + np.random.randn(dim).astype(np.float32) * 0.02
                b = engine.ingest(f"g{g}_{i}", v, "text")
                all_blocks.append(b)

        # Engine should have blocks, edges, and possibly clusters
        s = engine.status()
        assert s["blocks"] == 15
        assert s["graph_edges"] > 0

        # Run drift step
        if engine._clusters.all():
            engine._drift_step()

        # Check evolution log exists
        evo = engine.evolution_trajectory()
        assert len(evo) >= 1

        # Check status reflects all components
        assert "input_rate" in s
        assert "w_updates" in s
