"""
test_pantheon.py -- Tests for the God World Architecture (Pantheon routing layer)

Tests cover:
  1. Pantheon initialisation (empty state)
  2. GodPoint spawn / dissolve lifecycle
  3. Routing accuracy (nearest god, top-k, distance ordering)
  4. Centroid normalisation and updates
  5. Cluster sync (spawn, update, dissolve via sync)
  6. Persistence (save / load roundtrip)
  7. Thread safety (concurrent spawns)
  8. Engine integration (routed ingest, routed query, fallback)
"""

import tempfile
import threading

import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core.pantheon import Pantheon
from spacedb._core.models import GodPoint

DIM = 32


def _rand_vec(dim=DIM):
    """Random unit vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _axis_vec(dim, axis):
    """Unit vector pointing along one dimension (orthogonal basis)."""
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v


# ── Pantheon init ────────────────────────────────────────────


class TestPantheonInit:

    def test_empty_init(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        assert p.count == 0
        assert p.all() == []
        assert p.nearest_god(_rand_vec(dim)) is None

    def test_route_empty_returns_empty(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        assert p.route(_rand_vec(dim), top_k=3) == []


# ── GodPoint lifecycle ───────────────────────────────────────


class TestGodPointLifecycle:

    def test_spawn_creates_god(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        god = p.spawn("cluster_1", _axis_vec(dim, 0), block_count=10, spirit=2.5)
        assert god.id.startswith("god_")
        assert god.cluster_id == "cluster_1"
        assert god.block_count == 10
        assert p.count == 1

    def test_spawn_normalises_centroid(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        unnormed = np.ones(dim, dtype=np.float32) * 5.0
        god = p.spawn("c1", unnormed, 5, 1.0)
        assert abs(np.linalg.norm(god.centroid) - 1.0) < 1e-5

    def test_dissolve_removes_god(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        god = p.spawn("c1", _axis_vec(dim, 0), 5, 1.0)
        assert p.count == 1
        p.dissolve(god.id)
        assert p.count == 0
        assert p.get(god.id) is None

    def test_dissolve_nonexistent_is_noop(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        p.dissolve("god_nonexistent")  # should not raise

    def test_get_by_cluster(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        p.spawn("c_abc", _axis_vec(dim, 0), 5, 1.0)
        p.spawn("c_xyz", _axis_vec(dim, 1), 5, 1.0)
        g = p.get_by_cluster("c_abc")
        assert g is not None
        assert g.cluster_id == "c_abc"
        assert p.get_by_cluster("nonexistent") is None


# ── Routing ──────────────────────────────────────────────────


class TestRouting:

    def test_nearest_god_returns_closest(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        p.spawn("c0", _axis_vec(dim, 0), 10, 1.0)
        p.spawn("c1", _axis_vec(dim, 1), 10, 1.0)
        p.spawn("c2", _axis_vec(dim, 2), 10, 1.0)

        nearest = p.nearest_god(_axis_vec(dim, 1))
        assert nearest.cluster_id == "c1"

    def test_route_top_k(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        for i in range(5):
            p.spawn(f"c{i}", _axis_vec(dim, i), 10, 1.0)

        results = p.route(_axis_vec(dim, 0), top_k=3)
        assert len(results) == 3
        # First result should be the closest
        assert results[0][0].cluster_id == "c0"
        # Distances should be ascending
        dists = [d for _, d in results]
        assert dists == sorted(dists)

    def test_route_top_k_larger_than_gods(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        p.spawn("c0", _axis_vec(dim, 0), 10, 1.0)
        results = p.route(_rand_vec(dim), top_k=5)
        assert len(results) == 1

    def test_route_distance_values(self, tmp_dir, dim):
        """Cosine distance between identical vectors should be ~0."""
        p = Pantheon(tmp_dir, dim)
        v = _axis_vec(dim, 0)
        p.spawn("c0", v, 10, 1.0)
        results = p.route(v, top_k=1)
        assert len(results) == 1
        _, dist = results[0]
        assert dist < 0.01  # essentially zero


# ── Update ───────────────────────────────────────────────────


class TestUpdate:

    def test_update_centroid(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        god = p.spawn("c0", _axis_vec(dim, 0), 10, 1.0)

        p.update(god.id, _axis_vec(dim, 1), block_count=15, spirit=3.0)

        updated = p.get(god.id)
        assert updated.block_count == 15
        assert updated.spirit == 3.0
        # Now nearest to axis-1 should be this god
        nearest = p.nearest_god(_axis_vec(dim, 1))
        assert nearest.id == god.id

    def test_update_nonexistent_is_noop(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        p.update("god_nope", _rand_vec(dim), 5, 1.0)  # should not raise


# ── Persistence ──────────────────────────────────────────────


class TestPersistence:

    def test_save_load_roundtrip(self, tmp_dir, dim):
        p1 = Pantheon(tmp_dir, dim)
        p1.spawn("c0", _axis_vec(dim, 0), 10, 2.5)
        p1.spawn("c1", _axis_vec(dim, 1), 5, 1.0)
        assert p1.count == 2

        # Load into a fresh Pantheon instance
        p2 = Pantheon(tmp_dir, dim)
        assert p2.count == 2
        g0 = p2.get_by_cluster("c0")
        assert g0 is not None
        assert g0.block_count == 10
        assert abs(g0.spirit - 2.5) < 1e-6
        # Centroid should be normalised
        assert abs(np.linalg.norm(g0.centroid) - 1.0) < 1e-5
        # Routing should work after reload
        nearest = p2.nearest_god(_axis_vec(dim, 1))
        assert nearest.cluster_id == "c1"


# ── Thread safety ────────────────────────────────────────────


class TestThreadSafety:

    def test_concurrent_spawns(self, tmp_dir, dim):
        p = Pantheon(tmp_dir, dim)
        errors = []

        def spawn_gods(start_idx):
            try:
                for i in range(10):
                    p.spawn(f"c_{start_idx}_{i}", _rand_vec(dim), 5, 1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=spawn_gods, args=(t,))
                   for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert p.count == 40


# ── Engine integration ───────────────────────────────────────


class TestEngineIntegration:

    def test_no_gods_uses_full_scan(self):
        """With < 50 blocks, no gods exist — query uses full scan."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            for i in range(5):
                v = center + np.random.randn(DIM).astype(np.float32) * 0.01
                v = v / (np.linalg.norm(v) + 1e-8)
                engine.ingest(f"token_{i}", v, "text")
            assert engine.pantheon.count == 0
            results = engine.query(center, time_budget_ms=500, limit=10)
            assert len(results) >= 1

    def test_status_includes_god_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            s = engine.status()
            assert "god_points" in s
            assert s["god_points"] == 0

    def test_gods_created_via_manual_cluster_and_sync(self):
        """Manual cluster + sync creates God Points."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            blocks = []
            for i in range(10):
                v = center + np.random.randn(DIM).astype(np.float32) * 0.01
                v = v / (np.linalg.norm(v) + 1e-8)
                b = engine.ingest(f"t_{i}", v, "text")
                blocks.append(b)

            cid = engine.register_cluster([b.id for b in blocks], "test_cluster")
            engine.pantheon.sync_with_clusters(
                engine.clusters.all(),
                engine._vectors,
                engine._blocks,
            )
            assert engine.pantheon.count >= 1
            god = engine.pantheon.get_by_cluster(cid)
            assert god is not None
            assert god.block_count == 10

    def test_routed_query_returns_results(self):
        """With gods, query routes through them and returns results."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            blocks = []
            for i in range(10):
                v = center + np.random.randn(DIM).astype(np.float32) * 0.01
                v = v / (np.linalg.norm(v) + 1e-8)
                b = engine.ingest(f"t_{i}", v, "text")
                blocks.append(b)

            cid = engine.register_cluster([b.id for b in blocks])
            engine.pantheon.sync_with_clusters(
                engine.clusters.all(),
                engine._vectors,
                engine._blocks,
            )
            assert engine.pantheon.count >= 1

            # Query near the cluster center
            results = engine.query(center, time_budget_ms=500, limit=5)
            assert len(results) >= 1

    def test_routed_connect_creates_edges(self):
        """_connect_routed creates graph edges."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            blocks = []
            for i in range(5):
                v = center + np.random.randn(DIM).astype(np.float32) * 0.01
                v = v / (np.linalg.norm(v) + 1e-8)
                b = engine.ingest(f"t_{i}", v, "text")
                blocks.append(b)

            cid = engine.register_cluster([b.id for b in blocks])
            engine.pantheon.sync_with_clusters(
                engine.clusters.all(),
                engine._vectors,
                engine._blocks,
            )

            # Now ingest a new block — should use routed connect
            new_vec = center + np.random.randn(DIM).astype(np.float32) * 0.01
            new_vec = new_vec / (np.linalg.norm(new_vec) + 1e-8)
            new_block = engine.ingest("new_block", new_vec, "text")
            neighbours = engine.graph.neighbors(new_block.id)
            assert len(neighbours) > 0

    def test_sync_dissolves_dead_clusters(self):
        """When a cluster is gone, sync dissolves its God."""
        with tempfile.TemporaryDirectory() as tmp:
            engine = SpaceEngine(tmp, dim=DIM)
            center = _axis_vec(DIM, 0)
            blocks = []
            for i in range(5):
                v = center + np.random.randn(DIM).astype(np.float32) * 0.01
                v = v / (np.linalg.norm(v) + 1e-8)
                b = engine.ingest(f"t_{i}", v, "text")
                blocks.append(b)

            cid = engine.register_cluster([b.id for b in blocks])
            engine.pantheon.sync_with_clusters(
                engine.clusters.all(),
                engine._vectors,
                engine._blocks,
            )
            assert engine.pantheon.count >= 1

            # Dissolve the cluster
            engine.clusters.dissolve(cid)
            # Sync again — god should be gone
            engine.pantheon.sync_with_clusters(
                engine.clusters.all(),
                engine._vectors,
                engine._blocks,
            )
            assert engine.pantheon.get_by_cluster(cid) is None


# ── Space-level API ──────────────────────────────────────────


class TestSpaceAPI:

    def test_space_pantheon_view(self, space, dim):
        """Space.pantheon provides a read-only view."""
        assert space.pantheon.count == 0
        assert space.pantheon.all() == []

    def test_space_status_includes_god_points(self, space):
        s = space.status()
        assert "god_points" in s
        assert s["god_points"] == 0
