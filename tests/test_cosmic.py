"""Tests for Phase 4: 33 Crore Cosmic Architecture.

Tests cover:
  - CosmicLimits dataclass and presets
  - block_importance() scoring function
  - Storage deletion (BlockStore, VectorStore, GraphStore)
  - CosmicReaper eviction engine
  - Engine integration (dissolve_block, capacity guards, edge limits)
"""

import time
import tempfile
import pytest
import numpy as np

from spacedb._core.cosmic import (
    CosmicLimits, THIRTY_THREE_CRORE, SMALL_SPACE, UNLIMITED,
    block_importance, CosmicReaper,
)
from spacedb._core.block_store import BlockStore
from spacedb._core.vector_store import VectorStore
from spacedb._core.graph_store import GraphStore
from spacedb._core.models import MemoryBlock
from spacedb._core.engine import SpaceEngine
from spacedb._core._exceptions import CapacityExhaustedError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dim():
    return 32


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def random_vec(dim):
    def _make():
        v = np.random.randn(dim).astype(np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v /= n
        return v
    return _make


@pytest.fixture
def tiny_limits():
    """Very small limits for fast testing."""
    return CosmicLimits(
        max_blocks=10,
        max_gods=3,
        max_edges_per_block=5,
        max_total_edges=50,
        max_clusters=5,
        max_personalities=3,
        reaper_sample_size=8,
        reaper_evict_batch=3,
        reaper_min_age_seconds=0.0,
    )


@pytest.fixture
def cosmic_engine(tmp_dir, dim, tiny_limits):
    """Engine with tiny cosmic limits for testing."""
    return SpaceEngine(tmp_dir, dim=dim, cosmic_limits=tiny_limits)


# ---------------------------------------------------------------------------
# TestCosmicLimits
# ---------------------------------------------------------------------------

class TestCosmicLimits:

    def test_default_limits_are_33_crore(self):
        """Default preset uses 330 million blocks."""
        assert THIRTY_THREE_CRORE.max_blocks == 330_000_000
        assert THIRTY_THREE_CRORE.max_gods == 33
        assert THIRTY_THREE_CRORE.max_edges_per_block == 33
        assert THIRTY_THREE_CRORE.max_total_edges == 330_000_000
        assert THIRTY_THREE_CRORE.max_clusters == 3_333
        assert THIRTY_THREE_CRORE.max_personalities == 33

    def test_small_space_preset(self):
        """SMALL_SPACE preset has reduced limits for testing."""
        assert SMALL_SPACE.max_blocks == 1000
        assert SMALL_SPACE.max_gods == 5
        assert SMALL_SPACE.reaper_min_age_seconds == 0.0

    def test_unlimited_preset(self):
        """UNLIMITED preset has effectively no limits."""
        assert UNLIMITED.max_blocks >= 2**62
        assert UNLIMITED.max_gods >= 2**62

    def test_frozen_dataclass(self):
        """CosmicLimits is immutable (frozen)."""
        with pytest.raises(AttributeError):
            THIRTY_THREE_CRORE.max_blocks = 42

    def test_custom_limits(self):
        """Can create custom limits."""
        custom = CosmicLimits(max_blocks=100, max_gods=5)
        assert custom.max_blocks == 100
        assert custom.max_gods == 5
        # Other fields keep defaults
        assert custom.max_edges_per_block == 33


# ---------------------------------------------------------------------------
# TestBlockImportance
# ---------------------------------------------------------------------------

class TestBlockImportance:

    def _make_block(self, reinforcement=1.0, sensory_type="text",
                    timestamp=None):
        block = MemoryBlock.create("test", sensory_type=sensory_type)
        block.reinforcement_score = reinforcement
        if timestamp is not None:
            block.timestamp = timestamp
        return block

    def test_high_reinforcement_high_score(self):
        """Blocks with high reinforcement score more important."""
        now = time.time()
        high = self._make_block(reinforcement=5.0, timestamp=now)
        low = self._make_block(reinforcement=0.1, timestamp=now)
        limits = CosmicLimits()
        s_high = block_importance(high, 10, 0.5, now, limits)
        s_low = block_importance(low, 10, 0.5, now, limits)
        assert s_high > s_low

    def test_old_block_low_recency(self):
        """Old blocks have lower importance due to recency decay."""
        now = time.time()
        recent = self._make_block(timestamp=now)
        old = self._make_block(timestamp=now - 200_000)  # ~2.3 days ago
        limits = CosmicLimits()
        s_recent = block_importance(recent, 5, 0.5, now, limits)
        s_old = block_importance(old, 5, 0.5, now, limits)
        assert s_recent > s_old

    def test_many_edges_boost_score(self):
        """Blocks with more edges are more important."""
        now = time.time()
        block = self._make_block(timestamp=now)
        limits = CosmicLimits()
        s_many = block_importance(block, 30, 0.5, now, limits)
        s_few = block_importance(block, 1, 0.5, now, limits)
        assert s_many > s_few

    def test_value_type_gets_boost(self):
        """'value' sensory type gets 1.5x importance boost."""
        now = time.time()
        value_block = self._make_block(sensory_type="value", timestamp=now)
        text_block = self._make_block(sensory_type="text", timestamp=now)
        limits = CosmicLimits()
        s_value = block_importance(value_block, 5, 0.5, now, limits)
        s_text = block_importance(text_block, 5, 0.5, now, limits)
        assert s_value > s_text
        assert abs(s_value / s_text - 1.5) < 0.01

    def test_reflection_type_gets_boost(self):
        """'reflection' sensory type also gets 1.5x boost."""
        now = time.time()
        ref_block = self._make_block(sensory_type="reflection", timestamp=now)
        text_block = self._make_block(sensory_type="text", timestamp=now)
        limits = CosmicLimits()
        s_ref = block_importance(ref_block, 5, 0.5, now, limits)
        s_text = block_importance(text_block, 5, 0.5, now, limits)
        assert s_ref > s_text

    def test_score_always_positive(self):
        """Importance score is always positive."""
        now = time.time()
        block = self._make_block(reinforcement=0.0, timestamp=now - 1_000_000)
        limits = CosmicLimits()
        score = block_importance(block, 0, 0.0, now, limits)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# TestStorageDeletion
# ---------------------------------------------------------------------------

class TestStorageDeletion:

    def test_block_store_soft_delete(self, tmp_dir):
        """Soft-deleted blocks are no longer readable."""
        store = BlockStore(tmp_dir)
        block = MemoryBlock.create("hello", sensory_type="text")
        store.append(block)
        assert store.read(block.id) is not None

        store.soft_delete(block.id)
        assert store.read(block.id) is None
        assert store.is_deleted(block.id)

    def test_block_store_alive_count(self, tmp_dir):
        """alive_count excludes tombstoned blocks."""
        store = BlockStore(tmp_dir)
        blocks = [MemoryBlock.create(f"b{i}", sensory_type="text") for i in range(5)]
        for b in blocks:
            store.append(b)
        assert store.alive_count() == 5
        assert store.count() == 5

        store.soft_delete(blocks[0].id)
        store.soft_delete(blocks[1].id)
        assert store.alive_count() == 3
        assert store.count() == 5  # total unchanged

    def test_block_store_alive_ids(self, tmp_dir):
        """alive_ids returns only non-tombstoned block IDs."""
        store = BlockStore(tmp_dir)
        b1 = MemoryBlock.create("one", sensory_type="text")
        b2 = MemoryBlock.create("two", sensory_type="text")
        store.append(b1)
        store.append(b2)

        store.soft_delete(b1.id)
        alive = store.alive_ids()
        assert b1.id not in alive
        assert b2.id in alive

    def test_block_store_compact_removes_tombstones(self, tmp_dir):
        """Compaction rewrites log without tombstoned blocks."""
        store = BlockStore(tmp_dir)
        blocks = [MemoryBlock.create(f"b{i}", sensory_type="text") for i in range(5)]
        for b in blocks:
            store.append(b)

        store.soft_delete(blocks[0].id)
        store.soft_delete(blocks[2].id)
        store.compact()

        assert store.alive_count() == 3
        assert store.count() == 3  # after compaction, count == alive_count
        assert not store.is_deleted(blocks[0].id)  # tombstone cleared
        # Surviving blocks are still readable
        assert store.read(blocks[1].id) is not None
        assert store.read(blocks[3].id) is not None

    def test_block_store_tombstone_persistence(self, tmp_dir):
        """Tombstones survive store reload."""
        store1 = BlockStore(tmp_dir)
        block = MemoryBlock.create("persist", sensory_type="text")
        store1.append(block)
        store1.soft_delete(block.id)

        store2 = BlockStore(tmp_dir)
        assert store2.read(block.id) is None
        assert store2.is_deleted(block.id)

    def test_vector_store_remove_frees_row(self, tmp_dir, dim):
        """Removing a vector zeros out its row and frees the slot."""
        store = VectorStore(tmp_dir, dim)
        vec = np.random.randn(dim).astype(np.float32)
        store.put("block1", vec)
        assert store.alive_count() == 1

        store.remove("block1")
        assert store.alive_count() == 0
        assert store.count() == 1  # _count unchanged (row was used)

    def test_vector_store_put_reuses_freed_row(self, tmp_dir, dim):
        """New puts reuse freed rows instead of growing."""
        store = VectorStore(tmp_dir, dim)
        v1 = np.random.randn(dim).astype(np.float32)
        v2 = np.random.randn(dim).astype(np.float32)
        v3 = np.random.randn(dim).astype(np.float32)

        store.put("a", v1)
        store.put("b", v2)
        store.remove("a")  # frees row 0

        store.put("c", v3)  # should reuse row 0
        assert store.alive_count() == 2

        # Verify the new vector is correct
        got = store.get("c")
        np.testing.assert_array_almost_equal(got, v3)

    def test_vector_store_compact(self, tmp_dir, dim):
        """Compaction rebuilds matrix without dead rows."""
        store = VectorStore(tmp_dir, dim)
        vecs = [np.random.randn(dim).astype(np.float32) for _ in range(5)]
        for i, v in enumerate(vecs):
            store.put(f"b{i}", v)

        store.remove("b1")
        store.remove("b3")
        store.compact()

        assert store.alive_count() == 3
        assert store.count() == 3
        # Surviving vectors are correct
        np.testing.assert_array_almost_equal(store.get("b0"), vecs[0])
        np.testing.assert_array_almost_equal(store.get("b2"), vecs[2])

    def test_graph_store_remove_node(self, tmp_dir):
        """remove_node removes all edges involving a node."""
        graph = GraphStore(tmp_dir)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "c", 0.5)
        graph.add_edge("b", "c", 0.3)

        removed = graph.remove_node("a")
        assert removed == 2  # a-b and a-c

        # b-c should still exist
        assert graph.edge_count_for("b") == 1
        assert graph.edge_count_for("a") == 0

    def test_graph_store_edge_count_for(self, tmp_dir):
        """edge_count_for returns correct count for a node."""
        graph = GraphStore(tmp_dir)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "c", 0.5)
        graph.add_edge("a", "d", 0.3)
        assert graph.edge_count_for("a") == 3
        assert graph.edge_count_for("b") == 1

    def test_graph_store_total_edge_count(self, tmp_dir):
        """total_edge_count counts each undirected edge once."""
        graph = GraphStore(tmp_dir)
        graph.add_edge("a", "b", 1.0)
        graph.add_edge("a", "c", 0.5)
        graph.add_edge("b", "c", 0.3)
        assert graph.total_edge_count() == 3


# ---------------------------------------------------------------------------
# TestCosmicReaper
# ---------------------------------------------------------------------------

class TestCosmicReaper:

    def test_reaper_does_nothing_under_capacity(self, cosmic_engine, random_vec):
        """Reaper doesn't evict when under capacity."""
        for i in range(5):
            cosmic_engine.ingest(f"block{i}", random_vec())
        assert not cosmic_engine._reaper.should_reap()
        evicted = cosmic_engine._reaper.reap()
        assert evicted == 0

    def test_reaper_evicts_weakest_blocks(self, tmp_dir, dim):
        """Reaper evicts blocks when at capacity."""
        limits = CosmicLimits(
            max_blocks=8,
            max_gods=2**63,
            max_edges_per_block=2**63,
            max_total_edges=2**63,
            max_clusters=2**63,
            max_personalities=2**63,
            reaper_sample_size=16,
            reaper_evict_batch=3,
            reaper_min_age_seconds=0.0,
        )
        engine = SpaceEngine(tmp_dir, dim=dim, cosmic_limits=limits)

        # Ingest exactly at capacity
        for i in range(8):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"block{i}", v)

        assert engine._blocks.alive_count() == 8
        assert engine._reaper.should_reap()

        evicted = engine._reaper.reap()
        assert evicted > 0
        assert engine._blocks.alive_count() < 8

    def test_reaper_respects_min_age(self, tmp_dir, dim):
        """Reaper won't evict blocks younger than min_age."""
        limits = CosmicLimits(
            max_blocks=5,
            max_gods=2**63,
            max_edges_per_block=2**63,
            max_total_edges=2**63,
            max_clusters=2**63,
            max_personalities=2**63,
            reaper_sample_size=16,
            reaper_evict_batch=3,
            reaper_min_age_seconds=9999.0,  # all blocks are "too young"
        )
        engine = SpaceEngine(tmp_dir, dim=dim, cosmic_limits=limits)

        for i in range(5):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"block{i}", v)

        # All blocks are younger than 9999 seconds
        evicted = engine._reaper.reap()
        assert evicted == 0  # nothing old enough to reap

    def test_reaper_status(self, cosmic_engine, random_vec):
        """Reaper status reports correct info."""
        status = cosmic_engine._reaper.status()
        assert status["total_evictions"] == 0
        assert "limits" in status
        assert status["limits"]["max_blocks"] == 10


# ---------------------------------------------------------------------------
# TestEngineIntegration
# ---------------------------------------------------------------------------

class TestEngineIntegration:

    def test_dissolve_block_cascade(self, tmp_dir, dim):
        """dissolve_block removes block from all stores."""
        engine = SpaceEngine(tmp_dir, dim=dim)
        v1 = np.random.randn(dim).astype(np.float32)
        v2 = np.random.randn(dim).astype(np.float32)
        b1 = engine.ingest("first", v1)
        b2 = engine.ingest("second", v2)

        # Both blocks exist
        assert engine._blocks.read(b1.id) is not None
        assert engine._blocks.read(b2.id) is not None

        engine.dissolve_block(b1.id)

        # Block 1 is gone everywhere
        assert engine._blocks.read(b1.id) is None
        assert engine._blocks.is_deleted(b1.id)
        assert engine._graph.edge_count_for(b1.id) == 0
        # Block 2 still exists
        assert engine._blocks.read(b2.id) is not None

    def test_ingest_triggers_reap_at_capacity(self, tmp_dir, dim):
        """Ingesting at capacity triggers reaper before new block is added."""
        limits = CosmicLimits(
            max_blocks=5,
            max_gods=2**63,
            max_edges_per_block=2**63,
            max_total_edges=2**63,
            max_clusters=2**63,
            max_personalities=2**63,
            reaper_sample_size=16,
            reaper_evict_batch=2,
            reaper_min_age_seconds=0.0,
        )
        engine = SpaceEngine(tmp_dir, dim=dim, cosmic_limits=limits)

        # Fill to capacity
        for i in range(5):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"block{i}", v)

        # Next ingest should trigger reap + succeed
        v_new = np.random.randn(dim).astype(np.float32)
        new_block = engine.ingest("new_block", v_new)
        assert new_block is not None
        # Some old blocks were reaped
        assert engine._reaper._evictions > 0

    def test_edge_limit_per_block_enforced(self, tmp_dir, dim):
        """Blocks don't exceed max_edges_per_block."""
        limits = CosmicLimits(
            max_blocks=2**63,
            max_gods=2**63,
            max_edges_per_block=3,  # very tight limit
            max_total_edges=2**63,
            max_clusters=2**63,
            max_personalities=2**63,
        )
        engine = SpaceEngine(tmp_dir, dim=dim, cosmic_limits=limits)

        # Ingest many blocks - each should have at most 3 edges
        for i in range(15):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"block{i}", v)

        # Check edge counts
        for bid in engine._blocks.all_ids():
            assert engine._graph.edge_count_for(bid) <= 3

    def test_capacity_exhausted_error(self, tmp_dir, dim):
        """CapacityExhaustedError raised when reaping can't free space."""
        limits = CosmicLimits(
            max_blocks=3,
            max_gods=2**63,
            max_edges_per_block=2**63,
            max_total_edges=2**63,
            max_clusters=2**63,
            max_personalities=2**63,
            reaper_sample_size=8,
            reaper_evict_batch=1,
            reaper_min_age_seconds=99999.0,  # can't evict anything
        )
        engine = SpaceEngine(tmp_dir, dim=dim, cosmic_limits=limits)

        # Fill to capacity
        for i in range(3):
            v = np.random.randn(dim).astype(np.float32)
            engine.ingest(f"block{i}", v)

        # Next ingest should fail (can't reap due to min_age)
        v = np.random.randn(dim).astype(np.float32)
        with pytest.raises(CapacityExhaustedError):
            engine.ingest("overflow", v)

    def test_system_state_includes_pressure(self, cosmic_engine, random_vec):
        """SystemState includes capacity_pressure field."""
        cosmic_engine.ingest("first", random_vec())
        state = cosmic_engine._system_state()
        assert hasattr(state, 'capacity_pressure')
        assert 0.0 <= state.capacity_pressure <= 1.0
        # 1 block out of 10 limit = 0.1 pressure
        assert abs(state.capacity_pressure - 0.1) < 0.01

    def test_status_includes_alive_count(self, cosmic_engine, random_vec):
        """Engine status includes blocks_alive and reaper_evictions."""
        cosmic_engine.ingest("test", random_vec())
        status = cosmic_engine.status()
        assert 'blocks_alive' in status
        assert 'reaper_evictions' in status
        assert status['blocks_alive'] == 1
        assert status['reaper_evictions'] == 0


# ---------------------------------------------------------------------------
# TestSpaceAPI
# ---------------------------------------------------------------------------

class TestSpaceAPI:

    def test_space_cosmic_status(self, tmp_dir, dim):
        """Space.cosmic_status() returns capacity info."""
        from spacedb import SpaceClient, SMALL_SPACE
        client = SpaceClient(tmp_dir, dim=dim, silent=True,
                             cosmic_limits=SMALL_SPACE)
        space = client["test"]

        v = np.random.randn(dim).astype(np.float32)
        space.ingest(v, sensory_type="text", raw_input="hello")

        status = space.cosmic_status()
        assert status["blocks"]["alive"] == 1
        assert status["blocks"]["limit"] == 1000
        assert "reaper" in status

    def test_space_dissolve_block(self, tmp_dir, dim):
        """Space.dissolve_block() cascade-deletes a block."""
        from spacedb import SpaceClient
        client = SpaceClient(tmp_dir, dim=dim, silent=True)
        space = client["test"]

        v = np.random.randn(dim).astype(np.float32)
        block = space.ingest(v, sensory_type="text", raw_input="dissolve me")

        space.dissolve_block(block.id)
        assert space._engine._blocks.read(block.id) is None

    def test_space_compact(self, tmp_dir, dim):
        """Space.compact() reclaims space after evictions."""
        from spacedb import SpaceClient
        client = SpaceClient(tmp_dir, dim=dim, silent=True)
        space = client["test"]

        vecs = [np.random.randn(dim).astype(np.float32) for _ in range(5)]
        blocks = [space.ingest(v, sensory_type="text",
                               raw_input=f"block{i}")
                  for i, v in enumerate(vecs)]

        space.dissolve_block(blocks[0].id)
        space.dissolve_block(blocks[2].id)
        space.compact()

        # After compaction, dead blocks are truly gone
        assert space._engine._blocks.alive_count() == 3
        assert space._engine._blocks.count() == 3
