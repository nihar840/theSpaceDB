"""Tests for SpaceEngine — the core orchestrator."""

import os
import tempfile

import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb._core._version import VERSION_FILE

DIM = 32


def _rand_vec(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def engine():
    with tempfile.TemporaryDirectory() as tmp:
        yield SpaceEngine(tmp, dim=DIM)


# ── tests ─────────────────────────────────────────────────


def test_ingest_returns_block(engine):
    vec = _rand_vec()
    block = engine.ingest("hello world", vec, sensory_type="text")
    assert block is not None
    assert block.id  # non-empty string
    assert block.token == "hello world"
    assert block.sensory_type == "text"
    assert block.reinforcement_score == 1.0


def test_query_returns_results_sorted_by_distance(engine):
    # Ingest several blocks with known vectors
    base = _rand_vec()
    close = base + 0.01 * _rand_vec()
    close = close / (np.linalg.norm(close) + 1e-8)
    far = _rand_vec()  # random direction, likely far

    engine.ingest("close", close.astype(np.float32))
    engine.ingest("far", far.astype(np.float32))
    engine.ingest("base", base.astype(np.float32))

    results = engine.query(base, time_budget_ms=500, limit=10)
    assert len(results) >= 1
    # Results should be sorted by descending score
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_reinforce_strengthens_connection(engine):
    v1, v2 = _rand_vec(), _rand_vec()
    b1 = engine.ingest("alpha", v1)
    b2 = engine.ingest("beta", v2)

    original_score = b1.reinforcement_score
    engine.reinforce(b1.id, b2.id, strength=0.05)

    # Re-read the block from store to pick up the bump
    updated = engine._blocks.read(b1.id)
    assert updated.reinforcement_score > original_score


def test_decay_weakens_connection(engine):
    v1, v2 = _rand_vec(), _rand_vec()
    b1 = engine.ingest("alpha", v1)
    b2 = engine.ingest("beta", v2)

    # First reinforce so there is something to decay
    engine.reinforce(b1.id, b2.id, strength=0.1)
    boosted = engine._blocks.read(b1.id)
    boosted_score = boosted.reinforcement_score

    # Decay should weaken the graph edge weight (not the block score directly)
    engine.decay(b1.id, b2.id, strength=0.05)
    # After decay, the block score should not have increased further
    after_decay = engine._blocks.read(b1.id)
    assert after_decay.reinforcement_score <= boosted_score


def test_register_cluster(engine):
    blocks = [engine.ingest(f"item-{i}", _rand_vec()) for i in range(3)]
    ids = [b.id for b in blocks]

    cid = engine.register_cluster(ids, name="test-cluster")
    assert isinstance(cid, str)
    assert len(cid) > 0

    # Verify blocks were assigned the cluster
    for bid in ids:
        b = engine._blocks.read(bid)
        assert b.cluster_id == cid


def test_status_dict_keys(engine):
    engine.ingest("seed", _rand_vec())
    s = engine.status()
    expected_keys = {
        "blocks", "graph_nodes", "graph_edges", "clusters",
        "personalities", "threshold", "input_rate", "w_updates",
        "drift", "idle_s",
    }
    assert expected_keys.issubset(s.keys())
    assert s["blocks"] >= 1


def test_data_format_version_file_created():
    with tempfile.TemporaryDirectory() as tmp:
        SpaceEngine(tmp, dim=DIM)
        vpath = os.path.join(tmp, VERSION_FILE)
        assert os.path.exists(vpath)
        with open(vpath) as f:
            content = f.read().strip()
        assert content.isdigit()
