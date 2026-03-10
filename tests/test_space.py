"""Tests for Space — the public database-level API."""

import tempfile

import numpy as np
import pytest

from spacedb import Space, EmbedderNotAvailableError, VectorDimensionError

DIM = 32


def _rand_vec(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def space():
    with tempfile.TemporaryDirectory() as tmp:
        yield Space("test_space", tmp, dim=DIM)


# ── tests ─────────────────────────────────────────────────


def test_ingest_raw_vector(space):
    vec = _rand_vec()
    block = space.ingest(vec, sensory_type="text")
    assert block is not None
    assert block.id
    assert block.sensory_type == "text"


def test_ingest_string_without_embedder_raises(space):
    with pytest.raises(EmbedderNotAvailableError):
        space.ingest("some text that needs embedding")


def test_ingest_wrong_type_raises(space):
    with pytest.raises(TypeError, match="must be str or np.ndarray"):
        space.ingest(12345)


def test_query_raw_vector(space):
    # Ingest a few blocks
    vecs = [_rand_vec() for _ in range(5)]
    for i, v in enumerate(vecs):
        space.ingest(v, sensory_type="text")

    # Query with the first vector
    results = space.query(vecs[0]).within(ms=500).limit(10).fetch()
    assert isinstance(results, list)
    # Should return at least one result (the block we just ingested)
    assert len(results) >= 1
    # Each result should be a dict with expected keys
    for r in results:
        assert "id" in r
        assert "token" in r
        assert "score" in r
        assert "cluster" in r
        assert "sensory_type" in r
        assert "reinforcement" in r


def test_query_chain_within_limit(space):
    for _ in range(8):
        space.ingest(_rand_vec())

    qb = space.query(_rand_vec()).within(ms=300).limit(3)
    results = qb.fetch()
    assert len(results) <= 3


def test_reinforce_and_decay(space):
    b1 = space.ingest(_rand_vec())
    b2 = space.ingest(_rand_vec())

    # Reinforce should increase block reinforcement_score
    space.reinforce(b1, b2, strength=0.05)
    updated = space._engine._blocks.read(b1.id)
    assert updated.reinforcement_score > 1.0

    # Decay should not increase it further
    score_after_reinforce = updated.reinforcement_score
    space.decay(b1, b2, strength=0.01)
    after_decay = space._engine._blocks.read(b1.id)
    assert after_decay.reinforcement_score <= score_after_reinforce


def test_create_cluster(space):
    blocks = [space.ingest(_rand_vec()) for _ in range(4)]
    cid = space.create_cluster(blocks, name="my-cluster")
    assert isinstance(cid, str)
    assert len(cid) > 0

    # Verify cluster appears in cluster view
    info = space.clusters.get(cid)
    assert info is not None
    assert info["name"] == "my-cluster"
    assert info["blocks"] == 4


def test_status(space):
    space.ingest(_rand_vec())
    s = space.status()
    assert "space" in s
    assert s["space"] == "test_space"
    assert "blocks" in s
    assert s["blocks"] >= 1


def test_vector_dim_mismatch_raises(space):
    wrong_dim_vec = np.random.randn(DIM + 10).astype(np.float32)
    with pytest.raises(VectorDimensionError):
        space.ingest(wrong_dim_vec)
