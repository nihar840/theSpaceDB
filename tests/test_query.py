"""Tests for QueryBuilder — chainable query API."""

import tempfile

import numpy as np
import pytest

from spacedb import Space, QueryBuilder

DIM = 32


def _rand_vec(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def space():
    with tempfile.TemporaryDirectory() as tmp:
        s = Space("qb_test", tmp, dim=DIM)
        # Seed with some blocks so queries have data
        for _ in range(6):
            s.ingest(_rand_vec())
        yield s


# ── tests ─────────────────────────────────────────────────


def test_within_positive_only(space):
    qb = space.query(_rand_vec())
    with pytest.raises(ValueError):
        qb.within(ms=0)
    with pytest.raises(ValueError):
        qb.within(ms=-10)


def test_limit_positive_only(space):
    qb = space.query(_rand_vec())
    with pytest.raises(ValueError):
        qb.limit(0)
    with pytest.raises(ValueError):
        qb.limit(-5)


def test_fetch_returns_list_of_dicts(space):
    results = space.query(_rand_vec()).within(ms=500).limit(10).fetch()
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, dict)
        assert "id" in r
        assert "token" in r
        assert "score" in r
        assert "cluster" in r
        assert "sensory_type" in r
        assert "reinforcement" in r


def test_as_personality_chain(space):
    # as_personality should be chainable and not crash
    qb = space.query(_rand_vec()).as_personality("nonexistent").within(ms=200).limit(5)
    results = qb.fetch()
    assert isinstance(results, list)


def test_repr(space):
    qb = space.query(_rand_vec())
    r = repr(qb)
    assert "QueryBuilder" in r
    assert "<vector>" in r
    assert "within=" in r
    assert "limit=" in r
