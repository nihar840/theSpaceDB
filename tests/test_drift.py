"""Tests for DriftController — cognitive drift background engine."""

import tempfile
import time

import numpy as np
import pytest

from spacedb import Space

DIM = 32


def _rand_vec(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def space():
    with tempfile.TemporaryDirectory() as tmp:
        s = Space("drift_test", tmp, dim=DIM)
        # Seed some data so drift has something to walk
        for _ in range(5):
            s.ingest(_rand_vec())
        yield s


# ── tests ─────────────────────────────────────────────────


def test_start_stop(space):
    assert not space.drift.active
    space.drift.start(idle_seconds=60.0)
    assert space.drift.active
    space.drift.stop()
    assert not space.drift.active


def test_active_property(space):
    assert space.drift.active is False
    space.drift.start(idle_seconds=60.0)
    assert space.drift.active is True
    space.drift.stop()
    assert space.drift.active is False


def test_status_keys(space):
    s = space.drift.status()
    assert "active" in s
    assert "idle_s" in s
    assert "input_rate" in s
    assert s["active"] is False

    space.drift.start(idle_seconds=60.0)
    s2 = space.drift.status()
    assert s2["active"] is True
    space.drift.stop()
