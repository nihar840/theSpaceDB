"""Tests for SpaceClient — the top-level entry point."""

import tempfile

import numpy as np
import pytest

from spacedb import SpaceClient

DIM = 32


def _rand_vec(dim=DIM):
    v = np.random.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# ── fixtures ──────────────────────────────────────────────


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmp:
        yield SpaceClient(tmp, dim=DIM, silent=True)


# ── tests ─────────────────────────────────────────────────


def test_use_creates_space(client):
    space = client.use("alpha")
    assert space is not None
    assert space.name == "alpha"
    # Calling use again returns the cached instance
    assert client.use("alpha") is space


def test_getitem_shorthand(client):
    space = client["beta"]
    assert space is not None
    assert space.name == "beta"
    assert client["beta"] is space


def test_list_spaces(client):
    client.use("one")
    # Ingest a vector to force the directory to be fully created
    client["one"].ingest(_rand_vec())
    client.use("two")
    client["two"].ingest(_rand_vec())

    names = client.list_spaces()
    assert "one" in names
    assert "two" in names


def test_drop_space_requires_confirm(client):
    client.use("doomed")
    client["doomed"].ingest(_rand_vec())

    with pytest.raises(ValueError, match="confirm=True"):
        client.drop_space("doomed")


def test_drop_space_confirmed(client):
    client.use("doomed")
    client["doomed"].ingest(_rand_vec())
    assert "doomed" in client.list_spaces()

    client.drop_space("doomed", confirm=True)
    assert "doomed" not in client.list_spaces()


def test_status(client):
    client.use("s1")
    client["s1"].ingest(_rand_vec())
    s = client.status()
    assert "root" in s
    assert "spaces" in s
    assert "count" in s
    assert s["count"] >= 1


def test_silent_mode():
    """Creating a client with silent=True should not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        c = SpaceClient(tmp, dim=DIM, silent=True)
        space = c["quiet"]
        space.ingest(_rand_vec())
        assert space.status()["blocks"] == 1
