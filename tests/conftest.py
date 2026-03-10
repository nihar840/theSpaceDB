"""Shared fixtures for SpaceDB test suite."""

import tempfile
import numpy as np
import pytest

from spacedb._core.engine import SpaceEngine
from spacedb import SpaceClient


@pytest.fixture
def dim():
    """Default embedding dimension for tests."""
    return 32


@pytest.fixture
def tmp_dir():
    """Temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def engine(tmp_dir, dim):
    """A SpaceEngine backed by a fresh temp directory."""
    return SpaceEngine(tmp_dir, dim=dim)


@pytest.fixture
def random_vec(dim):
    """Factory fixture: call ``random_vec()`` to get a random float32 vector."""
    def _make():
        return np.random.randn(dim).astype(np.float32)
    return _make


@pytest.fixture
def client(tmp_dir, dim):
    """A SpaceClient pointed at a temp directory."""
    return SpaceClient(tmp_dir, dim=dim, silent=True)


@pytest.fixture
def space(client):
    """A Space created from the test client."""
    return client.use("test_space")
