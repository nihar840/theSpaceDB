"""Tests for storage measurement and quota enforcement."""

import tempfile
import numpy as np
import pytest
from spacedb import SpaceClient, StorageQuotaError


DIM = 32


def _vec():
    return np.random.randn(DIM).astype(np.float32)


def test_storage_returns_dict():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True)
        mind = client["s1"]
        s = mind.storage()
        assert "used_bytes" in s
        assert "used_mb" in s
        assert "max_mb" in s
        assert "percent" in s


def test_storage_grows_with_blocks():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True)
        mind = client["s2"]
        before = mind.storage()["used_bytes"]
        for _ in range(10):
            mind.ingest(_vec())
        after = mind.storage()["used_bytes"]
        assert after > before


def test_status_includes_storage():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True)
        mind = client["s3"]
        s = mind.status()
        assert "storage_bytes" in s
        assert "storage_mb" in s
        assert "max_size_mb" in s


def test_unlimited_quota_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True)
        mind = client["s4"]
        assert mind.storage()["max_mb"] is None
        assert mind.storage()["percent"] is None
        # Should ingest without limit
        for _ in range(20):
            mind.ingest(_vec())


def test_quota_stops_ingest():
    with tempfile.TemporaryDirectory() as tmp:
        # Tiny quota — the W matrix alone (~64KB) exceeds 0.01 MB (10 KB)
        client = SpaceClient(tmp, dim=DIM, silent=True, max_size_mb=0.01)
        mind = client["s5"]
        with pytest.raises(StorageQuotaError, match="storage limit"):
            mind.ingest(_vec())


def test_per_space_quota_override():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True, max_size_mb=0.01)
        # Override: this space gets a generous quota
        big = client.use("big", max_size_mb=100)
        assert big.storage()["max_mb"] == 100
        # Should ingest fine
        big.ingest(_vec())


def test_zero_means_unlimited():
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True, max_size_mb=0.01)
        # Override with 0 = explicitly unlimited
        free = client.use("free", max_size_mb=0)
        assert free.storage()["max_mb"] is None
        free.ingest(_vec())


def test_realistic_quota():
    """Test with a 1 MB quota — should allow many small blocks."""
    with tempfile.TemporaryDirectory() as tmp:
        client = SpaceClient(tmp, dim=DIM, silent=True, max_size_mb=1)
        mind = client["s6"]
        count = 0
        try:
            for _ in range(500):
                mind.ingest(_vec())
                count += 1
        except StorageQuotaError:
            pass
        # Should have ingested at least some blocks before hitting limit
        assert count > 0
        s = mind.storage()
        assert s["used_mb"] <= 1.1  # slight tolerance
        assert s["percent"] is not None
