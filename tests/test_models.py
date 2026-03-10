"""Tests for MemoryBlock and ClusterData."""

import pytest
from spacedb._core.models import MemoryBlock, ClusterData


class TestMemoryBlock:

    def test_create_block(self):
        """Valid creation produces a block with expected fields."""
        block = MemoryBlock.create("hello world", sensory_type="text")

        assert block.id  # non-empty UUID string
        assert block.token == "hello world"
        assert block.sensory_type == "text"
        assert block.reinforcement_score == 1.0
        assert block.cluster_id is None
        assert block.timestamp > 0

    def test_empty_token_raises(self):
        """Empty-string token must raise ValueError."""
        with pytest.raises(ValueError, match="token"):
            MemoryBlock.create("", sensory_type="text")

    def test_empty_sensory_type_raises(self):
        """Empty-string sensory_type must raise ValueError."""
        with pytest.raises(ValueError, match="sensory_type"):
            MemoryBlock.create("hello", sensory_type="")

    def test_custom_sensory_type(self):
        """Non-standard sensory types like 'lidar' are accepted."""
        block = MemoryBlock.create("point cloud data", sensory_type="lidar")
        assert block.sensory_type == "lidar"


class TestClusterData:

    def test_cluster_data_defaults(self):
        """ClusterData defaults are correct when only id is provided."""
        cd = ClusterData(id="c1")

        assert cd.id == "c1"
        assert cd.block_ids == []
        assert cd.spirit_size == 0.0
        assert cd.is_personality is False
        assert cd.created_at > 0
        assert cd.updated_at > 0
        assert cd.name is None
