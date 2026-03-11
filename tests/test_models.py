"""Tests for MemoryBlock, trait state, and ClusterData."""

import pytest
from spacedb import EmotionTag, PersonalityState, TraitSignal
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
        assert block.normalized_content == "hello world"
        assert block.linked_previous == []
        assert block.active_traits == {}
        assert isinstance(block.emotion, EmotionTag)

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

    def test_create_block_with_experience_metadata(self):
        block = MemoryBlock.create(
            "heard a bird",
            sensory_type="audio",
            raw_input={"samples": 2048},
            normalized_content="bird chirp nearby",
            emotion={"valence": 0.4, "arousal": 0.7, "tags": ["curiosity"]},
            importance=0.8,
            novelty=0.6,
            confidence=0.9,
            linked_previous=["a1", "b2"],
            semantic_neighbors=["c3"],
            active_traits={"curiosity": 0.5},
            personality_context=["observer"],
            action_taken="look_up",
            outcome={"seen": True},
            feedback={"reward": 0.2},
            metadata={"source": "microphone"},
        )

        assert block.raw_input == {"samples": 2048}
        assert block.normalized_content == "bird chirp nearby"
        assert block.emotion.tags == ["curiosity"]
        assert block.importance == 0.8
        assert block.linked_previous == ["a1", "b2"]
        assert block.semantic_neighbors == ["c3"]
        assert block.personality_context == ["observer"]
        assert block.metadata["source"] == "microphone"


class TestTraitAndPersonalityState:

    def test_trait_signal_defaults(self):
        signal = TraitSignal(name="curiosity")
        assert signal.name == "curiosity"
        assert signal.weight == 0.0
        assert signal.confidence == 0.0
        assert signal.sources == []
        assert signal.last_updated_utc > 0

    def test_personality_state_defaults(self):
        state = PersonalityState(id="p1", name="explorer")
        assert state.id == "p1"
        assert state.name == "explorer"
        assert state.status == "emergent"
        assert state.dominant_traits == {}
        assert state.source_clusters == []
        assert state.activation_score == 0.0
        assert state.stability_score == 0.0
        assert state.min_blocks_required == 0
        assert state.last_activated_utc > 0


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
