import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = [
    "EmotionTag",
    "TraitSignal",
    "PersonalityState",
    "MemoryBlock",
    "ClusterData",
]

log = logging.getLogger("spacedb.models")

# Well-known sensory types; custom strings are allowed.
KNOWN_SENSORY_TYPES = frozenset({"text", "audio", "vision", "internal"})


@dataclass
class EmotionTag:
    valence: float = 0.0
    arousal: float = 0.0
    tags: list[str] = field(default_factory=list)


@dataclass
class TraitSignal:
    name: str
    weight: float = 0.0
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    last_updated_utc: float = field(default_factory=time.time)


@dataclass
class PersonalityState:
    id: str
    name: str
    status: str = "emergent"
    dominant_traits: dict[str, float] = field(default_factory=dict)
    source_clusters: list[str] = field(default_factory=list)
    activation_score: float = 0.0
    stability_score: float = 0.0
    min_blocks_required: int = 0
    last_activated_utc: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBlock:
    id: str
    token: str
    timestamp: float
    sensory_type: str
    reinforcement_score: float = 1.0
    cluster_id: Optional[str] = None
    raw_input: Any = None
    normalized_content: Optional[str] = None
    emotion: EmotionTag = field(default_factory=EmotionTag)
    importance: float = 0.0
    novelty: float = 0.0
    confidence: float = 1.0
    linked_previous: list[str] = field(default_factory=list)
    semantic_neighbors: list[str] = field(default_factory=list)
    active_traits: dict[str, float] = field(default_factory=dict)
    personality_context: list[str] = field(default_factory=list)
    action_taken: Any = None
    outcome: Any = None
    feedback: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        token: str,
        sensory_type: str = "text",
        **kwargs: Any,
    ) -> "MemoryBlock":
        if not token or not isinstance(token, str):
            raise ValueError("token must be a non-empty string")
        if not sensory_type or not isinstance(sensory_type, str):
            raise ValueError("sensory_type must be a non-empty string")
        if sensory_type not in KNOWN_SENSORY_TYPES:
            log.debug("Custom sensory_type %r (known: %s)", sensory_type,
                       ", ".join(sorted(KNOWN_SENSORY_TYPES)))
        emotion = kwargs.pop("emotion", None)
        if emotion is None:
            emotion = EmotionTag()
        elif isinstance(emotion, dict):
            emotion = EmotionTag(**emotion)
        elif not isinstance(emotion, EmotionTag):
            raise TypeError("emotion must be EmotionTag, dict, or None")

        normalized_content = kwargs.pop("normalized_content", None) or token

        return cls(
            id=str(uuid.uuid4()),
            token=token,
            timestamp=time.time(),
            sensory_type=sensory_type,
            normalized_content=normalized_content,
            emotion=emotion,
            **kwargs,
        )

    def __setstate__(self, state: dict[str, Any]):
        self.__dict__.update(state)
        if not hasattr(self, "raw_input"):
            self.raw_input = None
        if not hasattr(self, "normalized_content") or self.normalized_content is None:
            self.normalized_content = self.token
        if not hasattr(self, "emotion") or self.emotion is None:
            self.emotion = EmotionTag()
        elif isinstance(self.emotion, dict):
            self.emotion = EmotionTag(**self.emotion)
        if not hasattr(self, "importance"):
            self.importance = 0.0
        if not hasattr(self, "novelty"):
            self.novelty = 0.0
        if not hasattr(self, "confidence"):
            self.confidence = 1.0
        if not hasattr(self, "linked_previous"):
            self.linked_previous = []
        if not hasattr(self, "semantic_neighbors"):
            self.semantic_neighbors = []
        if not hasattr(self, "active_traits"):
            self.active_traits = {}
        if not hasattr(self, "personality_context"):
            self.personality_context = []
        if not hasattr(self, "action_taken"):
            self.action_taken = None
        if not hasattr(self, "outcome"):
            self.outcome = None
        if not hasattr(self, "feedback"):
            self.feedback = None
        if not hasattr(self, "metadata"):
            self.metadata = {}


@dataclass
class ClusterData:
    id: str
    block_ids: list = field(default_factory=list)
    spirit_size: float = 0.0
    is_personality: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    name: Optional[str] = None
