import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = [
    "EmotionTag",
    "TraitSignal",
    "PersonalityState",
    "ActivationRecord",
    "DriverState",
    "MemoryBlock",
    "ClusterData",
    "GodPoint",
    "QueryConfidence",
]

log = logging.getLogger("spacedb.models")

# Well-known sensory types; custom strings are allowed.
KNOWN_SENSORY_TYPES = frozenset({
    "text", "audio", "vision", "internal",
    "response", "expression", "reflection", "dream", "insight",
    "lesson", "value", "perspective", "fact",  # Gurukul + Vedic training types
})


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
class ActivationRecord:
    """One personality activation event recorded by the Driver."""
    timestamp: float = field(default_factory=time.time)
    personality_id: Optional[str] = None
    personality_name: Optional[str] = None
    trigger_type: str = "query"           # query | ingest | drift | manual
    scores: dict[str, float] = field(default_factory=dict)   # pid -> composite
    context_tags: list[str] = field(default_factory=list)
    outcome: Optional[str] = None         # positive | negative | neutral
    feedback_score: float = 0.0
    duration_s: float = 0.0


@dataclass
class DriverState:
    """Persisted state of the Driver (the executive self)."""
    active_personality_id: Optional[str] = None
    active_since: float = field(default_factory=time.time)
    total_activations: int = 0
    total_switches: int = 0
    personality_affinity: dict[str, float] = field(default_factory=dict)     # pid -> [-1, 1]
    coactivation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)  # pid -> {pid -> float}
    drift_focus: Optional[str] = None       # cluster_id to steer drift toward
    drift_focus_strength: float = 0.0


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


@dataclass
class GodPoint:
    """A routing anchor in the data space — a Deva.

    God Points are cluster centroids that form a coarse navigation
    layer above individual memory blocks.  Instead of scanning every
    block on every query (O(n)), we first find the nearest God Point(s)
    using fast cosine similarity (O(g)), then search only within their
    cluster cells (O(k)).

    Cosmological mapping:
        Brahma creates God Points  (spawn)
        Vishnu routes through them (route)
        Shiva transforms them     (dissolve / merge)
    """
    id: str                                          # "god_" + uuid[:8]
    centroid: np.ndarray                             # L2-normalized mean vector
    cluster_id: str                                  # linked cluster in ClusterRegistry
    block_count: int                                 # cached member count
    spirit: float                                    # mirrored from ClusterData.spirit_size
    tier: int = 0                                    # 0 = base god, 1+ = higher (future)
    parent_id: Optional[str] = None                  # parent God (future hierarchy)
    children_ids: list[str] = field(default_factory=list)  # sub-Gods (future)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class QueryConfidence:
    """Confidence metrics for a query result set.

    Lets the mind know how sure it is about a response.
    High confidence = many strong matches across clusters.
    Low confidence = sparse, uncertain, "I don't know" territory.
    """
    score: float           # 0.0-1.0 overall confidence
    hit_count: int         # how many results found
    max_similarity: float  # highest similarity score among results
    mean_similarity: float # average similarity across results
    coverage: float        # fraction of clusters represented (0.0-1.0)
    sparse: bool           # True if hit_count < limit/3
