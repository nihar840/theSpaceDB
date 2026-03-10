import logging
import uuid
import time
from dataclasses import dataclass, field
from typing import Optional

__all__ = ["MemoryBlock", "ClusterData"]

log = logging.getLogger("spacedb.models")

# Well-known sensory types; custom strings are allowed.
KNOWN_SENSORY_TYPES = frozenset({"text", "audio", "vision", "internal"})


@dataclass
class MemoryBlock:
    id: str
    token: str
    timestamp: float
    sensory_type: str
    reinforcement_score: float = 1.0
    cluster_id: Optional[str] = None

    @classmethod
    def create(cls, token: str, sensory_type: str = "text") -> "MemoryBlock":
        if not token or not isinstance(token, str):
            raise ValueError("token must be a non-empty string")
        if not sensory_type or not isinstance(sensory_type, str):
            raise ValueError("sensory_type must be a non-empty string")
        if sensory_type not in KNOWN_SENSORY_TYPES:
            log.debug("Custom sensory_type %r (known: %s)", sensory_type,
                       ", ".join(sorted(KNOWN_SENSORY_TYPES)))
        return cls(id=str(uuid.uuid4()), token=token,
                   timestamp=time.time(), sensory_type=sensory_type)


@dataclass
class ClusterData:
    id: str
    block_ids: list = field(default_factory=list)
    spirit_size: float = 0.0
    is_personality: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    name: Optional[str] = None
