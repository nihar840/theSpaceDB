import uuid, time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryBlock:
    id: str
    token: str
    timestamp: float
    sensory_type: str
    reinforcement_score: float = 1.0
    cluster_id: Optional[str] = None

    @classmethod
    def create(cls, token: str, sensory_type: str = 'text') -> 'MemoryBlock':
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
