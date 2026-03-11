# theSpaceDB — A self-evolving cognitive database.
# Blocks in free space. Distance evolves. Clusters emerge. Personalities form.
# Nothing fixed. Everything evolving. We chase infinity.

from .client import SpaceClient
from .space  import Space
from .query  import QueryBuilder
from .drift  import DriftController

from ._core.models import (
    EmotionTag,
    TraitSignal,
    PersonalityState,
    MemoryBlock,
    ClusterData,
)
from ._core._exceptions import (
    SpaceDBError,
    BlockNotFoundError,
    VectorDimensionError,
    StoreCorruptedError,
    ClusterNotFoundError,
    EmbedderNotAvailableError,
    StorageQuotaError,
)

__version__ = "0.3.0"

__all__ = [
    # Main API
    "SpaceClient",
    "Space",
    "QueryBuilder",
    "DriftController",
    # Data models
    "EmotionTag",
    "TraitSignal",
    "PersonalityState",
    "MemoryBlock",
    "ClusterData",
    # Exceptions
    "SpaceDBError",
    "BlockNotFoundError",
    "VectorDimensionError",
    "StoreCorruptedError",
    "ClusterNotFoundError",
    "EmbedderNotAvailableError",
    "StorageQuotaError",
]
