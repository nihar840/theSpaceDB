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
    ActivationRecord,
    DriverState,
    MemoryBlock,
    ClusterData,
    GodPoint,
    QueryConfidence,
)
from ._core.cosmic import (
    CosmicLimits,
    THIRTY_THREE_CRORE,
    SMALL_SPACE,
    UNLIMITED,
)
from ._core._exceptions import (
    SpaceDBError,
    BlockNotFoundError,
    VectorDimensionError,
    StoreCorruptedError,
    ClusterNotFoundError,
    EmbedderNotAvailableError,
    StorageQuotaError,
    CapacityExhaustedError,
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
    "ActivationRecord",
    "DriverState",
    "MemoryBlock",
    "ClusterData",
    "GodPoint",
    "QueryConfidence",
    # Cosmic architecture
    "CosmicLimits",
    "THIRTY_THREE_CRORE",
    "SMALL_SPACE",
    "UNLIMITED",
    # Exceptions
    "SpaceDBError",
    "BlockNotFoundError",
    "VectorDimensionError",
    "StoreCorruptedError",
    "ClusterNotFoundError",
    "EmbedderNotAvailableError",
    "StorageQuotaError",
    "CapacityExhaustedError",
]
