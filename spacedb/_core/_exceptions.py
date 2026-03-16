"""Custom exception hierarchy for SpaceDB."""


class SpaceDBError(Exception):
    """Base exception for all SpaceDB errors."""


class BlockNotFoundError(SpaceDBError):
    """Raised when a block_id does not exist in the store."""


class VectorDimensionError(SpaceDBError):
    """Raised when an embedding's dimension does not match the space."""


class StoreCorruptedError(SpaceDBError):
    """Raised when persisted data is corrupted or unreadable."""


class ClusterNotFoundError(SpaceDBError):
    """Raised when a cluster_id does not exist."""


class EmbedderNotAvailableError(SpaceDBError):
    """Raised when sentence-transformers is not installed and text auto-embedding is attempted."""


class StorageQuotaError(SpaceDBError):
    """Raised when a Space has reached its max_size_mb limit and cannot ingest new blocks."""


class CapacityExhaustedError(SpaceDBError):
    """Raised when cosmic capacity limit is reached and eviction cannot free space."""
