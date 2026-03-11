"""
space.py — Space: the database-level object

Like a MongoDB database, a Space holds all blocks, clusters and
personalities for one named mind.

Supports both text auto-embedding (requires ``sentence-transformers``)
and raw numpy vectors — pick whichever fits your pipeline.

Usage (text — needs ``pip install thespacedb[embeddings]``)::

    mind = client["my_mind"]
    block  = mind.ingest("apple tastes sweet")
    result = mind.query("fruit").within(ms=200).limit(10).fetch()

Usage (raw vectors — zero extra dependencies)::

    vec = my_encoder.encode("apple tastes sweet")
    block  = mind.ingest(vec, sensory_type="text")
    result = mind.query(query_vec).limit(10).fetch()
"""

from __future__ import annotations
import logging
import os
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from ._core.engine       import SpaceEngine
from ._core._exceptions  import (
    EmbedderNotAvailableError, VectorDimensionError, StorageQuotaError,
)
from .query              import QueryBuilder
from .drift              import DriftController

if TYPE_CHECKING:
    from ._core.models import MemoryBlock, ClusterData

log = logging.getLogger("spacedb.space")


class ClusterView:
    """Read-only view of clusters & personalities for a Space."""

    def __init__(self, engine: SpaceEngine):
        self._e = engine

    def all(self) -> list[dict]:
        return [self._fmt(c) for c in self._e.clusters.all()]

    def personalities(self) -> list[dict]:
        return [self._fmt(c) for c in self._e.clusters.personalities()]

    def get(self, cluster_id: str) -> Optional[dict]:
        c = self._e.clusters.get(cluster_id)
        return self._fmt(c) if c else None

    def refresh_spirit(self, cluster_id: str) -> float:
        return self._e.compute_spirit(cluster_id)

    def _fmt(self, c) -> dict:
        return {
            'id':           c.id,
            'name':         c.name,
            'blocks':       len(c.block_ids),
            'spirit_size':  round(c.spirit_size, 4),
            'personality':  c.is_personality,
        }


class Space:
    """
    A named cognitive space (analogous to a MongoDB database).
    Holds all memory blocks, clusters, and personalities for one mind.
    """

    def __init__(self, name: str, base_path: str, dim: int = 384,
                 max_size_mb: Optional[float] = None):
        self.name  = name
        self._path = os.path.join(base_path, name)
        os.makedirs(self._path, exist_ok=True)

        self._engine       = SpaceEngine(self._path, dim)
        self._embedder     = None          # lazy-loaded
        self._dim          = dim
        self._max_size_mb  = max_size_mb   # None = unlimited

        self.drift    = DriftController(self._engine)
        self.clusters = ClusterView(self._engine)

    # ── ingest ───────────────────────────────────────────────
    def ingest(
        self,
        content: Union[str, np.ndarray],
        sensory_type: str = "text",
    ) -> "MemoryBlock":
        """
        Store a piece of experience.

        Parameters
        ----------
        content : str or np.ndarray
            * **str** — auto-embedded via sentence-transformers (must be
              installed: ``pip install thespacedb[embeddings]``).
            * **np.ndarray** — pre-computed embedding vector of shape
              ``(dim,)``. No embedding model required.
        sensory_type : str
            Label for the modality (``"text"``, ``"image"``, etc.).

        Returns
        -------
        MemoryBlock
            The stored block. Keep ``.id`` for later reinforce / decay.
        """
        self._check_quota()
        if isinstance(content, np.ndarray):
            vec = self._validate_vector(content)
            token = f"<raw:{sensory_type}>"
            return self._engine.ingest(token, vec, sensory_type)
        elif isinstance(content, str):
            vec = self._embed(content)
            return self._engine.ingest(content, vec, sensory_type)
        else:
            raise TypeError(
                f"content must be str or np.ndarray, got {type(content).__name__}"
            )

    # ── query ────────────────────────────────────────────────
    def query(self, content: Union[str, np.ndarray]) -> QueryBuilder:
        """
        Start a chainable query.

        Parameters
        ----------
        content : str or np.ndarray
            * **str** — auto-embedded (requires sentence-transformers).
            * **np.ndarray** — pre-computed query vector.

        Examples::

            space.query("fruit").fetch()
            space.query(my_vec).within(ms=500).limit(5).fetch()
            space.query("fruit").as_personality("food").within(ms=300).fetch()
        """
        if isinstance(content, np.ndarray):
            vec = self._validate_vector(content)
            return QueryBuilder(self, _vector=vec)
        elif isinstance(content, str):
            return QueryBuilder(self, _text=content)
        else:
            raise TypeError(
                f"content must be str or np.ndarray, got {type(content).__name__}"
            )

    # ── reinforce ────────────────────────────────────────────
    def reinforce(self, block_a, block_b, strength: float = 0.01):
        """
        Reinforce the connection between two blocks.
        Accepts MemoryBlock objects or block id strings.
        """
        id_a = block_a if isinstance(block_a, str) else block_a.id
        id_b = block_b if isinstance(block_b, str) else block_b.id
        self._engine.reinforce(id_a, id_b, strength)

    def decay(self, block_a, block_b, strength: float = 0.001):
        """Weaken the connection between two blocks."""
        id_a = block_a if isinstance(block_a, str) else block_a.id
        id_b = block_b if isinstance(block_b, str) else block_b.id
        self._engine.decay(id_a, id_b, strength)

    # ── cluster management ───────────────────────────────────
    def create_cluster(self, blocks: list, name: Optional[str] = None) -> str:
        """
        Manually group blocks into a named cluster.
        Accepts MemoryBlock objects or id strings.
        Returns cluster_id.
        """
        ids = [b if isinstance(b, str) else b.id for b in blocks]
        return self._engine.register_cluster(ids, name)

    # ── storage ───────────────────────────────────────────────
    def storage(self) -> dict:
        """
        Current storage usage for this space.

        Returns
        -------
        dict
            ``used_bytes``, ``used_mb``, ``max_mb`` (None if unlimited),
            ``percent`` (None if unlimited).
        """
        used = self._engine.storage_bytes()
        used_mb = round(used / (1024 * 1024), 2)
        pct = round(used_mb / self._max_size_mb * 100, 1) if self._max_size_mb else None
        return {
            "used_bytes": used,
            "used_mb":    used_mb,
            "max_mb":     self._max_size_mb,
            "percent":    pct,
        }

    # ── status ───────────────────────────────────────────────
    def status(self) -> dict:
        s = self._engine.status()
        s["space"]       = self.name
        s["max_size_mb"] = self._max_size_mb
        return s

    # ── evolution ──────────────────────────────────────────────
    def evolution(self) -> list[dict]:
        """Read the evolution trajectory log for this space."""
        return self._engine.evolution_trajectory()

    # ── internal ─────────────────────────────────────────────
    def _check_quota(self):
        """Raise StorageQuotaError if space has reached its size limit."""
        if self._max_size_mb is None:
            return
        used = self._engine.storage_bytes()
        limit = self._max_size_mb * 1024 * 1024
        if used >= limit:
            used_mb = round(used / (1024 * 1024), 2)
            raise StorageQuotaError(
                f"Space '{self.name}' has reached its storage limit "
                f"({used_mb} MB / {self._max_size_mb} MB). "
                f"Cannot ingest new blocks. "
                f"Increase max_size_mb or create a new space."
            )
    def _validate_vector(self, vec: np.ndarray) -> np.ndarray:
        """Validate shape and values of a raw embedding vector."""
        vec = np.asarray(vec, dtype=np.float32).squeeze()
        if vec.ndim != 1:
            raise VectorDimensionError(
                f"Expected 1-D vector, got shape {vec.shape}"
            )
        if vec.shape[0] != self._dim:
            raise VectorDimensionError(
                f"Vector dim {vec.shape[0]} ≠ space dim {self._dim}"
            )
        if not np.isfinite(vec).all():
            raise ValueError("Vector contains NaN or Inf values")
        return vec

    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            self._load_embedder()
        return self._embedder.encode([text], normalize_embeddings=True)[0]

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbedderNotAvailableError(
                "sentence-transformers is not installed.\n"
                "Install it:  pip install thespacedb[embeddings]\n"
                "Or pass pre-computed np.ndarray vectors instead of strings."
            )
        log.info("Loading embedding model (all-MiniLM-L6-v2)…")
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def _resolve_personality(self, name_or_id: str) -> Optional[str]:
        """Resolve personality name to cluster_id."""
        for c in self._engine.clusters.personalities():
            if c.id == name_or_id or c.name == name_or_id:
                return c.id
        return name_or_id   # assume it's already an id

    def __repr__(self):
        s = self._engine.status()
        return (f"Space(name={self.name!r}, blocks={s['blocks']}, "
                f"clusters={s['clusters']}, personalities={s['personalities']})")
