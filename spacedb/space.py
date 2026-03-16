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
from ._core.cosmic       import CosmicLimits
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


class DriverView:
    """Public API for the Driver (the executive self)."""

    def __init__(self, engine: SpaceEngine):
        self._d = engine.driver

    def select_personality(self, vec, scores: dict) -> str | None:
        """Strategic personality selection. Returns personality id or None."""
        return self._d.select_personality(vec, scores)

    def feedback(self, outcome: str, score: float = 0.0):
        """Learn from outcome: 'positive', 'negative', or 'neutral'."""
        self._d.feedback(outcome, score)

    def direct_drift(self, cluster_id: str, strength: float = 1.0):
        """Steer drift to focus on a specific cluster."""
        self._d.direct_drift(cluster_id, strength)

    @property
    def active_personality(self) -> str | None:
        """Who's currently speaking."""
        return self._d.active_personality

    def history(self, limit: int = 50) -> list[dict]:
        """Recent activation records."""
        return self._d.history(limit)

    def affinity(self) -> dict[str, float]:
        """Learned preference weights per personality."""
        return self._d.affinity()

    def status(self) -> dict:
        """Overview of Driver state."""
        return self._d.status()


class PantheonView:
    """Read-only view of the God Point routing layer.

    God Points are cluster centroids that enable fast O(g + k) queries
    instead of O(n) full scans.  They emerge automatically when
    auto-clustering runs (every 50 blocks).

    Example::

        space.pantheon.count          # number of God Points
        space.pantheon.all()          # list of all God Points
        space.pantheon.route(vec, 3)  # find 3 nearest Gods
    """

    def __init__(self, engine: SpaceEngine):
        self._e = engine

    def all(self) -> list[dict]:
        """Return all God Points as dicts."""
        return [self._fmt(g) for g in self._e.pantheon.all()]

    def get(self, god_id: str) -> "Optional[dict]":
        """Get a single God Point by ID."""
        g = self._e.pantheon.get(god_id)
        return self._fmt(g) if g else None

    def route(self, vec: np.ndarray, top_k: int = 3) -> list[dict]:
        """Route a vector to the nearest God Points.

        Returns list of dicts with ``distance`` included, sorted ascending.
        """
        results = self._e.pantheon.route(vec, top_k=top_k)
        return [
            {**self._fmt(god), "distance": round(dist, 6)}
            for god, dist in results
        ]

    @property
    def count(self) -> int:
        """Number of God Points currently alive."""
        return self._e.pantheon.count

    def _fmt(self, g) -> dict:
        return {
            "id": g.id,
            "cluster_id": g.cluster_id,
            "block_count": g.block_count,
            "spirit": round(g.spirit, 4),
            "tier": g.tier,
        }


class Space:
    """
    A named cognitive space (analogous to a MongoDB database).
    Holds all memory blocks, clusters, and personalities for one mind.
    """

    def __init__(self, name: str, base_path: str, dim: int = 384,
                 max_size_mb: Optional[float] = None,
                 cosmic_limits: Optional[CosmicLimits] = None):
        self.name  = name
        self._path = os.path.join(base_path, name)
        os.makedirs(self._path, exist_ok=True)

        self._engine       = SpaceEngine(self._path, dim,
                                         cosmic_limits=cosmic_limits)
        self._embedder     = None          # lazy-loaded
        self._dim          = dim
        self._max_size_mb  = max_size_mb   # None = unlimited

        self.drift    = DriftController(self._engine)
        self.clusters = ClusterView(self._engine)
        self.driver   = DriverView(self._engine)
        self.pantheon = PantheonView(self._engine)

    # ── ingest ───────────────────────────────────────────────
    def ingest(
        self,
        content: Union[str, np.ndarray],
        sensory_type: str = "text",
        **memory_kwargs,
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
            return self._engine.ingest(
                token,
                vec,
                sensory_type,
                raw_input=memory_kwargs.pop("raw_input", None),
                normalized_content=memory_kwargs.pop("normalized_content", token),
                **memory_kwargs,
            )
        elif isinstance(content, str):
            vec = self._embed(content)
            return self._engine.ingest(
                content,
                vec,
                sensory_type,
                raw_input=memory_kwargs.pop("raw_input", content),
                normalized_content=memory_kwargs.pop("normalized_content", content),
                **memory_kwargs,
            )
        else:
            raise TypeError(
                f"content must be str or np.ndarray, got {type(content).__name__}"
            )

    # ── bulk ingest mode ─────────────────────────────────────
    def begin_bulk(self):
        """Enter bulk mode for fast batch ingestion.
        Defers clustering, reduces disk I/O, skips passive decay."""
        self._engine.begin_bulk()

    def end_bulk(self):
        """Exit bulk mode. Flushes all pending data and runs clustering."""
        self._engine.end_bulk()

    def ingest_fast(
        self,
        content: Union[str, np.ndarray],
        sensory_type: str = "text",
        **memory_kwargs,
    ) -> "MemoryBlock":
        """Fast ingest for bulk operations. Use between begin_bulk/end_bulk."""
        self._check_quota()
        if isinstance(content, np.ndarray):
            vec = self._validate_vector(content)
            token = f"<raw:{sensory_type}>"
            return self._engine.ingest_fast(
                token, vec, sensory_type,
                raw_input=memory_kwargs.pop("raw_input", None),
                normalized_content=memory_kwargs.pop("normalized_content", token),
                **memory_kwargs,
            )
        elif isinstance(content, str):
            vec = self._embed(content)
            return self._engine.ingest_fast(
                content, vec, sensory_type,
                raw_input=memory_kwargs.pop("raw_input", content),
                normalized_content=memory_kwargs.pop("normalized_content", content),
                **memory_kwargs,
            )
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

    # ── cosmic capacity ───────────────────────────────────────
    def cosmic_status(self) -> dict:
        """Current cosmic capacity usage.

        Returns dict with blocks/edges/gods/clusters/personalities usage
        and reaper statistics.
        """
        e = self._engine
        limits = e._cosmic_limits
        alive = e._blocks.alive_count()
        max_b = limits.max_blocks
        return {
            "blocks": {
                "alive": alive,
                "total": e._blocks.count(),
                "limit": max_b,
                "pressure": round(alive / max_b, 6) if max_b > 0 else 0.0,
            },
            "edges": {
                "total": e._graph.total_edge_count(),
                "limit": limits.max_total_edges,
            },
            "gods": {
                "count": e._pantheon.count,
                "limit": limits.max_gods,
            },
            "clusters": {
                "count": e._clusters.count(),
                "limit": limits.max_clusters,
            },
            "personalities": {
                "count": len(e._clusters.personalities()),
                "limit": limits.max_personalities,
            },
            "reaper": e._reaper.status(),
        }

    def compact(self):
        """Reclaim disk space by rewriting stores without dead entries.

        Call this after a batch of evictions to shrink on-disk files.
        This is an offline operation — do not ingest while compacting.
        """
        self._engine._blocks.compact()
        self._engine._vectors.compact()
        self._engine._graph.flush()

    def dissolve_block(self, block_id: str):
        """Manually dissolve a specific block (cascade delete)."""
        self._engine.dissolve_block(block_id)

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
