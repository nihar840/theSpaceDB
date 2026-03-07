"""
space.py — Space: the database-level object

Like a MongoDB database, a Space holds all blocks, clusters and
personalities for one named mind. Auto-embeds text so callers never
touch raw vectors.

Usage:
    mind = client["my_mind"]

    block  = mind.ingest("apple tastes sweet")
    result = mind.query("fruit").within(ms=200).limit(10).fetch()

    mind.reinforce(block_a, block_b)
    mind.clusters.all()
    mind.drift.start()
"""

from __future__ import annotations
import os
from typing import Optional, TYPE_CHECKING

import numpy as np

from ._core.engine import SpaceEngine
from .query        import QueryBuilder
from .drift        import DriftController

if TYPE_CHECKING:
    from ._core.models import MemoryBlock, ClusterData


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

    def __init__(self, name: str, base_path: str, dim: int = 384):
        self.name  = name
        self._path = os.path.join(base_path, name)
        os.makedirs(self._path, exist_ok=True)

        self._engine   = SpaceEngine(self._path, dim)
        self._embedder = None          # lazy-loaded
        self._dim      = dim

        self.drift    = DriftController(self._engine)
        self.clusters = ClusterView(self._engine)

    # ── ingest ───────────────────────────────────────────────
    def ingest(self, text: str, sensory_type: str = 'text') -> 'MemoryBlock':
        """
        Store a piece of experience. Text is auto-embedded.
        Returns the MemoryBlock — save the .id if you want to reinforce later.
        """
        vec = self._embed(text)
        return self._engine.ingest(text, vec, sensory_type)

    # ── query ────────────────────────────────────────────────
    def query(self, text: str) -> QueryBuilder:
        """
        Start a chainable query.

        Examples:
            space.query("fruit").fetch()
            space.query("fruit").within(ms=500).limit(5).fetch()
            space.query("fruit").as_personality("food").within(ms=300).fetch()
        """
        return QueryBuilder(self, text)

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

    # ── status ───────────────────────────────────────────────
    def status(self) -> dict:
        return {'space': self.name, **self._engine.status()}

    # ── internal ─────────────────────────────────────────────
    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            self._load_embedder()
        return self._embedder.encode([text], normalize_embeddings=True)[0]

    def _load_embedder(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers is required.\n"
                "Install: pip install sentence-transformers"
            )

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
