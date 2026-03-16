"""
pantheon.py -- Pantheon: the God Point routing layer

God Points are routing anchors that form a coarse navigation layer
above individual memory blocks.  Each God Point holds the centroid
of a cluster, enabling O(g + k) queries instead of O(n) full scans.

How it works
------------
1. Auto-clustering (HDBSCAN) groups blocks into clusters.
2. ``sync_with_clusters`` creates a God Point for each cluster,
   storing its centroid (mean of member vectors, L2-normalised).
3. On **ingest** (Brahma), the nearest God Point routes the new
   block to its cluster for local connection — O(k) not O(n).
4. On **query** (Vishnu), the top-k nearest God Points narrow the
   search to a handful of clusters — O(g + k*top_k) not O(n).
5. On **drift** (Shiva), lifecycle checks dissolve weak Gods and
   merge Gods whose centroids are too close.

Cosmological mapping
--------------------
- GodPoint = Deva  — a routing anchor in the data space
- Pantheon = The registry of all Devas
- spawn    = Brahma creates a new Deva
- route    = Vishnu navigates through Devas
- dissolve = Shiva destroys a Deva
"""

import json
import logging
import os
import threading
import time
import uuid
from typing import Optional

import numpy as np

from .models import GodPoint, ClusterData
from ._exceptions import StoreCorruptedError

logger = logging.getLogger("spacedb.pantheon")


class Pantheon:
    """Registry of God Points — the routing layer of the data space.

    Parameters
    ----------
    path : str
        Directory where ``gods.json`` is persisted.
    dim : int
        Embedding dimensionality (must match the Space).
    """

    # ── lifecycle thresholds ──────────────────────────────────
    MAX_GOD_SIZE = 200            # blocks per God → split consideration
    MIN_GOD_DISTANCE = 0.15       # cosine distance below which Gods merge
    DISSOLVE_SPIRIT_RATIO = 0.3   # dissolve when spirit < threshold × this

    def __init__(self, path: str, dim: int):
        self._path = os.path.join(path, "gods.json")
        self._dim = dim
        self._gods: dict[str, GodPoint] = {}
        self._centroid_matrix: Optional[np.ndarray] = None   # (g, dim)
        self._god_ids: list[str] = []                        # row ↔ god_id
        self._lock = threading.Lock()
        self._load()

    # ── spawn (Brahma) ────────────────────────────────────────

    def spawn(
        self,
        cluster_id: str,
        centroid: np.ndarray,
        block_count: int,
        spirit: float,
    ) -> GodPoint:
        """Create a new God Point from a cluster centroid.

        The centroid is L2-normalised so that routing can use a
        simple dot-product for cosine similarity.
        """
        with self._lock:
            god_id = "god_" + str(uuid.uuid4())[:8]
            centroid = _normalise(centroid, self._dim)
            god = GodPoint(
                id=god_id,
                centroid=centroid,
                cluster_id=cluster_id,
                block_count=block_count,
                spirit=spirit,
            )
            self._gods[god_id] = god
            self._rebuild_centroid_matrix()
            self._save()
            logger.info(
                "God spawned: %s (cluster=%s, blocks=%d, spirit=%.3f)",
                god_id, cluster_id, block_count, spirit,
            )
            return god

    # ── route (Vishnu) ────────────────────────────────────────

    def route(
        self,
        vec: np.ndarray,
        top_k: int = 1,
    ) -> list[tuple[GodPoint, float]]:
        """Find the ``top_k`` nearest God Points by cosine distance.

        Returns
        -------
        list of (GodPoint, cosine_distance) sorted ascending.
        Cosine distance = 1 − dot(normalised_query, centroid).
        """
        with self._lock:
            if self._centroid_matrix is None or len(self._god_ids) == 0:
                return []

            query = vec.astype(np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm

            # (g, dim) @ (dim,) → (g,)  — cosine similarity in one shot
            similarities = self._centroid_matrix @ query
            distances = 1.0 - similarities

            k = min(top_k, len(self._god_ids))
            if k >= len(self._god_ids):
                indices = np.argsort(distances)
            else:
                indices = np.argpartition(distances, k)[:k]
                indices = indices[np.argsort(distances[indices])]

            return [
                (self._gods[self._god_ids[i]], float(distances[i]))
                for i in indices[:k]
            ]

    def nearest_god(self, vec: np.ndarray) -> Optional[GodPoint]:
        """Convenience: return the single nearest God Point, or None."""
        results = self.route(vec, top_k=1)
        return results[0][0] if results else None

    # ── update ────────────────────────────────────────────────

    def update(
        self,
        god_id: str,
        centroid: np.ndarray,
        block_count: int,
        spirit: float,
    ):
        """Update an existing God Point's centroid and statistics."""
        with self._lock:
            god = self._gods.get(god_id)
            if god is None:
                return
            god.centroid = _normalise(centroid, self._dim)
            god.block_count = block_count
            god.spirit = spirit
            god.updated_at = time.time()
            self._rebuild_centroid_matrix()
            self._save()

    # ── dissolve (Shiva) ──────────────────────────────────────

    def dissolve(self, god_id: str):
        """Remove a God Point from the Pantheon."""
        with self._lock:
            god = self._gods.pop(god_id, None)
            if god is not None:
                self._rebuild_centroid_matrix()
                self._save()
                logger.info("God dissolved: %s (cluster=%s)", god_id, god.cluster_id)

    # ── sync with cluster state ───────────────────────────────

    def sync_with_clusters(self, clusters: list, vector_store, block_store):
        """Synchronise God Points with the current cluster state.

        Called after ``_auto_cluster()`` runs.  For each cluster:

        * **New cluster** → spawn a God Point.
        * **Existing cluster** → update centroid and stats.
        * **Dead cluster** (no longer in the list) → dissolve its God.

        Parameters
        ----------
        clusters : list[ClusterData]
            Current cluster list from ClusterRegistry.
        vector_store : VectorStore
            For reading member vectors to compute centroids.
        block_store : BlockStore
            (reserved for future use)
        """
        with self._lock:
            cluster_ids = {c.id for c in clusters}
            god_by_cluster: dict[str, GodPoint] = {
                g.cluster_id: g for g in self._gods.values()
            }

            # Dissolve Gods whose clusters no longer exist
            to_dissolve = [
                gid for gid, g in self._gods.items()
                if g.cluster_id not in cluster_ids
            ]
            changed = False
            for gid in to_dissolve:
                self._gods.pop(gid, None)
                logger.info("God dissolved (cluster gone): %s", gid)
                changed = True

            # Spawn / update for each cluster
            for cluster in clusters:
                if not cluster.block_ids or len(cluster.block_ids) < 2:
                    continue

                valid_ids, vecs = vector_store.get_many(cluster.block_ids)
                if len(valid_ids) == 0:
                    continue

                centroid = _normalise(vecs.mean(axis=0), self._dim)

                if cluster.id in god_by_cluster:
                    # Update existing God
                    god = god_by_cluster[cluster.id]
                    god.centroid = centroid
                    god.block_count = len(cluster.block_ids)
                    god.spirit = cluster.spirit_size
                    god.updated_at = time.time()
                    changed = True
                else:
                    # Spawn new God
                    god_id = "god_" + str(uuid.uuid4())[:8]
                    god = GodPoint(
                        id=god_id,
                        centroid=centroid,
                        cluster_id=cluster.id,
                        block_count=len(cluster.block_ids),
                        spirit=cluster.spirit_size,
                    )
                    self._gods[god_id] = god
                    logger.info(
                        "God spawned via sync: %s (cluster=%s, blocks=%d)",
                        god_id, cluster.id, len(cluster.block_ids),
                    )
                    changed = True

            if changed:
                self._rebuild_centroid_matrix()
                self._save()

    # ── introspection ─────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of God Points currently alive."""
        return len(self._gods)

    def all(self) -> list[GodPoint]:
        """Return all God Points (snapshot)."""
        return list(self._gods.values())

    def get(self, god_id: str) -> Optional[GodPoint]:
        """Look up a God Point by ID."""
        return self._gods.get(god_id)

    def get_by_cluster(self, cluster_id: str) -> Optional[GodPoint]:
        """Find the God Point linked to a given cluster."""
        for g in self._gods.values():
            if g.cluster_id == cluster_id:
                return g
        return None

    # ── internal helpers ──────────────────────────────────────

    def _rebuild_centroid_matrix(self):
        """(Re)build the (g, dim) numpy matrix for fast routing."""
        if not self._gods:
            self._centroid_matrix = None
            self._god_ids = []
            return
        self._god_ids = list(self._gods.keys())
        self._centroid_matrix = np.stack(
            [self._gods[gid].centroid for gid in self._god_ids]
        ).astype(np.float32)

    # ── persistence ───────────────────────────────────────────

    def _save(self):
        """Persist to gods.json."""
        data = {
            "gods": {
                gid: {
                    "id": g.id,
                    "centroid": g.centroid.tolist(),
                    "cluster_id": g.cluster_id,
                    "block_count": g.block_count,
                    "spirit": g.spirit,
                    "tier": g.tier,
                    "parent_id": g.parent_id,
                    "children_ids": g.children_ids,
                    "created_at": g.created_at,
                    "updated_at": g.updated_at,
                }
                for gid, g in self._gods.items()
            }
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as exc:
            logger.warning("Failed to save Pantheon: %s", exc)

    def _load(self):
        """Load from gods.json (missing file → empty Pantheon)."""
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            for gid, gd in data.get("gods", {}).items():
                self._gods[gid] = GodPoint(
                    id=gd["id"],
                    centroid=np.array(gd["centroid"], dtype=np.float32),
                    cluster_id=gd["cluster_id"],
                    block_count=gd["block_count"],
                    spirit=gd["spirit"],
                    tier=gd.get("tier", 0),
                    parent_id=gd.get("parent_id"),
                    children_ids=gd.get("children_ids", []),
                    created_at=gd.get("created_at", time.time()),
                    updated_at=gd.get("updated_at", time.time()),
                )
            self._rebuild_centroid_matrix()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise StoreCorruptedError(
                f"Failed to load Pantheon: {exc}"
            ) from exc


# ── module-level helpers ──────────────────────────────────────

def _normalise(vec: np.ndarray, dim: int) -> np.ndarray:
    """L2-normalise a vector to unit length (float32)."""
    v = vec.astype(np.float32)
    if v.shape == (1, dim):
        v = v.squeeze()
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v
