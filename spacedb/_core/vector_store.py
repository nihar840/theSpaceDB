import os, pickle, threading, logging
import numpy as np
from ._exceptions import BlockNotFoundError, VectorDimensionError, StoreCorruptedError

logger = logging.getLogger("spacedb.vector_store")

_GROW = 512


class VectorStore:
    def __init__(self, path: str, dim: int):
        self._emb  = os.path.join(path, 'embeddings.npy')
        self._idx  = os.path.join(path, 'embeddings.idx')
        self._free_path = os.path.join(path, 'embeddings.free')
        self.dim   = dim
        self._map: dict[str, int] = {}
        self._free_rows: list[int] = []
        self._count = 0
        self._lock = threading.Lock()
        self._load()
        self._load_free_list()

    def put(self, block_id: str, vec: np.ndarray, flush: bool = True):
        with self._lock:
            if not block_id:
                raise ValueError("block_id must be non-empty")
            if not isinstance(vec, np.ndarray):
                raise TypeError("embedding must be a numpy ndarray")
            if vec.shape != (self.dim,):
                raise VectorDimensionError(
                    f"Expected shape ({self.dim},), got {vec.shape}"
                )
            if np.any(~np.isfinite(vec)):
                raise ValueError("embedding contains NaN or Inf")
            # Reuse freed row if available, otherwise grow
            if self._free_rows:
                row = self._free_rows.pop()
            else:
                self._grow_if_needed()
                row = self._count
                self._count += 1
            self._matrix[row] = vec.astype(np.float32)
            self._map[block_id] = row
            if flush:
                self._flush()

    def get(self, block_id: str) -> np.ndarray:
        with self._lock:
            if block_id not in self._map:
                raise BlockNotFoundError(f"No embedding for block: {block_id!r}")
            return self._matrix[self._map[block_id]].copy()

    def get_many(self, ids: list[str]) -> tuple[list[str], np.ndarray]:
        with self._lock:
            valid = [i for i in ids if i in self._map]
            if not valid:
                return [], np.empty((0, self.dim), dtype=np.float32)
            return valid, np.stack([self._matrix[self._map[i]] for i in valid])

    def get_all(self) -> tuple[list[str], np.ndarray]:
        with self._lock:
            if self._count == 0:
                return [], np.empty((0, self.dim), dtype=np.float32)
            ids = list(self._map.keys())
            return ids, np.stack([self._matrix[self._map[i]] for i in ids])

    def count(self) -> int:
        return self._count

    def alive_count(self) -> int:
        """Vectors minus freed rows."""
        return len(self._map)

    # -- deletion support -------------------------------------------------------

    def remove(self, block_id: str):
        """Zero out row and add to free list. O(1)."""
        with self._lock:
            if block_id not in self._map:
                return
            row = self._map.pop(block_id)
            self._matrix[row] = 0.0
            self._free_rows.append(row)
            self._flush()
            self._save_free_list()

    def compact(self):
        """Rewrite matrix without dead rows. Rebuilds map with new indices."""
        with self._lock:
            if not self._free_rows:
                return  # nothing to compact
            alive_ids = list(self._map.keys())
            if not alive_ids:
                # All vectors removed
                self._matrix = np.zeros((_GROW, self.dim), dtype=np.float32)
                self._map = {}
                self._free_rows = []
                self._count = 0
                self._flush()
                self._save_free_list()
                return

            new_matrix = np.zeros((len(alive_ids) + _GROW, self.dim), dtype=np.float32)
            new_map = {}
            for i, bid in enumerate(alive_ids):
                new_matrix[i] = self._matrix[self._map[bid]]
                new_map[bid] = i

            self._matrix = new_matrix
            self._map = new_map
            self._count = len(alive_ids)
            self._free_rows = []
            self._flush()
            self._save_free_list()
            logger.info("Compacted vector store: %d vectors remain", self._count)

    def _load_free_list(self):
        if os.path.exists(self._free_path):
            try:
                with open(self._free_path, 'rb') as f:
                    self._free_rows = pickle.load(f)
            except Exception as exc:
                logger.warning("Corrupt free list, resetting: %s", exc)
                self._free_rows = []

    def _save_free_list(self):
        with open(self._free_path, 'wb') as f:
            pickle.dump(self._free_rows, f)

    def _grow_if_needed(self):
        if self._count >= self._matrix.shape[0]:
            grown = np.zeros((self._matrix.shape[0] + _GROW, self.dim), dtype=np.float32)
            grown[:self._count] = self._matrix[:self._count]
            self._matrix = grown

    def _flush(self):
        np.save(self._emb, self._matrix)
        with open(self._idx, 'wb') as f:
            pickle.dump({'map': self._map, 'count': self._count}, f)

    def _load(self):
        if os.path.exists(self._idx):
            try:
                with open(self._idx, 'rb') as f:
                    d = pickle.load(f)
                    self._map, self._count = d['map'], d['count']
            except (pickle.UnpicklingError, EOFError, KeyError, Exception) as exc:
                raise StoreCorruptedError(
                    f"Corrupt vector index: {exc}"
                ) from exc
        cap = max(self._count + _GROW, _GROW)
        if os.path.exists(self._emb):
            try:
                self._matrix = np.load(self._emb)
            except Exception as exc:
                raise StoreCorruptedError(
                    f"Corrupt embeddings file: {exc}"
                ) from exc
        else:
            self._matrix = np.zeros((cap, self.dim), dtype=np.float32)
            np.save(self._emb, self._matrix)
