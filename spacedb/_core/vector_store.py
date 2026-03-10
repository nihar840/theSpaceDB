import os, pickle, threading, logging
import numpy as np
from ._exceptions import BlockNotFoundError, VectorDimensionError, StoreCorruptedError

logger = logging.getLogger("spacedb.vector_store")

_GROW = 512


class VectorStore:
    def __init__(self, path: str, dim: int):
        self._emb  = os.path.join(path, 'embeddings.npy')
        self._idx  = os.path.join(path, 'embeddings.idx')
        self.dim   = dim
        self._map: dict[str, int] = {}
        self._count = 0
        self._lock = threading.Lock()
        self._load()

    def put(self, block_id: str, vec: np.ndarray):
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
            self._grow_if_needed()
            self._matrix[self._count] = vec.astype(np.float32)
            self._map[block_id] = self._count
            self._count += 1
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
