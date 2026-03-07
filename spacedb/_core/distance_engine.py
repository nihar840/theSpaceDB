import os, time
import numpy as np

_CHECKPOINT_EVERY = 50
_EIGVAL_FLOOR     = 1e-8


class DistanceEngine:
    """
    Learned Mahalanobis distance via W matrix.
    W starts as identity. Evolves through reinforce() / decay().
    Rate of separation ∝ input_rate  (core SpaceDB property).
    """

    def __init__(self, path: str, dim: int):
        self._wpath  = os.path.join(path, 'w_matrix.npy')
        self._mpath  = os.path.join(path, 'w_meta.npy')
        self.dim     = dim
        self.W       = np.eye(dim, dtype=np.float64)
        self._rate   = 0.0
        self._last_t = time.time()
        self._n_inp  = 0
        self._n_upd  = 0
        self._load()

    # ── distance ─────────────────────────────────────────────
    def distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        d = v1.astype(np.float64) - v2.astype(np.float64)
        return float(np.sqrt(max(d @ self.W @ d, 0.0)))

    def distance_batch(self, query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        diff  = matrix.astype(np.float64) - query.astype(np.float64)
        Wd    = diff @ self.W
        return np.sqrt(np.maximum(np.einsum('ij,ij->i', diff, Wd), 0.0))

    def transform(self, v: np.ndarray) -> np.ndarray:
        return (self.W @ v.astype(np.float64)).astype(np.float32)

    # ── learning ─────────────────────────────────────────────
    def reinforce(self, v1: np.ndarray, v2: np.ndarray, strength: float = 0.01):
        d = v1.astype(np.float64) - v2.astype(np.float64)
        self.W -= strength * (1.0 + 0.1 * self._rate) * np.outer(d, d)
        self._after_update()

    def decay(self, v1: np.ndarray, v2: np.ndarray, strength: float = 0.001):
        d = v1.astype(np.float64) - v2.astype(np.float64)
        self.W += strength * (1.0 + self._rate) * np.outer(d, d)
        self._after_update()

    def record_input(self):
        now = time.time()
        dt  = now - self._last_t
        if dt > 0:
            self._rate = 0.9 * self._rate + 0.1 / dt
        self._last_t = now
        self._n_inp += 1

    @property
    def input_rate(self) -> float:
        return round(self._rate, 6)

    @property
    def update_count(self) -> int:
        return self._n_upd

    # ── internal ─────────────────────────────────────────────
    def _after_update(self):
        self._n_upd += 1
        if self._n_upd % _CHECKPOINT_EVERY == 0:
            self._project_psd()
        self._save()

    def _project_psd(self):
        self.W = (self.W + self.W.T) / 2.0
        ev, ec = np.linalg.eigh(self.W)
        self.W = ec @ np.diag(np.maximum(ev, _EIGVAL_FLOOR)) @ ec.T

    def _save(self):
        np.save(self._wpath, self.W)
        np.save(self._mpath, [self._rate, self._n_inp, self._n_upd, self._last_t])

    def _load(self):
        if os.path.exists(self._wpath):
            self.W = np.load(self._wpath)
        if os.path.exists(self._mpath):
            m = np.load(self._mpath)
            self._rate, self._n_inp, self._n_upd, self._last_t = (
                float(m[0]), int(m[1]), int(m[2]), float(m[3]))
