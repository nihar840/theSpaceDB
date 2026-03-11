"""
engine.py — SpaceEngine: raw internal engine (takes embeddings directly)
The public Space class wraps this with auto-embedding + clean API.
"""

import os, math, time, random, threading, logging
from typing import Optional
import numpy as np

from .models           import MemoryBlock, ClusterData
from .block_store      import BlockStore
from .vector_store     import VectorStore
from .distance_engine  import DistanceEngine
from .graph_store      import GraphStore
from .cluster_registry import ClusterRegistry
from ._exceptions      import BlockNotFoundError, VectorDimensionError, StoreCorruptedError
from ._version         import DATA_FORMAT_VERSION, VERSION_FILE

logger = logging.getLogger("spacedb.engine")


class SpaceEngine:

    CONNECT_K          = 5
    DRIFT_NUDGE        = 0.0005
    PASSIVE_DECAY      = 0.0001
    DECAY_SAMPLE       = 20
    PRUNE_THRESH       = 0.01
    AUTO_CLUSTER_EVERY = 50
    CROSS_LINK         = 0.002
    R_MIN, R_MAX, KAPPA = 0.05, 0.95, 0.003

    def __init__(self, path: str, dim: int = 384):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.dim  = dim

        vpath = os.path.join(path, VERSION_FILE)
        if os.path.exists(vpath):
            with open(vpath) as f:
                stored = int(f.read().strip())
            if stored != DATA_FORMAT_VERSION:
                raise StoreCorruptedError(
                    f"Data format v{stored} is incompatible with "
                    f"current v{DATA_FORMAT_VERSION}. Migration required.")
        else:
            with open(vpath, 'w') as f:
                f.write(str(DATA_FORMAT_VERSION))

        self._blocks   = BlockStore(path)
        self._vectors  = VectorStore(path, dim)
        self._dist     = DistanceEngine(path, dim)
        self._graph    = GraphStore(path)
        self._clusters = ClusterRegistry(path)

        self._last_input = time.time()
        self._ingest_since_cluster = 0
        self._drift_running = False
        self._drift_thread: Optional[threading.Thread] = None

    # ── ingest ───────────────────────────────────────────────
    def ingest(self, token: str, vec: np.ndarray,
               sensory_type: str = 'text',
               **block_kwargs) -> MemoryBlock:
        if not token:
            raise ValueError("token must be non-empty")
        if not isinstance(vec, np.ndarray) or vec.shape != (self.dim,):
            raise VectorDimensionError(
                f"Expected ndarray shape ({self.dim},), got {type(vec).__name__} "
                f"{getattr(vec, 'shape', '?')}")
        block = MemoryBlock.create(token, sensory_type, **block_kwargs)
        self._blocks.append(block)
        self._vectors.put(block.id, vec)
        self._connect(block.id, vec)
        self._passive_decay()
        self._graph.flush()
        self._dist.record_input()
        self._clusters.record_experience()
        self._last_input = time.time()
        self._ingest_since_cluster += 1
        if self._ingest_since_cluster >= self.AUTO_CLUSTER_EVERY:
            self._auto_cluster()
            self._ingest_since_cluster = 0
            self._log_event("auto_cluster")
        if self._blocks.count() % 10 == 0:
            self._log_event("ingest")
        return block

    def _connect(self, bid: str, vec: np.ndarray):
        ids, mat = self._vectors.get_all()
        if len(ids) <= 1: return
        dists = self._dist.distance_batch(vec, mat)
        pairs = sorted([(ids[i], dists[i]) for i in range(len(ids)) if ids[i] != bid],
                       key=lambda x: x[1])
        for nid, d in pairs[:self.CONNECT_K]:
            self._graph.add_edge(bid, nid, weight=1.0 / (d + 1e-8))

    # ── passive decay (forgetting) ──────────────────────────
    def _passive_decay(self):
        """Weaken random unreinforced edges. Rate ∝ input_rate (capped)."""
        edges = self._graph.random_edges(self.DECAY_SAMPLE)
        if not edges:
            return
        # Cap effective rate to prevent runaway decay during bursts
        rate = min(max(self._dist.input_rate, 0.01), 1.0)
        strength = self.PASSIVE_DECAY * rate
        for a, b, w in edges:
            delta = strength * 5
            self._graph.update_weight(a, b, -delta)
            if w - delta < self.PRUNE_THRESH:
                self._graph.remove_edge(a, b)
            try:
                v1, v2 = self._vectors.get(a), self._vectors.get(b)
                self._dist.decay(v1, v2, strength)
            except KeyError:
                pass

    # ── query ────────────────────────────────────────────────
    def query(self, vec: np.ndarray, time_budget_ms: int = 100,
              personality_id: Optional[str] = None,
              limit: int = 20) -> list[tuple[MemoryBlock, float]]:
        if time_budget_ms <= 0:
            raise ValueError("time_budget_ms must be positive")
        if limit <= 0:
            raise ValueError("limit must be positive")
        radius = self._radius(time_budget_ms)
        ids, mat = self._vectors.get_all()
        if not ids: return []

        dists = self._dist.distance_batch(vec, mat)
        now   = time.time()
        scored = []

        for i, bid in enumerate(ids):
            if dists[i] > radius: continue
            b = self._blocks.read(bid)
            if not b: continue
            recency  = math.exp(-0.001 * (now - b.timestamp))
            p_bias   = 1.5 if personality_id and b.cluster_id == personality_id else 1.0
            score    = (1.0 / (dists[i] + 1e-8)) * b.reinforcement_score * recency * p_bias
            scored.append((b, score))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    def _radius(self, ms: int) -> float:
        return self.R_MIN + (self.R_MAX - self.R_MIN) * (1 - math.exp(-self.KAPPA * ms))

    # ── reinforcement ────────────────────────────────────────
    def reinforce(self, b1_id: str, b2_id: str, strength: float = 0.01):
        try:
            v1 = self._vectors.get(b1_id)
            v2 = self._vectors.get(b2_id)
        except (KeyError, BlockNotFoundError) as e:
            raise BlockNotFoundError(f"Cannot reinforce: {e}") from e
        self._dist.reinforce(v1, v2, strength)
        self._graph.update_weight(b1_id, b2_id, strength * 10)
        self._bump(b1_id, strength); self._bump(b2_id, strength)
        self._refresh_cluster_for_block(b1_id)
        self._refresh_cluster_for_block(b2_id)

    def decay(self, b1_id: str, b2_id: str, strength: float = 0.001):
        try:
            v1 = self._vectors.get(b1_id)
            v2 = self._vectors.get(b2_id)
        except (KeyError, BlockNotFoundError) as e:
            raise BlockNotFoundError(f"Cannot reinforce: {e}") from e
        self._dist.decay(v1, v2, strength)
        self._graph.update_weight(b1_id, b2_id, -strength * 5)
        self._refresh_cluster_for_block(b1_id)
        self._refresh_cluster_for_block(b2_id)

    # ── cross-personality dynamics ───────────────────────────
    def cross_personality_link(self, block_ids: list[str]):
        """
        If blocks span multiple personalities, gently reinforce
        cross-personality connections. Called after query returns
        results from 2+ different personalities.
        """
        personality_blocks: dict[str, list[str]] = {}
        for bid in block_ids:
            b = self._blocks.read(bid)
            if b and b.cluster_id:
                c = self._clusters.get(b.cluster_id)
                if c and c.is_personality:
                    personality_blocks.setdefault(b.cluster_id, []).append(bid)
        pids = list(personality_blocks.keys())
        if len(pids) < 2:
            return
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                b1 = random.choice(personality_blocks[pids[i]])
                b2 = random.choice(personality_blocks[pids[j]])
                try:
                    self.reinforce(b1, b2, self.CROSS_LINK)
                except BlockNotFoundError:
                    pass

    def _bump(self, bid: str, delta: float):
        b = self._blocks.read(bid)
        if b:
            b.reinforcement_score = min(b.reinforcement_score + delta, 10.0)
            self._blocks.update(b)

    # ── clusters ─────────────────────────────────────────────
    def register_cluster(self, block_ids: list[str],
                         name: Optional[str] = None) -> str:
        cid = self._clusters.register(block_ids, name)
        for bid in block_ids:
            b = self._blocks.read(bid)
            if b: b.cluster_id = cid; self._blocks.update(b)
        self.compute_spirit(cid)
        return cid

    def _auto_cluster(self):
        """Run density-based clustering on learned W-distances."""
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError:
            logger.debug("scikit-learn not installed, skipping auto-cluster")
            return
        ids, mat = self._vectors.get_all()
        if len(ids) < 6:
            return
        n = len(ids)
        dist_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = self._dist.distance(mat[i], mat[j])
                dist_mat[i][j] = dist_mat[j][i] = d
        clusterer = HDBSCAN(min_cluster_size=3, min_samples=2,
                            metric='precomputed')
        labels = clusterer.fit_predict(dist_mat)
        label_map: dict[int, list[str]] = {}
        for i, label in enumerate(labels):
            if label < 0:
                continue
            label_map.setdefault(label, []).append(ids[i])
        existing = {frozenset(c.block_ids): c for c in self._clusters.all()}
        for label, block_ids in label_map.items():
            new_set = frozenset(block_ids)
            if new_set in existing:
                continue
            already_clustered = any(
                self._blocks.read(bid) and self._blocks.read(bid).cluster_id
                for bid in block_ids
            )
            if already_clustered:
                continue
            cid = self._clusters.register(block_ids)
            for bid in block_ids:
                b = self._blocks.read(bid)
                if b:
                    b.cluster_id = cid
                    self._blocks.update(b)
            self.compute_spirit(cid)
        logger.debug("Auto-cluster: %d groups found, %d blocks",
                     len(label_map), len(ids))

    def _refresh_cluster_for_block(self, bid: str):
        b = self._blocks.read(bid)
        if b and b.cluster_id:
            self.compute_spirit(b.cluster_id)

    def compute_spirit(self, cid: str) -> float:
        c = self._clusters.get(cid)
        if not c or len(c.block_ids) < 2:
            self._clusters.update_spirit(cid, 0.0); return 0.0
        ids, mat = self._vectors.get_many(c.block_ids)
        if len(ids) < 2: return 0.0
        total, pairs = 0.0, 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                d   = self._dist.distance(mat[i], mat[j])
                b   = self._blocks.read(ids[i])
                r   = b.reinforcement_score if b else 1.0
                total += r / (d + 1e-8); pairs += 1
        spirit = (total / pairs) * math.log1p(len(ids)) if pairs else 0.0
        self._clusters.update_spirit(cid, spirit)
        return spirit

    # ── drift ────────────────────────────────────────────────
    def start_drift(self, idle_s: float = 10.0, interval_s: float = 2.0):
        if self._drift_running:
            return
        self._drift_running = True

        def _loop():
            while self._drift_running:
                time.sleep(interval_s)
                if time.time() - self._last_input >= idle_s:
                    self._drift_step()

        self._drift_thread = threading.Thread(
            target=_loop,
            name="spacedb-drift",
            daemon=True,
        )
        self._drift_thread.start()

    def stop_drift(self):
        self._drift_running = False
        if self._drift_thread:
            self._drift_thread.join(timeout=1.0)
            self._drift_thread = None

    def _drift_step(self):
        try:
            clusters = self._clusters.all()
            if not clusters: return
            weights = [max(c.spirit_size, 0.01) for c in clusters]
            seed    = random.choices(clusters, weights=weights)[0]
            if not seed.block_ids: return
            path = self._graph.random_walk(random.choice(seed.block_ids),
                                           random.randint(3, 12))
            for i in range(len(path) - 1):
                try:
                    v1 = self._vectors.get(path[i])
                    v2 = self._vectors.get(path[i + 1])
                    self._dist.reinforce(v1, v2, self.DRIFT_NUDGE)
                except KeyError:
                    continue
                b2 = self._blocks.read(path[i + 1])
                if b2 and b2.cluster_id and b2.cluster_id != seed.id:
                    foreign = self._clusters.get(b2.cluster_id)
                    if foreign:
                        if foreign.spirit_size > seed.spirit_size * 0.3:
                            self._graph.add_edge(path[i], b2.id, 0.3)
                        else:
                            self._clusters.remove_block(foreign.id, b2.id)
                            self._clusters.add_block(seed.id, b2.id)
                            b2.cluster_id = seed.id; self._blocks.update(b2)
                            self.compute_spirit(seed.id)
                            self.compute_spirit(foreign.id)
            # Drift decay: randomly weaken some edges ("thinking also forgets")
            decay_edges = self._graph.random_edges(random.randint(2, 5))
            for a, b, w in decay_edges:
                try:
                    va, vb = self._vectors.get(a), self._vectors.get(b)
                    self._dist.decay(va, vb, self.DRIFT_NUDGE * 0.5)
                except KeyError:
                    pass
                self._graph.update_weight(a, b, -self.DRIFT_NUDGE * 5)
                if w - self.DRIFT_NUDGE * 5 < self.PRUNE_THRESH:
                    self._graph.remove_edge(a, b)
            self._graph.flush()
        except Exception as e:
            logger.warning("Drift step failed: %s", e)

    # ── evolution log ──────────────────────────────────────────
    def _log_event(self, event_type: str):
        """Append an evolution event to the log."""
        entry = {
            "t": round(time.time(), 3),
            "event": event_type,
            "blocks": self._blocks.count(),
            "edges": self._graph.edge_count(),
            "clusters": self._clusters.count(),
            "personalities": len(self._clusters.personalities()),
            "input_rate": round(self._dist.input_rate, 4),
            "w_updates": self._dist.update_count,
        }
        evo_path = os.path.join(self.path, "evolution.log")
        try:
            import json
            with open(evo_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def evolution_trajectory(self) -> list[dict]:
        """Read the evolution log. Returns list of event dicts."""
        evo_path = os.path.join(self.path, "evolution.log")
        if not os.path.exists(evo_path):
            return []
        import json
        events = []
        try:
            with open(evo_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            pass
        return events

    # ── storage ───────────────────────────────────────────────
    def storage_bytes(self) -> int:
        """Total bytes used by all data files in this space."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(self.path):
            for f in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
        return total

    # ── status ───────────────────────────────────────────────
    def status(self) -> dict:
        size_bytes = self.storage_bytes()
        return {
            'blocks':        self._blocks.count(),
            'graph_nodes':   self._graph.node_count(),
            'graph_edges':   self._graph.edge_count(),
            'clusters':      self._clusters.count(),
            'personalities': len(self._clusters.personalities()),
            'threshold':     round(self._clusters.threshold, 4),
            'input_rate':    self._dist.input_rate,
            'w_updates':     self._dist.update_count,
            'drift':         self._drift_running,
            'idle_s':        round(time.time() - self._last_input, 1),
            'storage_bytes': size_bytes,
            'storage_mb':    round(size_bytes / (1024 * 1024), 2),
        }

    # expose internals for query builder / drift controller
    @property
    def clusters(self): return self._clusters

    @property
    def blocks(self): return self._blocks

    @property
    def graph(self): return self._graph

    @property
    def dist(self): return self._dist
