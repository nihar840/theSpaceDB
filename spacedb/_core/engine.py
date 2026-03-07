"""
engine.py — SpaceEngine: raw internal engine (takes embeddings directly)
The public Space class wraps this with auto-embedding + clean API.
"""

import os, math, time, random, asyncio
from typing import Optional
import numpy as np

from .models           import MemoryBlock, ClusterData
from .block_store      import BlockStore
from .vector_store     import VectorStore
from .distance_engine  import DistanceEngine
from .graph_store      import GraphStore
from .cluster_registry import ClusterRegistry


class SpaceEngine:

    CONNECT_K   = 5
    DRIFT_NUDGE = 0.0005
    R_MIN, R_MAX, KAPPA = 0.05, 0.95, 0.003

    def __init__(self, path: str, dim: int = 384):
        os.makedirs(path, exist_ok=True)
        self.path = path
        self.dim  = dim

        self._blocks   = BlockStore(path)
        self._vectors  = VectorStore(path, dim)
        self._dist     = DistanceEngine(path, dim)
        self._graph    = GraphStore(path)
        self._clusters = ClusterRegistry(path)

        self._last_input = time.time()
        self._drift_running = False
        self._drift_task: Optional[object] = None

    # ── ingest ───────────────────────────────────────────────
    def ingest(self, token: str, vec: np.ndarray,
               sensory_type: str = 'text') -> MemoryBlock:
        block = MemoryBlock.create(token, sensory_type)
        self._blocks.append(block)
        self._vectors.put(block.id, vec)
        self._connect(block.id, vec)
        self._dist.record_input()
        self._clusters.record_experience()
        self._last_input = time.time()
        return block

    def _connect(self, bid: str, vec: np.ndarray):
        ids, mat = self._vectors.get_all()
        if len(ids) <= 1: return
        dists = self._dist.distance_batch(vec, mat)
        pairs = sorted([(ids[i], dists[i]) for i in range(len(ids)) if ids[i] != bid],
                       key=lambda x: x[1])
        for nid, d in pairs[:self.CONNECT_K]:
            self._graph.add_edge(bid, nid, weight=1.0 / (d + 1e-8))

    # ── query ────────────────────────────────────────────────
    def query(self, vec: np.ndarray, time_budget_ms: int = 100,
              personality_id: Optional[str] = None,
              limit: int = 20) -> list[tuple[MemoryBlock, float]]:
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
        v1, v2 = self._vectors.get(b1_id), self._vectors.get(b2_id)
        self._dist.reinforce(v1, v2, strength)
        self._graph.update_weight(b1_id, b2_id, strength * 10)
        self._bump(b1_id, strength); self._bump(b2_id, strength)

    def decay(self, b1_id: str, b2_id: str, strength: float = 0.001):
        v1, v2 = self._vectors.get(b1_id), self._vectors.get(b2_id)
        self._dist.decay(v1, v2, strength)
        self._graph.update_weight(b1_id, b2_id, -strength * 5)

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
        return cid

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
        if self._drift_running: return
        self._drift_running = True
        async def _loop():
            while self._drift_running:
                await asyncio.sleep(interval_s)
                if time.time() - self._last_input >= idle_s:
                    await self._drift_step()
        loop = asyncio.get_event_loop()
        self._drift_task = loop.create_task(_loop())

    def stop_drift(self):
        self._drift_running = False
        if self._drift_task: self._drift_task.cancel(); self._drift_task = None

    async def _drift_step(self):
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
                        self._graph.add_edge(seed.id, foreign.id, 0.3)
                    else:
                        self._clusters.remove_block(foreign.id, b2.id)
                        self._clusters.add_block(seed.id, b2.id)
                        b2.cluster_id = seed.id; self._blocks.update(b2)

    # ── status ───────────────────────────────────────────────
    def status(self) -> dict:
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
        }

    # expose internals for query builder / drift controller
    @property def clusters(self): return self._clusters
    @property def blocks(self):   return self._blocks
    @property def graph(self):    return self._graph
    @property def dist(self):     return self._dist
