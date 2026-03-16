"""
engine.py — SpaceEngine: raw internal engine (takes embeddings directly)
The public Space class wraps this with auto-embedding + clean API.
"""

import os, math, time, random, threading, logging
from typing import Optional
import numpy as np

from .models           import MemoryBlock, ClusterData, QueryConfidence
from .block_store      import BlockStore
from .vector_store     import VectorStore
from .distance_engine  import DistanceEngine
from .graph_store      import GraphStore
from .cluster_registry import ClusterRegistry
from .pantheon         import Pantheon
from .driver           import Driver
from .adaptive         import AdaptiveParams, SystemState
from .cosmic           import CosmicLimits, CosmicReaper, UNLIMITED
from ._exceptions      import (BlockNotFoundError, VectorDimensionError,
                               StoreCorruptedError, CapacityExhaustedError)
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
    ROUTE_TOP_K        = 3       # God Points to probe per query
    R_MIN, R_MAX, KAPPA = 0.05, 0.95, 0.003

    def __init__(self, path: str, dim: int = 384,
                 cosmic_limits: Optional[CosmicLimits] = None):
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
        self._pantheon = Pantheon(path, dim)

        self._last_input = time.time()
        self._ingest_since_cluster = 0
        self._drift_running = False
        self._drift_thread: Optional[threading.Thread] = None

        # Adaptive parameter engine — self-tuning constants
        self._adaptive = AdaptiveParams()

        # Cosmic carrying capacity — 33 crore architecture
        self._cosmic_limits = cosmic_limits or UNLIMITED
        self._reaper = CosmicReaper(self, self._cosmic_limits)

        # Dream engine — deep thinking during idle (wired into drift)
        from .dream import DreamEngine
        self._dream = DreamEngine(self)

        # The Driver — executive self
        self._driver = Driver(self)
        self._clusters._on_personality_change = self._on_personality_change

    # ── bulk ingest mode ─────────────────────────────────────
    _bulk_mode = False

    def begin_bulk(self):
        """Enter bulk ingest mode: defer clustering, passive decay, and
        reduce disk flushes for much faster batch ingestion."""
        self._bulk_mode = True
        self._bulk_count = 0

    def end_bulk(self):
        """Exit bulk ingest mode: flush everything, run auto-cluster."""
        self._bulk_mode = False
        self._graph.flush()
        self._vectors._flush()
        if self._ingest_since_cluster > 0:
            self._auto_cluster()
            self._ingest_since_cluster = 0
            self._log_event("auto_cluster")
        self._log_event("bulk_complete")

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
        # Cosmic capacity check: reap if at limit
        if self._reaper.should_reap():
            self._reaper.reap()
            if self._blocks.alive_count() >= self._cosmic_limits.max_blocks:
                raise CapacityExhaustedError(
                    f"Block limit {self._cosmic_limits.max_blocks} reached after reaping")
        block = MemoryBlock.create(token, sensory_type, **block_kwargs)
        self._blocks.append(block)
        self._vectors.put(block.id, vec)
        # Route-aware connection: O(k) via God Point vs O(n) full scan
        if self._pantheon.count > 0:
            self._connect_routed(block.id, vec)
        else:
            self._connect(block.id, vec)
        self._passive_decay()
        self._graph.flush()
        self._dist.record_input()
        self._clusters.record_experience()
        self._last_input = time.time()
        self._ingest_since_cluster += 1
        # Respect explicit instance override; otherwise use adaptive
        if 'AUTO_CLUSTER_EVERY' in self.__dict__:
            cluster_every = self.__dict__['AUTO_CLUSTER_EVERY']
        else:
            cluster_every = self._adaptive.auto_cluster_every(self._system_state())
        if self._ingest_since_cluster >= cluster_every:
            self._auto_cluster()
            self._ingest_since_cluster = 0
            self._log_event("auto_cluster")
        if self._blocks.count() % 10 == 0:
            self._log_event("ingest")
        self._driver.on_ingest(block, vec)
        return block

    def ingest_fast(self, token: str, vec: np.ndarray,
                    sensory_type: str = 'text',
                    **block_kwargs) -> MemoryBlock:
        """Fast ingest for bulk operations. Skips passive decay, defers
        disk flush, and only connects every 5th block. Call end_bulk()
        when done to flush and cluster."""
        if not token:
            raise ValueError("token must be non-empty")
        if not isinstance(vec, np.ndarray) or vec.shape != (self.dim,):
            raise VectorDimensionError(
                f"Expected ndarray shape ({self.dim},), got {type(vec).__name__} "
                f"{getattr(vec, 'shape', '?')}")
        block = MemoryBlock.create(token, sensory_type, **block_kwargs)
        self._blocks.append(block)
        self._vectors.put(block.id, vec, flush=False)  # deferred flush
        self._bulk_count = getattr(self, '_bulk_count', 0) + 1
        # Connect only every 5th block (reduces overhead by 80%)
        if self._bulk_count % 5 == 0:
            if self._pantheon.count > 0:
                self._connect_routed(block.id, vec)
            else:
                self._connect(block.id, vec)
        self._dist.record_input()
        self._last_input = time.time()
        self._ingest_since_cluster += 1
        # Flush to disk every 20 blocks
        if self._bulk_count % 20 == 0:
            self._vectors._flush()
            self._graph.flush()
        return block

    def _connect(self, bid: str, vec: np.ndarray):
        ids, mat = self._vectors.get_all()
        if len(ids) <= 1: return
        dists = self._dist.distance_batch(vec, mat)
        pairs = sorted([(ids[i], dists[i]) for i in range(len(ids)) if ids[i] != bid],
                       key=lambda x: x[1])
        k = self._adaptive.connect_k(self._system_state())
        # Cosmic edge-per-block limit
        current_edges = self._graph.edge_count_for(bid)
        max_new = max(0, self._cosmic_limits.max_edges_per_block - current_edges)
        k = min(k, max_new)
        edge_limit = self._cosmic_limits.max_edges_per_block
        added = 0
        for nid, d in pairs:
            if added >= k:
                break
            # Also check neighbor's edge count
            if self._graph.edge_count_for(nid) >= edge_limit:
                continue
            self._graph.add_edge(bid, nid, weight=1.0 / (d + 1e-8))
            added += 1

    def _connect_routed(self, bid: str, vec: np.ndarray):
        """Connect a new block to neighbours within the nearest God's cell.

        O(k) where k = blocks in the nearest God's cluster,
        instead of O(n) full scan in ``_connect()``.
        Falls back to ``_connect()`` if routing fails.
        """
        god = self._pantheon.nearest_god(vec)
        if god is None:
            self._connect(bid, vec)
            return
        cluster = self._clusters.get(god.cluster_id)
        if not cluster or not cluster.block_ids:
            self._connect(bid, vec)
            return
        valid_ids, candidate_vecs = self._vectors.get_many(cluster.block_ids)
        if len(valid_ids) == 0:
            return
        dists = self._dist.distance_batch(vec, candidate_vecs)
        pairs = sorted(
            [(valid_ids[i], dists[i]) for i in range(len(valid_ids)) if valid_ids[i] != bid],
            key=lambda x: x[1],
        )
        k = self._adaptive.connect_k(self._system_state())
        # Cosmic edge-per-block limit
        current_edges = self._graph.edge_count_for(bid)
        max_new = max(0, self._cosmic_limits.max_edges_per_block - current_edges)
        k = min(k, max_new)
        edge_limit = self._cosmic_limits.max_edges_per_block
        added = 0
        for nid, d in pairs:
            if added >= k:
                break
            if self._graph.edge_count_for(nid) >= edge_limit:
                continue
            self._graph.add_edge(bid, nid, weight=1.0 / (d + 1e-8))
            added += 1

    # ── passive decay (forgetting) ──────────────────────────
    def _passive_decay(self):
        """Weaken random unreinforced edges. Rate ∝ input_rate (capped)."""
        edges = self._graph.random_edges(self.DECAY_SAMPLE)
        if not edges:
            return
        # Cap effective rate to prevent runaway decay during bursts
        rate = min(max(self._dist.input_rate, 0.01), 1.0)
        decay_base = self._adaptive.passive_decay(self._system_state())
        strength = decay_base * rate
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
        # Route through God Points when available (O(g+k) vs O(n))
        if self._pantheon.count > 0:
            return self._query_routed(vec, radius, personality_id, limit)
        return self._query_full_scan(vec, radius, personality_id, limit)

    def _query_full_scan(self, vec: np.ndarray, radius: float,
                         personality_id: Optional[str],
                         limit: int) -> list[tuple[MemoryBlock, float]]:
        """Original O(n) full-scan query. Used when no God Points exist."""
        ids, mat = self._vectors.get_all()
        if not ids:
            return []
        dists = self._dist.distance_batch(vec, mat)
        now = time.time()
        state = self._system_state()
        rec_decay = self._adaptive.recency_decay(state)
        p_bias_val = self._adaptive.personality_bias(state)
        scored = []
        for i, bid in enumerate(ids):
            if dists[i] > radius:
                continue
            b = self._blocks.read(bid)
            if not b:
                continue
            recency = math.exp(-rec_decay * (now - b.timestamp))
            p_bias = p_bias_val if personality_id and b.cluster_id == personality_id else 1.0
            score = (1.0 / (dists[i] + 1e-8)) * b.reinforcement_score * recency * p_bias
            scored.append((b, score))
        scored.sort(key=lambda x: -x[1])
        result = scored[:limit]
        self._driver.on_query(vec, result, personality_id)
        return result

    def _query_routed(self, vec: np.ndarray, radius: float,
                      personality_id: Optional[str],
                      limit: int) -> list[tuple[MemoryBlock, float]]:
        """O(g + k*top_k) routed query using God Points (Vishnu mode).

        Instead of scoring every block, we find the nearest God Points
        and search only within their cluster cells.
        """
        state = self._system_state()
        top_k = self._adaptive.route_top_k(state)
        gods = self._pantheon.route(vec, top_k=top_k)
        if not gods:
            return self._query_full_scan(vec, radius, personality_id, limit)
        # Gather candidate block IDs from the nearest Gods' clusters
        candidate_ids = []
        seen: set[str] = set()
        for god, _ in gods:
            cluster = self._clusters.get(god.cluster_id)
            if cluster:
                for bid in cluster.block_ids:
                    if bid not in seen:
                        candidate_ids.append(bid)
                        seen.add(bid)
        if not candidate_ids:
            return self._query_full_scan(vec, radius, personality_id, limit)
        valid_ids, candidate_vecs = self._vectors.get_many(candidate_ids)
        if len(valid_ids) == 0:
            return []
        # Score candidates using learned W matrix (same formula as full scan)
        dists = self._dist.distance_batch(vec, candidate_vecs)
        now = time.time()
        rec_decay = self._adaptive.recency_decay(state)
        p_bias_val = self._adaptive.personality_bias(state)
        scored = []
        for i, bid in enumerate(valid_ids):
            if dists[i] > radius:
                continue
            b = self._blocks.read(bid)
            if not b:
                continue
            recency = math.exp(-rec_decay * (now - b.timestamp))
            p_bias = p_bias_val if personality_id and b.cluster_id == personality_id else 1.0
            score = (1.0 / (dists[i] + 1e-8)) * b.reinforcement_score * recency * p_bias
            scored.append((b, score))
        scored.sort(key=lambda x: -x[1])
        result = scored[:limit]
        self._driver.on_query(vec, result, personality_id)
        return result

    def _radius(self, ms: int) -> float:
        return self.R_MIN + (self.R_MAX - self.R_MIN) * (1 - math.exp(-self.KAPPA * ms))

    def _system_state(self) -> SystemState:
        """Snapshot of current metrics for adaptive parameter computation."""
        alive = self._blocks.alive_count()
        max_blocks = self._cosmic_limits.max_blocks
        pressure = alive / max_blocks if max_blocks > 0 else 0.0
        return SystemState(
            block_count=alive,
            edge_count=self._graph.edge_count(),
            cluster_count=self._clusters.count(),
            personality_count=len(self._clusters.personalities()),
            god_count=self._pantheon.count,
            input_rate=self._dist.input_rate,
            w_updates=self._dist.update_count,
            capacity_pressure=min(pressure, 1.0),
        )

    def _compute_confidence(self, results: list[tuple], limit: int) -> QueryConfidence:
        """Compute confidence metrics for a query result set."""
        if not results:
            return QueryConfidence(
                score=0.0, hit_count=0, max_similarity=0.0,
                mean_similarity=0.0, coverage=0.0, sparse=True,
            )
        scores = [s for _, s in results]
        clusters_hit = len({b.cluster_id for b, _ in results if b.cluster_id})
        total_clusters = max(self._clusters.count(), 1)

        hit_ratio = min(len(results) / max(limit, 1), 1.0)
        max_sim = max(scores)
        mean_sim = sum(scores) / len(scores)
        coverage = clusters_hit / total_clusters

        # Composite: weighted combination
        conf = 0.4 * hit_ratio + 0.35 * min(max_sim / 100, 1.0) + 0.25 * coverage
        conf = max(0.0, min(1.0, conf))
        sparse = len(results) < max(2, limit // 3)

        return QueryConfidence(
            score=round(conf, 4),
            hit_count=len(results),
            max_similarity=round(max_sim, 4),
            mean_similarity=round(mean_sim, 4),
            coverage=round(coverage, 4),
            sparse=sparse,
        )

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

    # ── personality change callback ──────────────────────────
    def _on_personality_change(self, event_type: str, cid: str):
        """Called by ClusterRegistry when a personality is promoted/demoted."""
        self._driver.on_cluster_event(event_type, cid)

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
        # Cosmic cluster limit: skip if at max
        if self._clusters.count() >= self._cosmic_limits.max_clusters:
            logger.debug("Cluster limit %d reached, skipping auto-cluster",
                         self._cosmic_limits.max_clusters)
            return
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
        # Sync God Points with updated cluster state
        self._pantheon.sync_with_clusters(
            self._clusters.all(), self._vectors, self._blocks,
        )

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
            # Driver drift bias: boost weight of focused cluster
            focus_cid, focus_str = self._driver.get_drift_bias()
            if focus_cid:
                for i, c in enumerate(clusters):
                    if c.id == focus_cid:
                        weights[i] += focus_str * 5.0
                        break
            seed    = random.choices(clusters, weights=weights)[0]
            if not seed.block_ids: return
            path = self._graph.random_walk(random.choice(seed.block_ids),
                                           random.randint(3, 12))
            # Adaptive drift nudge strength
            nudge = self._adaptive.drift_nudge(self._system_state())
            for i in range(len(path) - 1):
                try:
                    v1 = self._vectors.get(path[i])
                    v2 = self._vectors.get(path[i + 1])
                    self._dist.reinforce(v1, v2, nudge)
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
                    self._dist.decay(va, vb, nudge * 0.5)
                except KeyError:
                    pass
                self._graph.update_weight(a, b, -nudge * 5)
                if w - nudge * 5 < self.PRUNE_THRESH:
                    self._graph.remove_edge(a, b)
            self._graph.flush()
            self._driver.on_drift_step(seed.id, path)
            # Shiva lifecycle: check God Points for dissolve/merge
            if self._pantheon.count > 0 and random.random() < 0.1:
                self._god_lifecycle_check()
            # Dream mode: 30% chance of deep thinking per drift step
            if hasattr(self, '_dream') and random.random() < 0.3:
                dream_result = self._dream.dream_step()
                if dream_result and dream_result.insights:
                    for insight_token in dream_result.insights:
                        try:
                            touched_vecs = []
                            for cid in dream_result.clusters_touched:
                                god = self._pantheon.get_by_cluster(cid)
                                if god:
                                    touched_vecs.append(god.centroid)
                            if touched_vecs:
                                insight_vec = np.mean(touched_vecs, axis=0).astype(np.float32)
                                norm = np.linalg.norm(insight_vec)
                                if norm > 0:
                                    insight_vec /= norm
                                    self.ingest(insight_token, insight_vec,
                                                sensory_type="dream")
                        except Exception as de:
                            logger.debug("Dream ingest failed: %s", de)
                    self._log_event("dream_" + dream_result.mode)
        except Exception as e:
            logger.warning("Drift step failed: %s", e)

    def _god_lifecycle_check(self):
        """Shiva lifecycle: dissolve weak Gods, merge close ones.
        Also enforces cosmic god limit (33 Koti)."""
        gods = self._pantheon.all()
        if not gods:
            return
        # Cosmic god limit: dissolve weakest gods if over limit
        if len(gods) > self._cosmic_limits.max_gods:
            gods_by_spirit = sorted(gods, key=lambda g: g.spirit)
            excess = len(gods) - self._cosmic_limits.max_gods
            for god in gods_by_spirit[:excess]:
                self._pantheon.dissolve(god.id)
                logger.debug("God %s dissolved (over cosmic limit)", god.id)
            gods = self._pantheon.all()
        threshold = self._clusters.threshold
        # Dissolve: Gods whose spirit is too low
        for god in gods:
            if god.spirit < threshold * self._pantheon.DISSOLVE_SPIRIT_RATIO:
                cluster = self._clusters.get(god.cluster_id)
                if not cluster or len(cluster.block_ids) < 2:
                    self._pantheon.dissolve(god.id)
                    logger.debug("God %s dissolved (spirit=%.3f)", god.id, god.spirit)
        # Merge: two Gods whose centroids are too close
        gods = self._pantheon.all()  # refresh after dissolves
        if len(gods) < 2:
            return
        for i in range(len(gods)):
            for j in range(i + 1, len(gods)):
                g1, g2 = gods[i], gods[j]
                dist = 1.0 - float(np.dot(g1.centroid, g2.centroid))
                if dist < self._pantheon.MIN_GOD_DISTANCE:
                    # Keep the one with higher spirit
                    if g1.spirit >= g2.spirit:
                        self._pantheon.dissolve(g2.id)
                        logger.debug("God %s merged into %s (dist=%.3f)",
                                     g2.id, g1.id, dist)
                    else:
                        self._pantheon.dissolve(g1.id)
                        logger.debug("God %s merged into %s (dist=%.3f)",
                                     g1.id, g2.id, dist)
                    return  # restart next drift step after structural change

    # ── block dissolution (cosmic lifecycle) ──────────────────
    def dissolve_block(self, block_id: str):
        """Full cascade deletion of a single block.

        Order matters:
        1. Remove from graph (all edges involving this block)
        2. Remove from cluster membership
        3. Remove from vector store (zero out row, free slot)
        4. Soft-delete from block store (tombstone)
        5. Log evolution event
        """
        # 1. Graph: remove all edges
        edges_removed = self._graph.remove_node(block_id)

        # 2. Cluster: remove from membership
        block = self._blocks.read(block_id)  # read before tombstoning
        if block and block.cluster_id:
            try:
                self._clusters.remove_block(block.cluster_id, block_id)
            except Exception:
                pass

        # 3. Vector: zero out and free row
        self._vectors.remove(block_id)

        # 4. Block: tombstone
        self._blocks.soft_delete(block_id)

        # 5. Evolution log
        import json
        entry = {
            "t": round(time.time(), 3),
            "event": "block_dissolved",
            "block_id": block_id,
            "edges_removed": edges_removed,
            "sensory_type": block.sensory_type if block else "unknown",
        }
        evo_path = os.path.join(self.path, "evolution.log")
        try:
            with open(evo_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

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
            "god_points": self._pantheon.count,
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
        alive = self._blocks.alive_count()
        return {
            'blocks':        self._blocks.count(),
            'blocks_alive':  alive,
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
            'god_points':    self._pantheon.count,
            'reaper_evictions': self._reaper._evictions,
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

    @property
    def driver(self): return self._driver

    @property
    def pantheon(self): return self._pantheon

    @property
    def dream(self): return self._dream

    @property
    def adaptive(self): return self._adaptive
