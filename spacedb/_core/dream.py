"""
dream.py — Dream Engine: deep thinking during idle

When the mind is idle, it doesn't just randomly walk — it *dreams*.
Dreams are structured explorations that build bridges between distant
ideas, detect contradictions, pursue goals, and synthesise insights.

Dream modes (one picked per drift cycle):
  1. BRIDGE_DISCOVERY  — Find connections between distant clusters
  2. CONTRADICTION_SCAN — Detect conflicting memories nearby but in
                          different clusters (cognitive dissonance)
  3. GOAL_DIRECTED     — Explore a specific topic (set by Driver or user)
  4. SYNTHESIS          — Create novel connection tokens (dream::link)

Each mode produces a DreamResult which may contain insight tokens
to be ingested as sensory_type="dream".
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .engine import SpaceEngine

log = logging.getLogger("spacedb.dream")


@dataclass
class DreamResult:
    """Output of one dream cycle."""
    mode: str                                          # bridge / contradiction / goal / synthesis
    path: list[str] = field(default_factory=list)      # block IDs traversed
    bridges_found: int = 0
    contradictions_found: int = 0
    insights: list[str] = field(default_factory=list)  # text tokens for new blocks
    clusters_touched: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class DreamEngine:
    """
    Enhanced cognitive drift with multiple dream modes.

    Called by SpaceEngine._drift_step() when dream mode activates
    (30% chance per drift step).

    The Dream Engine doesn't replace regular drift — it *augments* it.
    Regular drift does memory consolidation (random walk + reinforce).
    Dreams do exploration, contradiction detection, and synthesis.
    """

    # Mode probabilities (normalised at runtime)
    MODE_WEIGHTS = {
        "bridge": 0.35,
        "contradiction": 0.20,
        "goal": 0.25,
        "synthesis": 0.20,
    }

    BRIDGE_MAX_HOPS = 20             # max walk length for bridge discovery
    BRIDGE_REINFORCE = 0.0003        # strength when bridge path found
    CONTRADICTION_THRESHOLD = 0.15   # blocks this close but in different clusters
    CONTRADICTION_DECAY = 0.0002     # decay for contradictions
    GOAL_REINFORCE = 0.0002          # strength toward goal topic
    GOAL_MODE_BOOST = 0.50           # weight when goal is active

    def __init__(self, engine: "SpaceEngine"):
        self._engine = engine
        self._goal_topic_vec: Optional[np.ndarray] = None
        self._goal_topic_text: Optional[str] = None
        self._dream_count = 0

    def set_goal(self, topic_vec: np.ndarray, topic_text: str = ""):
        """Direct dreaming toward a specific topic."""
        self._goal_topic_vec = topic_vec
        self._goal_topic_text = topic_text
        log.info("Dream goal set: '%s'", topic_text[:60] if topic_text else "<vector>")

    def clear_goal(self):
        """Remove the dream goal."""
        self._goal_topic_vec = None
        self._goal_topic_text = None
        log.info("Dream goal cleared")

    # ── main entry point ──────────────────────────────────────

    def dream_step(self) -> Optional[DreamResult]:
        """Execute one dream cycle. Returns result or None if nothing interesting."""
        t0 = time.time()

        # Pick mode (weighted random, boost "goal" if a goal is set)
        weights = dict(self.MODE_WEIGHTS)
        if self._goal_topic_vec is not None:
            weights["goal"] = self.GOAL_MODE_BOOST
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        mode = random.choices(
            list(weights.keys()),
            weights=list(weights.values()),
        )[0]

        result = None
        if mode == "bridge":
            result = self._bridge_discovery()
        elif mode == "contradiction":
            result = self._contradiction_scan()
        elif mode == "goal":
            result = self._goal_directed()
        elif mode == "synthesis":
            result = self._synthesis()

        if result:
            result.duration_ms = (time.time() - t0) * 1000
            self._dream_count += 1

        return result

    # ── dream modes ───────────────────────────────────────────

    def _bridge_discovery(self) -> Optional[DreamResult]:
        """Find paths between distant clusters.

        Pick 2 random clusters, walk from one toward the other.
        If a bridge is found, reinforce the entire path.
        """
        clusters = self._engine._clusters.all()
        clusters_with_blocks = [c for c in clusters if c.block_ids]
        if len(clusters_with_blocks) < 2:
            return None

        c1, c2 = random.sample(clusters_with_blocks, 2)
        start = random.choice(c1.block_ids)
        target_set = set(c2.block_ids)

        path = [start]
        current = start
        found_bridge = False

        for _ in range(self.BRIDGE_MAX_HOPS):
            neighbors = self._engine._graph.neighbors(current)
            if not neighbors:
                break
            ids, weights = zip(*neighbors)
            current = random.choices(ids, weights=weights)[0]
            path.append(current)
            if current in target_set:
                found_bridge = True
                break

        if found_bridge and len(path) > 3:
            # Reinforce the entire bridge path
            for i in range(len(path) - 1):
                try:
                    v1 = self._engine._vectors.get(path[i])
                    v2 = self._engine._vectors.get(path[i + 1])
                    self._engine._dist.reinforce(v1, v2, self.BRIDGE_REINFORCE)
                except KeyError:
                    continue
            log.debug("Dream: bridge found %s -> %s (%d hops)", c1.id[:8], c2.id[:8], len(path))
            return DreamResult(
                mode="bridge",
                path=path,
                bridges_found=1,
                clusters_touched=[c1.id, c2.id],
            )
        return None

    def _contradiction_scan(self) -> Optional[DreamResult]:
        """Detect blocks very close in W-space but in different clusters.

        This is cognitive dissonance — similar ideas that ended up in
        different conceptual buckets. We slightly decay the W-matrix
        connection to signal "these should be reconsidered."
        """
        clusters = self._engine._clusters.all()
        clusters_with_blocks = [c for c in clusters if c.block_ids]
        if len(clusters_with_blocks) < 2:
            return None

        source = random.choice(clusters_with_blocks)
        sample_bid = random.choice(source.block_ids)
        try:
            sample_vec = self._engine._vectors.get(sample_bid)
        except KeyError:
            return None

        # Check a sample of blocks from OTHER clusters
        other_bids = []
        for c in clusters_with_blocks:
            if c.id != source.id:
                other_bids.extend(c.block_ids[:10])  # cap per cluster
        if not other_bids:
            return None

        # Sample to keep fast
        check_bids = random.sample(other_bids, min(30, len(other_bids)))
        valid_ids, mat = self._engine._vectors.get_many(check_bids)
        if len(valid_ids) == 0:
            return None

        dists = self._engine._dist.distance_batch(sample_vec, mat)
        contradictions = 0

        for i, bid in enumerate(valid_ids):
            if dists[i] < self.CONTRADICTION_THRESHOLD:
                contradictions += 1
                try:
                    self._engine._dist.decay(sample_vec, mat[i], self.CONTRADICTION_DECAY)
                except Exception:
                    pass

        if contradictions > 0:
            log.debug("Dream: %d contradictions near %s", contradictions, source.id[:8])
            return DreamResult(
                mode="contradiction",
                contradictions_found=contradictions,
                clusters_touched=[source.id],
            )
        return None

    def _goal_directed(self) -> Optional[DreamResult]:
        """Walk toward the set goal topic, reinforcing paths.

        If no goal is set, falls back to bridge discovery.
        """
        if self._goal_topic_vec is None:
            return self._bridge_discovery()

        # Find nearest blocks to the goal
        results = self._engine.query(
            self._goal_topic_vec, time_budget_ms=200, limit=5,
        )
        if not results:
            return None

        # Random walk starting from a goal-adjacent block
        start_block, _ = random.choice(results)
        path = self._engine._graph.random_walk(
            start_block.id, random.randint(5, 15),
        )

        # Reinforce each block on the path toward the goal
        for bid in path:
            try:
                v = self._engine._vectors.get(bid)
                self._engine._dist.reinforce(v, self._goal_topic_vec, self.GOAL_REINFORCE)
            except KeyError:
                continue

        clusters_touched = set()
        for bid in path:
            b = self._engine._blocks.read(bid)
            if b and b.cluster_id:
                clusters_touched.add(b.cluster_id)

        log.debug("Dream: goal-directed walk (%d hops, %d clusters)",
                   len(path), len(clusters_touched))
        return DreamResult(
            mode="goal",
            path=path,
            clusters_touched=list(clusters_touched),
        )

    def _synthesis(self) -> Optional[DreamResult]:
        """Find unexpected connections between topics.

        Picks blocks from 2 different clusters and generates a
        dream::link insight token connecting them. These get ingested
        as sensory_type="dream" to form dream memories.
        """
        clusters = self._engine._clusters.all()
        candidates = [c for c in clusters if len(c.block_ids) >= 2]
        if len(candidates) < 2:
            return None

        c1, c2 = random.sample(candidates, 2)
        b1_id = random.choice(c1.block_ids)
        b2_id = random.choice(c2.block_ids)

        block1 = self._engine._blocks.read(b1_id)
        block2 = self._engine._blocks.read(b2_id)
        if not block1 or not block2:
            return None

        t1 = (block1.normalized_content or block1.token)[:60]
        t2 = (block2.normalized_content or block2.token)[:60]
        insight_token = f"dream::link({t1} <-> {t2})"

        log.debug("Dream: synthesis link between %s and %s", c1.id[:8], c2.id[:8])
        return DreamResult(
            mode="synthesis",
            path=[b1_id, b2_id],
            insights=[insight_token],
            clusters_touched=[c1.id, c2.id],
        )

    # ── status ────────────────────────────────────────────────

    @property
    def dream_count(self) -> int:
        """Total dream cycles completed."""
        return self._dream_count

    def status(self) -> dict:
        """Current dream engine state."""
        return {
            "dream_count": self._dream_count,
            "has_goal": self._goal_topic_vec is not None,
            "goal_text": self._goal_topic_text or "",
        }
