"""
cosmic.py -- Cosmic Carrying Capacity for SpaceDB.

Inspired by the 33 crore devtas of Hindu cosmology.
Everything has a limit. When full, old memories naturally dissolve
to make room for new ones. Not garbage collection -- natural death
and rebirth.

Contains:
  CosmicLimits   -- configurable capacity limits
  block_importance()  -- importance scoring for eviction
  CosmicReaper   -- sample-based eviction engine (tournament selection)
"""

import math
import random
import time
import logging
from dataclasses import dataclass

log = logging.getLogger("spacedb.cosmic")


# ---------------------------------------------------------------------------
# Cosmic Limits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CosmicLimits:
    """Carrying capacity for a Space. Inspired by 33 crore devtas.

    All limits are configurable. Use presets for common configurations:
      THIRTY_THREE_CRORE  -- production default (330M blocks)
      SMALL_SPACE         -- testing (1000 blocks)
      UNLIMITED           -- backward compatible (no limits)
    """

    max_blocks: int = 330_000_000           # 33 crore
    max_gods: int = 33                      # the 33 Koti
    max_edges_per_block: int = 33           # selective connections
    max_total_edges: int = 330_000_000      # cosmic symmetry
    max_clusters: int = 3_333               # 33 x ~100
    max_personalities: int = 33             # one per God

    # Reaper config
    reaper_sample_size: int = 64            # blocks sampled per eviction round
    reaper_evict_batch: int = 10            # blocks evicted per round
    reaper_min_age_seconds: float = 3600.0  # don't evict blocks < 1 hour old

    # Importance weights (for eviction scoring)
    w_reinforcement: float = 0.35
    w_recency: float = 0.30
    w_connectivity: float = 0.20
    w_spirit: float = 0.15


# -- Presets -----------------------------------------------------------------

THIRTY_THREE_CRORE = CosmicLimits()  # production default

SMALL_SPACE = CosmicLimits(
    max_blocks=1000,
    max_gods=5,
    max_edges_per_block=10,
    max_total_edges=5000,
    max_clusters=50,
    max_personalities=5,
    reaper_sample_size=16,
    reaper_evict_batch=3,
    reaper_min_age_seconds=0.0,
)

UNLIMITED = CosmicLimits(
    max_blocks=2**63,
    max_gods=2**63,
    max_edges_per_block=2**63,
    max_total_edges=2**63,
    max_clusters=2**63,
    max_personalities=2**63,
)


# ---------------------------------------------------------------------------
# Block Importance Scoring
# ---------------------------------------------------------------------------

def block_importance(block, edge_count: int, cluster_spirit: float,
                     now: float, limits: CosmicLimits) -> float:
    """Score a block's importance for eviction decisions.

    Lower score = more likely to be evicted.

    Components:
      reinforcement  -- block.reinforcement_score (W matrix care)
      recency        -- exp(-decay * age), half-life ~19 hours
      connectivity   -- log1p(edge_count) / log1p(max_edges_per_block)
      spirit         -- cluster spirit (how alive is the block's cluster)

    Protected types: 'value' and 'reflection' get 1.5x boost.
    """
    age = max(0.0, now - block.timestamp)
    recency = math.exp(-0.00001 * age)  # half-life ~19.25 hours

    max_epb = limits.max_edges_per_block if limits.max_edges_per_block < 2**62 else 33
    connectivity = math.log1p(edge_count) / math.log1p(max_epb)
    reinforcement = min(block.reinforcement_score, 1.0)
    spirit = min(cluster_spirit, 1.0)

    base = (limits.w_reinforcement * reinforcement +
            limits.w_recency * recency +
            limits.w_connectivity * connectivity +
            limits.w_spirit * spirit)

    # Type boost: sacred/reflection memories are harder to evict
    if block.sensory_type in ("value", "reflection"):
        base *= 1.5

    return base


# ---------------------------------------------------------------------------
# CosmicReaper -- Sample-based Eviction Engine
# ---------------------------------------------------------------------------

class CosmicReaper:
    """Tournament selection eviction. O(sample_size) per round, not O(n).

    At 33 crore blocks, scanning all is impossible. Random sampling gives
    statistically fair eviction without full scans.

    Flow:
      1. Sample `sample_size` random alive blocks
      2. Score each by block_importance()
      3. Evict the `evict_batch` lowest-scoring blocks
      4. Call engine.dissolve_block() for each
    """

    def __init__(self, engine, limits: CosmicLimits):
        self._engine = engine
        self._limits = limits
        self._evictions = 0
        self._last_reap_time = 0.0

    def should_reap(self) -> bool:
        """Check if any capacity limit is exceeded."""
        e = self._engine
        return (e._blocks.alive_count() >= self._limits.max_blocks or
                e._graph.total_edge_count() >= self._limits.max_total_edges)

    def reap(self) -> int:
        """Run one eviction round. Returns count of dissolved blocks.
        Does nothing if capacity limits are not exceeded."""
        if not self.should_reap():
            return 0

        e = self._engine
        limits = self._limits
        now = time.time()

        # Sample random alive blocks
        alive = e._blocks.alive_ids()
        if not alive:
            return 0

        if len(alive) <= limits.reaper_sample_size:
            sample_ids = list(alive)
        else:
            sample_ids = random.sample(list(alive), limits.reaper_sample_size)

        # Score each
        scored = []
        for bid in sample_ids:
            try:
                block = e._blocks.read(bid)
            except Exception:
                continue
            if block is None:
                continue

            # Skip young blocks (protection period)
            age = now - block.timestamp
            if age < limits.reaper_min_age_seconds:
                continue

            edge_count = e._graph.edge_count_for(bid)
            cluster_spirit = 0.0
            if block.cluster_id:
                cd = e._clusters.get(block.cluster_id)
                if cd:
                    cluster_spirit = cd.spirit_size

            score = block_importance(block, edge_count, cluster_spirit, now, limits)
            scored.append((bid, score))

        if not scored:
            return 0

        # Sort by importance (ascending = weakest first)
        scored.sort(key=lambda x: x[1])

        # Evict the weakest
        evict_count = min(limits.reaper_evict_batch, len(scored))
        evicted = 0
        for bid, score in scored[:evict_count]:
            try:
                e.dissolve_block(bid)
                evicted += 1
                log.debug("Reaped block %s (importance=%.4f)", bid[:8], score)
            except Exception as ex:
                log.warning("Reap failed for %s: %s", bid[:8], ex)

        self._evictions += evicted
        self._last_reap_time = now
        if evicted:
            log.info("Reaped %d blocks (total evictions: %d)", evicted, self._evictions)
        return evicted

    def status(self) -> dict:
        """Return reaper status."""
        return {
            "total_evictions": self._evictions,
            "last_reap_time": self._last_reap_time,
            "limits": {
                "max_blocks": self._limits.max_blocks,
                "max_gods": self._limits.max_gods,
                "max_edges_per_block": self._limits.max_edges_per_block,
                "max_total_edges": self._limits.max_total_edges,
                "max_clusters": self._limits.max_clusters,
                "max_personalities": self._limits.max_personalities,
            },
        }
