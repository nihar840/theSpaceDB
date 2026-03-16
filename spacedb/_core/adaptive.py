"""
adaptive.py — Self-tuning parameter engine for SpaceDB

Replaces hardcoded constants with pure functions of system state.
Every parameter is a pure function: SystemState → float.
No side effects, no disk I/O, easy to test.

As the mind grows, parameters adapt:
  - Young mind (few blocks): cluster often, connect densely, preserve everything
  - Mature mind (many blocks): cluster less, connect selectively, forget more
  - Active mind (high input rate): smaller drift nudges (don't overwrite learning)
  - Idle mind (low input rate): larger drift nudges (dreaming matters more)
"""

import math
from dataclasses import dataclass


@dataclass
class SystemState:
    """Snapshot of SpaceDB metrics used for parameter computation."""
    block_count: int
    edge_count: int
    cluster_count: int
    personality_count: int
    god_count: int
    input_rate: float          # from DistanceEngine
    w_updates: int             # from DistanceEngine
    capacity_pressure: float = 0.0   # 0.0 = empty, 1.0 = at limit


class AdaptiveParams:
    """
    Compute dynamic parameters from system state.

    All methods are static pure functions — no mutation, no disk I/O.
    Each replaces one hardcoded constant in SpaceEngine.
    """

    @staticmethod
    def connect_k(state: SystemState) -> int:
        """How many neighbours to wire on ingest.

        Scales with sqrt(block_count / 10), clamped [3, 15].
        Small corpus → dense local wiring (3).
        Large corpus → broader reach (up to 15).
        High pressure → fewer connections (conserve edges).
        """
        k = int(math.sqrt(max(state.block_count, 1) / 10) + 3)
        # Pressure scaling: reduce connections at high capacity
        pressure_mult = 1.0 - 0.4 * state.capacity_pressure  # 1.0 → 0.6
        k = int(k * pressure_mult)
        return max(3, min(15, k))

    @staticmethod
    def drift_nudge(state: SystemState) -> float:
        """W-matrix reinforcement strength per drift step.

        High input rate → smaller nudge (active learning dominates).
        Low input rate → larger nudge (idle dreaming matters more).
        High pressure → stronger drift (faster reorganization).
        Clamped [0.0001, 0.003].
        """
        base = 0.0005
        # Inverse relationship: more active = less drift nudge
        rate_factor = 1.0 / (1.0 + state.input_rate * 10)
        result = base * rate_factor * 2
        # Pressure scaling: stronger drift at high capacity
        pressure_mult = 1.0 + 0.5 * state.capacity_pressure
        return max(0.0001, min(0.003, result * pressure_mult))

    @staticmethod
    def passive_decay(state: SystemState) -> float:
        """Forgetting rate applied during ingest.

        Young mind (few blocks) → low decay (preserve everything).
        Mature mind (many blocks) → higher decay (need to forget).
        High pressure → forget faster (make room for new).
        Clamped [0.00005, 0.001].
        """
        maturity = min(state.block_count / 1000, 1.0)
        base = 0.00005 + maturity * 0.00045
        # Pressure scaling: up to 2x decay at full capacity
        pressure_mult = 1.0 + state.capacity_pressure
        return max(0.00005, min(0.001, base * pressure_mult))

    @staticmethod
    def auto_cluster_every(state: SystemState) -> int:
        """Blocks between re-clustering.

        Small corpus → cluster often (every 20).
        Large corpus → less often (up to every 200).
        High pressure → cluster more often (find eviction targets).
        """
        every = int(20 + state.block_count / 10)
        # Pressure scaling: cluster more often at high capacity
        pressure_mult = 1.0 - 0.3 * state.capacity_pressure  # 1.0 → 0.7
        every = int(every * pressure_mult)
        return max(20, min(200, every))

    @staticmethod
    def route_top_k(state: SystemState) -> int:
        """God Points to probe per query.

        Scales with god_count: probe ~30% of gods, clamped [1, 10].
        """
        if state.god_count == 0:
            return 3  # default fallback
        return max(1, min(10, int(state.god_count * 0.3 + 1)))

    @staticmethod
    def personality_bias(state: SystemState) -> float:
        """Personality bias multiplier in query scoring.

        Few personalities → strong bias (1.8×) to encourage differentiation.
        Many personalities → weaker bias (1.2×) for cross-pollination.
        """
        if state.personality_count <= 1:
            return 1.8
        return max(1.2, 1.8 - 0.1 * state.personality_count)

    @staticmethod
    def recency_decay(state: SystemState) -> float:
        """Recency decay constant for query scoring: exp(-k × age_seconds).

        High input rate → faster decay (recent is more important).
        Low input rate → slower decay (old memories still relevant).

        Scale: per-second constant, designed so memories remain useful
        for days/weeks, not minutes.
          - At minimum (2e-7/s): half-life ≈ 40 days
          - At maximum (5e-6/s): half-life ≈ 1.6 days
          - Default base (1e-6/s): half-life ≈ 8 days

        Clamped [0.0000002, 0.000005].
        """
        base = 0.000001          # 1e-6 per second → ~8 day half-life
        rate_factor = 1.0 + state.input_rate * 2
        return max(0.0000002, min(0.000005, base * rate_factor))
