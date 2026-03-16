"""
test_adaptive.py -- Tests for the adaptive parameter engine.

Verifies that every adaptive function:
  - Returns correct types
  - Respects clamping bounds
  - Scales monotonically with the expected input
  - Is a pure function (same input -> same output)
"""

import pytest
from spacedb._core.adaptive import AdaptiveParams, SystemState


def _state(**overrides) -> SystemState:
    """Create a SystemState with sensible defaults, overridable."""
    defaults = dict(
        block_count=100,
        edge_count=200,
        cluster_count=3,
        personality_count=1,
        god_count=2,
        input_rate=0.5,
        w_updates=50,
    )
    defaults.update(overrides)
    return SystemState(**defaults)


# ── connect_k ────────────────────────────────────────────────


class TestConnectK:

    def test_returns_int(self):
        assert isinstance(AdaptiveParams.connect_k(_state()), int)

    def test_minimum_clamp(self):
        assert AdaptiveParams.connect_k(_state(block_count=0)) >= 3

    def test_maximum_clamp(self):
        assert AdaptiveParams.connect_k(_state(block_count=100_000)) <= 15

    def test_increases_with_blocks(self):
        k_small = AdaptiveParams.connect_k(_state(block_count=10))
        k_large = AdaptiveParams.connect_k(_state(block_count=5000))
        assert k_large >= k_small

    def test_pure_function(self):
        s = _state(block_count=500)
        assert AdaptiveParams.connect_k(s) == AdaptiveParams.connect_k(s)


# ── drift_nudge ──────────────────────────────────────────────


class TestDriftNudge:

    def test_returns_float(self):
        assert isinstance(AdaptiveParams.drift_nudge(_state()), float)

    def test_minimum_clamp(self):
        assert AdaptiveParams.drift_nudge(_state(input_rate=100.0)) >= 0.0001

    def test_maximum_clamp(self):
        assert AdaptiveParams.drift_nudge(_state(input_rate=0.0)) <= 0.002

    def test_decreases_with_input_rate(self):
        nudge_idle = AdaptiveParams.drift_nudge(_state(input_rate=0.01))
        nudge_active = AdaptiveParams.drift_nudge(_state(input_rate=5.0))
        assert nudge_idle >= nudge_active


# ── passive_decay ────────────────────────────────────────────


class TestPassiveDecay:

    def test_returns_float(self):
        assert isinstance(AdaptiveParams.passive_decay(_state()), float)

    def test_minimum_clamp(self):
        assert AdaptiveParams.passive_decay(_state(block_count=0)) >= 0.00005

    def test_maximum_clamp(self):
        assert AdaptiveParams.passive_decay(_state(block_count=100_000)) <= 0.0005

    def test_increases_with_maturity(self):
        young = AdaptiveParams.passive_decay(_state(block_count=10))
        mature = AdaptiveParams.passive_decay(_state(block_count=5000))
        assert mature >= young


# ── auto_cluster_every ───────────────────────────────────────


class TestAutoClusterEvery:

    def test_returns_int(self):
        assert isinstance(AdaptiveParams.auto_cluster_every(_state()), int)

    def test_minimum_clamp(self):
        assert AdaptiveParams.auto_cluster_every(_state(block_count=0)) >= 20

    def test_maximum_clamp(self):
        assert AdaptiveParams.auto_cluster_every(_state(block_count=100_000)) <= 200

    def test_increases_with_blocks(self):
        small = AdaptiveParams.auto_cluster_every(_state(block_count=10))
        large = AdaptiveParams.auto_cluster_every(_state(block_count=5000))
        assert large >= small


# ── route_top_k ──────────────────────────────────────────────


class TestRouteTopK:

    def test_returns_int(self):
        assert isinstance(AdaptiveParams.route_top_k(_state()), int)

    def test_fallback_when_no_gods(self):
        assert AdaptiveParams.route_top_k(_state(god_count=0)) == 3

    def test_minimum_clamp(self):
        assert AdaptiveParams.route_top_k(_state(god_count=1)) >= 1

    def test_maximum_clamp(self):
        assert AdaptiveParams.route_top_k(_state(god_count=100)) <= 10


# ── personality_bias ─────────────────────────────────────────


class TestPersonalityBias:

    def test_returns_float(self):
        assert isinstance(AdaptiveParams.personality_bias(_state()), float)

    def test_stronger_with_few_personalities(self):
        few = AdaptiveParams.personality_bias(_state(personality_count=1))
        many = AdaptiveParams.personality_bias(_state(personality_count=6))
        assert few >= many

    def test_minimum_clamp(self):
        assert AdaptiveParams.personality_bias(_state(personality_count=100)) >= 1.2


# ── recency_decay ────────────────────────────────────────────


class TestRecencyDecay:

    def test_returns_float(self):
        assert isinstance(AdaptiveParams.recency_decay(_state()), float)

    def test_increases_with_input_rate(self):
        idle = AdaptiveParams.recency_decay(_state(input_rate=0.01))
        active = AdaptiveParams.recency_decay(_state(input_rate=5.0))
        assert active >= idle

    def test_minimum_clamp(self):
        # Updated: 1e-6/s base → half-life ~8 days, clamp [2e-7, 5e-6]
        assert AdaptiveParams.recency_decay(_state(input_rate=0.0)) >= 0.0000002

    def test_maximum_clamp(self):
        # Updated: max 5e-6/s → half-life ~1.6 days
        assert AdaptiveParams.recency_decay(_state(input_rate=100.0)) <= 0.000005


# ── integration ──────────────────────────────────────────────


class TestAdaptiveIntegration:

    def test_system_state_from_engine(self, engine):
        """Engine._system_state() returns a valid SystemState."""
        state = engine._system_state()
        assert state.block_count == 0
        assert state.god_count == 0
        assert state.input_rate >= 0

    def test_params_work_with_engine_state(self, engine):
        """All adaptive functions work with a real engine state."""
        state = engine._system_state()
        assert AdaptiveParams.connect_k(state) >= 3
        assert AdaptiveParams.drift_nudge(state) > 0
        assert AdaptiveParams.passive_decay(state) > 0
        assert AdaptiveParams.auto_cluster_every(state) >= 20
        assert AdaptiveParams.route_top_k(state) >= 1
        assert AdaptiveParams.personality_bias(state) >= 1.2
        assert AdaptiveParams.recency_decay(state) > 0
