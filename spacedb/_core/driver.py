"""
driver.py — The Driver: executive layer / the Self

The Driver is always active, observes everything, and decides which
personality speaks.  It learns from feedback and can steer drift.

The Driver does NOT replace existing mechanisms — it wraps and augments them.

Every N activations it stores a meta-observation block (sensory_type="internal")
that participates in clustering/drift — the Driver can form its OWN personality.
"""

import json
import logging
import os
import time
from collections import deque
from typing import Optional, TYPE_CHECKING

import numpy as np

from .models import ActivationRecord, DriverState

if TYPE_CHECKING:
    from .engine import SpaceEngine

log = logging.getLogger("spacedb.driver")


class Driver:
    """
    The executive self.  Always active. Observes all events.
    Decides which personality speaks.  Learns from feedback.
    """

    # ── constants ──────────────────────────────────────────────
    HISTORY_WINDOW    = 100     # keep last N activation records
    MOMENTUM_WEIGHT   = 0.3    # inertia bonus for staying in current personality
    SWITCH_COST       = 0.15   # penalty for switching (prevents thrashing)
    FEEDBACK_LR       = 0.01   # learning rate for affinity from feedback
    META_INGEST_EVERY = 10     # emit meta-observation block every N activations
    DRIFT_BIAS_DECAY  = 0.98   # drift focus fades each step
    FATIGUE_WINDOW    = 20     # recent queries to check for overuse
    COACTIVATION_LR   = 0.005  # coactivation learning rate

    def __init__(self, engine: "SpaceEngine"):
        self._engine = engine
        self._path = os.path.join(engine.path, "driver.json")
        self._state = DriverState()
        self._history: deque[ActivationRecord] = deque(maxlen=self.HISTORY_WINDOW)
        self._activation_count_since_meta = 0
        self._load()

    # ── core: strategic personality selection ──────────────────
    def select_personality(
        self,
        query_vec: np.ndarray,
        candidate_scores: dict[str, float],
    ) -> Optional[str]:
        """
        Strategic selection — replaces max-score heuristic.

        Parameters
        ----------
        query_vec : np.ndarray
            The query embedding vector.
        candidate_scores : dict[str, float]
            Personality id -> base query score from engine.

        Returns
        -------
        Optional[str]
            Selected personality id, or None if all < threshold.
        """
        if not candidate_scores:
            return None

        active = self._state.active_personality_id
        composites: dict[str, float] = {}

        for pid, base_score in candidate_scores.items():
            composite = base_score

            # Momentum: bonus for staying in current personality
            if pid == active:
                composite += self.MOMENTUM_WEIGHT
            else:
                composite -= self.SWITCH_COST

            # Learned affinity [-1, 1] scaled by 0.2
            affinity = self._state.personality_affinity.get(pid, 0.0)
            composite += affinity * 0.2

            # Coactivation: bonus if this pid often co-fires with active
            if active and active != pid:
                coact = self._state.coactivation_matrix.get(active, {}).get(pid, 0.0)
                composite += coact * 0.1

            # Fatigue: penalty if overused in recent window
            recent = list(self._history)[-self.FATIGUE_WINDOW:]
            if recent:
                use_count = sum(1 for r in recent if r.personality_id == pid)
                fatigue = use_count / len(recent)
                composite -= fatigue * 0.1

            # Stability: small bonus for personalities with high spirit
            cluster = self._engine.clusters.get(pid)
            if cluster:
                stability = min(cluster.spirit_size / 10.0, 1.0)
                composite += stability * 0.05

            composites[pid] = composite

        # Pick highest composite
        best_pid = max(composites, key=composites.get)
        best_score = composites[best_pid]

        if best_score < 0.01:
            return None

        # Record activation
        switched = (best_pid != active)
        if switched:
            self._state.total_switches += 1
            self._state.active_since = time.time()
        self._state.active_personality_id = best_pid
        self._state.total_activations += 1

        # Record history
        cluster = self._engine.clusters.get(best_pid)
        record = ActivationRecord(
            personality_id=best_pid,
            personality_name=cluster.name if cluster else None,
            trigger_type="query",
            scores=composites,
        )
        self._history.append(record)
        self._activation_count_since_meta += 1

        # Update coactivation: active and best_pid co-fired
        if active and active != best_pid:
            self._update_coactivation(active, best_pid)

        # Maybe emit meta-observation
        if self._activation_count_since_meta >= self.META_INGEST_EVERY:
            self._emit_meta_observation()
            self._activation_count_since_meta = 0

        self._save()
        log.info("Driver selected: %s (composite=%.3f, switched=%s)",
                 best_pid, best_score, switched)
        return best_pid

    # ── feedback: learn from outcomes ─────────────────────────
    def feedback(self, outcome: str, score: float = 0.0):
        """
        Learn from results — update affinity for active personality.

        Parameters
        ----------
        outcome : str
            "positive", "negative", or "neutral"
        score : float
            Strength of the outcome signal (0-1).
        """
        pid = self._state.active_personality_id
        if not pid:
            return

        # Update affinity
        current = self._state.personality_affinity.get(pid, 0.0)
        if outcome == "positive":
            delta = self.FEEDBACK_LR * (1.0 + score)
        elif outcome == "negative":
            delta = -self.FEEDBACK_LR * (1.0 + score)
        else:
            delta = 0.0

        new_affinity = max(-1.0, min(1.0, current + delta))
        self._state.personality_affinity[pid] = new_affinity

        # Update last record with feedback
        if self._history:
            last = self._history[-1]
            last.outcome = outcome
            last.feedback_score = score

        self._save()
        log.info("Driver feedback: %s -> %s (affinity %.3f -> %.3f)",
                 pid, outcome, current, new_affinity)

    # ── drift steering ────────────────────────────────────────
    def direct_drift(self, cluster_id: str, strength: float = 1.0):
        """Tell drift to focus on a specific cluster."""
        self._state.drift_focus = cluster_id
        self._state.drift_focus_strength = min(max(strength, 0.0), 1.0)
        self._save()
        log.info("Driver directing drift -> %s (strength=%.2f)",
                 cluster_id, strength)

    def get_drift_bias(self) -> tuple[Optional[str], float]:
        """Called by drift to check if Driver wants to steer."""
        if self._state.drift_focus and self._state.drift_focus_strength > 0.01:
            return self._state.drift_focus, self._state.drift_focus_strength
        return None, 0.0

    # ── observation hooks ─────────────────────────────────────
    def on_ingest(self, block, vec: np.ndarray):
        """Observe every ingest. Skip internal blocks to prevent recursion."""
        if block.sensory_type == "internal":
            return

    def on_query(self, vec: np.ndarray, results: list, pid: Optional[str]):
        """Observe every query."""
        pass

    def on_cluster_event(self, event_type: str, cluster_id: str, data: dict = None):
        """Observe personality promotion/demotion."""
        if event_type == "promoted":
            # Initialize affinity for new personality
            if cluster_id not in self._state.personality_affinity:
                self._state.personality_affinity[cluster_id] = 0.0
            log.info("Driver observed promotion: %s", cluster_id)
        elif event_type == "demoted":
            log.info("Driver observed demotion: %s", cluster_id)
        self._save()

    def on_drift_step(self, seed_id: str, path: list[str]):
        """Observe every drift step, decay focus."""
        if self._state.drift_focus_strength > 0.01:
            self._state.drift_focus_strength *= self.DRIFT_BIAS_DECAY
            if self._state.drift_focus_strength < 0.01:
                self._state.drift_focus = None
                self._state.drift_focus_strength = 0.0
            self._save()

    # ── meta-observations ─────────────────────────────────────
    def _emit_meta_observation(self):
        """
        Store a meta-block (sensory_type="internal") in SpaceDB.
        These blocks participate in clustering/drift — the Driver
        can literally form its OWN personality cluster.
        """
        recent = list(self._history)[-self.META_INGEST_EVERY:]
        if not recent:
            return

        # Summarize recent activations as a text token
        pid_counts: dict[str, int] = {}
        outcomes: dict[str, int] = {"positive": 0, "negative": 0, "neutral": 0}
        for r in recent:
            if r.personality_id:
                pid_counts[r.personality_id] = pid_counts.get(r.personality_id, 0) + 1
            if r.outcome:
                outcomes[r.outcome] = outcomes.get(r.outcome, 0) + 1

        dominant = max(pid_counts, key=pid_counts.get) if pid_counts else "none"
        cluster = self._engine.clusters.get(dominant)
        dominant_name = cluster.name if cluster else dominant

        token = (
            f"meta::driver activations={len(recent)} "
            f"dominant={dominant_name} "
            f"switches={self._state.total_switches} "
            f"pos={outcomes['positive']} neg={outcomes['negative']}"
        )

        # Create an internal block using the engine's current average vector
        # as a rough centroid of recent activity
        ids, mat = self._engine._vectors.get_all()
        if len(ids) == 0:
            return
        avg_vec = mat.mean(axis=0).astype(np.float32)

        try:
            self._engine.ingest(token, avg_vec, sensory_type="internal")
            log.info("Driver meta-observation stored: %s", token[:60])
        except Exception as e:
            log.warning("Failed to store meta-observation: %s", e)

    # ── coactivation ──────────────────────────────────────────
    def _update_coactivation(self, pid_a: str, pid_b: str):
        """Update coactivation matrix: a and b co-fired."""
        if pid_a not in self._state.coactivation_matrix:
            self._state.coactivation_matrix[pid_a] = {}
        if pid_b not in self._state.coactivation_matrix:
            self._state.coactivation_matrix[pid_b] = {}

        current_ab = self._state.coactivation_matrix[pid_a].get(pid_b, 0.0)
        current_ba = self._state.coactivation_matrix[pid_b].get(pid_a, 0.0)

        self._state.coactivation_matrix[pid_a][pid_b] = min(current_ab + self.COACTIVATION_LR, 1.0)
        self._state.coactivation_matrix[pid_b][pid_a] = min(current_ba + self.COACTIVATION_LR, 1.0)

    # ── introspection ─────────────────────────────────────────
    @property
    def active_personality(self) -> Optional[str]:
        """Who's currently speaking."""
        return self._state.active_personality_id

    def status(self) -> dict:
        return {
            "active_personality": self._state.active_personality_id,
            "active_since": self._state.active_since,
            "total_activations": self._state.total_activations,
            "total_switches": self._state.total_switches,
            "personality_count": len(self._state.personality_affinity),
            "drift_focus": self._state.drift_focus,
            "drift_focus_strength": round(self._state.drift_focus_strength, 4),
            "history_size": len(self._history),
        }

    def history(self, limit: int = 50) -> list[dict]:
        records = list(self._history)[-limit:]
        return [
            {
                "timestamp": r.timestamp,
                "personality_id": r.personality_id,
                "personality_name": r.personality_name,
                "trigger_type": r.trigger_type,
                "scores": r.scores,
                "outcome": r.outcome,
                "feedback_score": r.feedback_score,
            }
            for r in records
        ]

    def affinity(self) -> dict[str, float]:
        return dict(self._state.personality_affinity)

    # ── persistence ───────────────────────────────────────────
    def _save(self):
        data = {
            "active_personality_id": self._state.active_personality_id,
            "active_since": self._state.active_since,
            "total_activations": self._state.total_activations,
            "total_switches": self._state.total_switches,
            "personality_affinity": self._state.personality_affinity,
            "coactivation_matrix": self._state.coactivation_matrix,
            "drift_focus": self._state.drift_focus,
            "drift_focus_strength": self._state.drift_focus_strength,
            "history": [
                {
                    "timestamp": r.timestamp,
                    "personality_id": r.personality_id,
                    "personality_name": r.personality_name,
                    "trigger_type": r.trigger_type,
                    "scores": r.scores,
                    "outcome": r.outcome,
                    "feedback_score": r.feedback_score,
                }
                for r in self._history
            ],
        }
        try:
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            log.warning("Failed to save driver state: %s", e)

    def _load(self):
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                data = json.load(f)
            self._state.active_personality_id = data.get("active_personality_id")
            self._state.active_since = data.get("active_since", time.time())
            self._state.total_activations = data.get("total_activations", 0)
            self._state.total_switches = data.get("total_switches", 0)
            self._state.personality_affinity = data.get("personality_affinity", {})
            self._state.coactivation_matrix = data.get("coactivation_matrix", {})
            self._state.drift_focus = data.get("drift_focus")
            self._state.drift_focus_strength = data.get("drift_focus_strength", 0.0)
            for rec in data.get("history", []):
                self._history.append(ActivationRecord(
                    timestamp=rec.get("timestamp", 0),
                    personality_id=rec.get("personality_id"),
                    personality_name=rec.get("personality_name"),
                    trigger_type=rec.get("trigger_type", "query"),
                    scores=rec.get("scores", {}),
                    outcome=rec.get("outcome"),
                    feedback_score=rec.get("feedback_score", 0.0),
                ))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            log.warning("Failed to load driver state: %s — starting fresh", e)
            self._state = DriverState()
            self._history.clear()
