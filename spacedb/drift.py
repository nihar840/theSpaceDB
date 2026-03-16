"""
drift.py — DriftController

Controls the cognitive drift background engine.
When idle, the mind starts 'thinking' — random walks through the graph,
forming new connections, linking clusters, evolving quietly.

Usage::

    space.drift.start()
    space.drift.stop()
    space.drift.status()
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ._core.engine import SpaceEngine

log = logging.getLogger("spacedb.drift")


class DriftController:
    """Start / stop / inspect the cognitive drift background loop.

    Now includes Dream Mode — structured deep thinking that goes
    beyond random walks: bridge discovery, contradiction detection,
    goal-directed exploration, and insight synthesis.
    """

    def __init__(self, engine: "SpaceEngine"):
        self._engine = engine

    def start(self, idle_seconds: float = 10.0):
        """
        Start the cognitive drift engine.

        After *idle_seconds* of no input the mind begins random walks,
        forming new connections and evolving the distance matrix.
        Dreams activate automatically during drift (30% chance per step).
        """
        self._engine.start_drift(idle_s=idle_seconds)
        log.info("Cognitive drift started (idle threshold: %.1fs)", idle_seconds)

    def stop(self):
        """Stop the cognitive drift engine."""
        self._engine.stop_drift()
        log.info("Cognitive drift stopped.")

    @property
    def active(self) -> bool:
        """``True`` while the drift loop is running."""
        return self._engine._drift_running

    def status(self) -> dict:
        """Return drift-related metrics including dream state."""
        s = self._engine.status()
        dream = self._engine.dream.status()
        return {
            "active":      s["drift"],
            "idle_s":      s["idle_s"],
            "input_rate":  s["input_rate"],
            "dream_count": dream["dream_count"],
            "dream_goal":  dream["goal_text"],
        }

    # ── dream mode API ────────────────────────────────────────

    def dream_goal(self, topic_vec: np.ndarray, topic_text: str = ""):
        """Direct idle dreaming toward a specific topic.

        While dreaming, the mind will preferentially explore and
        reinforce paths related to this topic.

        Parameters
        ----------
        topic_vec : np.ndarray
            Embedding vector of the topic to dream about.
        topic_text : str
            Human-readable description (for logging/status).
        """
        self._engine.dream.set_goal(topic_vec, topic_text)

    def clear_dream_goal(self):
        """Remove the dream goal. Dreaming returns to exploration mode."""
        self._engine.dream.clear_goal()

    def dream_status(self) -> dict:
        """Return dream-specific metrics."""
        return self._engine.dream.status()

    def __repr__(self):
        state = "ON" if self.active else "OFF"
        return f"DriftController(state={state})"
