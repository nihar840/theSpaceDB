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

if TYPE_CHECKING:
    from ._core.engine import SpaceEngine

log = logging.getLogger("spacedb.drift")


class DriftController:
    """Start / stop / inspect the cognitive drift background loop."""

    def __init__(self, engine: "SpaceEngine"):
        self._engine = engine

    def start(self, idle_seconds: float = 10.0):
        """
        Start the cognitive drift engine.

        After *idle_seconds* of no input the mind begins random walks,
        forming new connections and evolving the distance matrix.
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
        """Return drift-related metrics."""
        s = self._engine.status()
        return {
            "active":     s["drift"],
            "idle_s":     s["idle_s"],
            "input_rate": s["input_rate"],
        }

    def __repr__(self):
        state = "ON" if self.active else "OFF"
        return f"DriftController(state={state})"
