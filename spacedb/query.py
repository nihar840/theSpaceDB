"""
query.py — Chainable QueryBuilder

Usage:
    results = (
        space.query("apple memories")
             .within(ms=500)
             .as_personality("food")
             .limit(10)
             .fetch()
    )
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .space import Space


class QueryBuilder:

    def __init__(self, space: 'Space', text: str):
        self._space          = space
        self._text           = text
        self._time_budget_ms = 100
        self._personality    = None
        self._limit          = 20

    def within(self, ms: int) -> 'QueryBuilder':
        """
        Set the time budget for the search.

        Low ms  → nearest blocks only  (fast, surface memory)
        High ms → far clusters explored (deep reasoning)
        """
        self._time_budget_ms = ms
        return self

    def as_personality(self, name_or_id: str) -> 'QueryBuilder':
        """Bias results toward a specific personality cluster."""
        self._personality = name_or_id
        return self

    def limit(self, n: int) -> 'QueryBuilder':
        """Maximum number of results to return."""
        self._limit = n
        return self

    def fetch(self) -> list[dict]:
        """
        Execute the query. Returns list of result dicts:
        [{ 'id', 'token', 'score', 'cluster', 'sensory_type', 'reinforcement' }, ...]
        """
        vec = self._space._embed(self._text)

        # Resolve personality name → cluster_id
        pid = None
        if self._personality:
            pid = self._space._resolve_personality(self._personality)

        raw = self._space._engine.query(
            vec,
            time_budget_ms=self._time_budget_ms,
            personality_id=pid,
            limit=self._limit,
        )

        return [
            {
                'id':           b.id,
                'token':        b.token,
                'score':        round(score, 6),
                'cluster':      b.cluster_id,
                'sensory_type': b.sensory_type,
                'reinforcement': round(b.reinforcement_score, 4),
            }
            for b, score in raw
        ]

    def __repr__(self):
        return (f"QueryBuilder(text={self._text!r}, "
                f"within={self._time_budget_ms}ms, "
                f"personality={self._personality}, limit={self._limit})")
