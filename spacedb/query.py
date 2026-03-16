"""
query.py — Chainable QueryBuilder

Usage (text — requires sentence-transformers)::

    results = (
        space.query("apple memories")
             .within(ms=500)
             .as_personality("food")
             .limit(10)
             .fetch()
    )

Usage (raw vector — no extra dependencies)::

    results = space.query(my_vec).within(ms=200).limit(5).fetch()
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .space import Space


class QueryBuilder:
    """
    Chainable query builder for SpaceDB.

    Constructed via ``Space.query()`` — do not instantiate directly.
    """

    def __init__(
        self,
        space: "Space",
        _text: Optional[str] = None,
        _vector: Optional[np.ndarray] = None,
    ):
        if _text is None and _vector is None:
            raise ValueError("QueryBuilder requires either _text or _vector")
        self._space          = space
        self._text           = _text
        self._vector         = _vector
        self._time_budget_ms = 100
        self._personality    = None
        self._auto_pid       = False
        self._limit          = 20

    def within(self, ms: int) -> "QueryBuilder":
        """
        Set the time budget for the search.

        Low ms  → nearest blocks only  (fast, surface memory)
        High ms → far clusters explored (deep reasoning)

        Parameters
        ----------
        ms : int
            Milliseconds. Must be > 0.
        """
        if not isinstance(ms, int) or ms <= 0:
            raise ValueError(f"within(ms) must be a positive integer, got {ms!r}")
        self._time_budget_ms = ms
        return self

    def as_personality(self, name_or_id: str) -> "QueryBuilder":
        """Bias results toward a specific personality cluster."""
        self._personality = name_or_id
        self._auto_pid = False
        return self

    def auto_personality(self) -> "QueryBuilder":
        """Let the Driver (executive self) choose the personality."""
        self._auto_pid = True
        self._personality = None
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """
        Maximum number of results to return.

        Parameters
        ----------
        n : int
            Must be > 0.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"limit(n) must be a positive integer, got {n!r}")
        self._limit = n
        return self

    def fetch(self) -> list[dict]:
        """
        Execute the query.

        Returns
        -------
        list[dict]
            Each dict contains: ``id``, ``token``, ``score``, ``cluster``,
            ``sensory_type``, ``reinforcement``.
        """
        results, _ = self._execute()
        return results

    def fetch_with_confidence(self):
        """
        Execute the query and return confidence metrics alongside results.

        Returns
        -------
        tuple[list[dict], QueryConfidence]
            Results list (same as ``fetch()``) plus a ``QueryConfidence``
            object with ``score``, ``hit_count``, ``coverage``, ``sparse``, etc.

        Example::

            results, confidence = (
                space.query(vec).within(ms=300).limit(5)
                    .auto_personality().fetch_with_confidence()
            )
            if confidence.sparse:
                print("Not much memory about this topic...")
        """
        return self._execute()

    def _execute(self):
        """Shared execution logic for fetch() and fetch_with_confidence()."""
        # Resolve query vector
        if self._vector is not None:
            vec = self._vector
        else:
            vec = self._space._embed(self._text)

        # Resolve personality: manual, auto (Driver), or none
        pid = None
        if self._auto_pid:
            # Let the Driver decide
            personalities = self._space._engine.clusters.personalities()
            if personalities:
                candidate_scores = {}
                for p in personalities:
                    raw_p = self._space._engine.query(
                        vec, time_budget_ms=50, personality_id=p.id, limit=3,
                    )
                    if raw_p:
                        avg = sum(s for _, s in raw_p) / len(raw_p)
                        candidate_scores[p.id] = avg
                if candidate_scores:
                    pid = self._space._engine.driver.select_personality(
                        vec, candidate_scores,
                    )
        elif self._personality:
            pid = self._space._resolve_personality(self._personality)

        raw = self._space._engine.query(
            vec,
            time_budget_ms=self._time_budget_ms,
            personality_id=pid,
            limit=self._limit,
        )

        results = [
            {
                "id":            b.id,
                "token":         b.token,
                "score":         round(score, 6),
                "cluster":       b.cluster_id,
                "sensory_type":  b.sensory_type,
                "reinforcement": round(b.reinforcement_score, 4),
            }
            for b, score in raw
        ]

        # Cross-personality dynamics: if results span multiple
        # personalities, gently reinforce inter-personality links.
        if len(results) >= 2:
            self._space._engine.cross_personality_link(
                [r["id"] for r in results]
            )

        # Compute confidence metrics
        confidence = self._space._engine._compute_confidence(raw, self._limit)

        return results, confidence

    def __repr__(self):
        source = self._text if self._text is not None else "<vector>"
        return (
            f"QueryBuilder(source={source!r}, "
            f"within={self._time_budget_ms}ms, "
            f"personality={self._personality}, limit={self._limit})"
        )
