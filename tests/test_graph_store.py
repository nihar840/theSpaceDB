"""Tests for GraphStore."""

import tempfile
import pytest

from spacedb._core.graph_store import GraphStore


@pytest.fixture
def graph(tmp_dir):
    return GraphStore(tmp_dir)


class TestGraphStore:

    def test_connect_and_neighbors(self, graph):
        """Adding an edge makes both nodes appear in each other's neighbors."""
        graph.add_edge("a", "b", weight=1.0)

        nb_a = graph.neighbors("a")
        nb_b = graph.neighbors("b")

        assert len(nb_a) == 1
        assert nb_a[0] == ("b", 1.0)

        assert len(nb_b) == 1
        assert nb_b[0] == ("a", 1.0)

    def test_self_loop_ignored(self, graph):
        """Self-loops (a -> a) are silently ignored."""
        graph.add_edge("x", "x", weight=2.0)

        nb = graph.neighbors("x")
        assert len(nb) == 0, "Self-loop should not create a neighbor entry"
        assert graph.edge_count() == 0

    def test_random_walk_returns_valid_ids(self, graph):
        """random_walk produces a path of valid node ids."""
        graph.add_edge("a", "b", weight=1.0)
        graph.add_edge("b", "c", weight=1.0)
        graph.add_edge("c", "d", weight=1.0)

        path = graph.random_walk("a", steps=5)

        assert len(path) >= 1  # at least the start node
        assert path[0] == "a"
        # Every node in the path must be one of the known nodes
        known = {"a", "b", "c", "d"}
        for node in path:
            assert node in known, f"Unexpected node {node!r} in walk"

    def test_flush_persistence(self, tmp_dir):
        """Flushed graph data survives a reload from the same directory."""
        g1 = GraphStore(tmp_dir)
        g1.add_edge("p", "q", weight=3.0)
        g1.add_edge("q", "r", weight=1.5)
        g1.flush()

        # Reload from same path
        g2 = GraphStore(tmp_dir)

        nb = g2.neighbors("p")
        assert len(nb) == 1
        assert nb[0] == ("q", 3.0)

        nb_q = dict(g2.neighbors("q"))
        assert "p" in nb_q
        assert "r" in nb_q
        assert nb_q["p"] == 3.0
        assert nb_q["r"] == 1.5
