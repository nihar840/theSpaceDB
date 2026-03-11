import os, pickle, random, threading, logging
from collections import defaultdict
from ._exceptions import StoreCorruptedError

logger = logging.getLogger("spacedb.graph")


class GraphStore:
    def __init__(self, path: str):
        self._path = os.path.join(path, 'graph.adj')
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._dirty = False
        self._load()

    def add_edge(self, a: str, b: str, weight: float = 1.0):
        if a == b: return
        with self._lock:
            w = max(self._adj[a].get(b, 0.0), weight)
            self._adj[a][b] = self._adj[b][a] = w
            self._dirty = True

    def update_weight(self, a: str, b: str, delta: float):
        if a == b: return
        with self._lock:
            w = max(0.0, self._adj[a].get(b, 0.0) + delta)
            self._adj[a][b] = self._adj[b][a] = w
            self._dirty = True

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        with self._lock:
            return list(self._adj.get(node, {}).items())

    def random_walk(self, start: str, steps: int) -> list[str]:
        with self._lock:
            path, cur = [start], start
            for _ in range(steps):
                nb = list(self._adj.get(cur, {}).items())
                if not nb: break
                ids, ws = zip(*nb)
                cur = random.choices(ids, weights=ws)[0]
                path.append(cur)
            return path

    def flush(self):
        with self._lock:
            if self._dirty:
                self._save()
                self._dirty = False

    def random_edges(self, n: int) -> list[tuple[str, str, float]]:
        """Return up to *n* random edges as (a, b, weight) tuples."""
        with self._lock:
            edges = []
            for a, nb in self._adj.items():
                for b, w in nb.items():
                    if a < b:           # each undirected edge once
                        edges.append((a, b, w))
            if not edges:
                return []
            return random.sample(edges, min(n, len(edges)))

    def remove_edge(self, a: str, b: str):
        """Remove an edge entirely from the graph."""
        with self._lock:
            self._adj[a].pop(b, None)
            self._adj[b].pop(a, None)
            self._dirty = True

    def node_count(self) -> int: return len(self._adj)
    def edge_count(self) -> int: return sum(len(v) for v in self._adj.values()) // 2

    def _save(self):
        with open(self._path, 'wb') as f: pickle.dump(dict(self._adj), f)

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path, 'rb') as f:
                    self._adj = defaultdict(dict, pickle.load(f))
            except (pickle.UnpicklingError, EOFError, Exception) as exc:
                logger.warning("Corrupt graph file, resetting: %s", exc)
                self._adj = defaultdict(dict)
