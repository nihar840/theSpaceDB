import os, pickle, random
from collections import defaultdict


class GraphStore:
    def __init__(self, path: str):
        self._path = os.path.join(path, 'graph.adj')
        self._adj: dict[str, dict[str, float]] = defaultdict(dict)
        self._load()

    def add_edge(self, a: str, b: str, weight: float = 1.0):
        if a == b: return
        w = max(self._adj[a].get(b, 0.0), weight)
        self._adj[a][b] = self._adj[b][a] = w
        self._save()

    def update_weight(self, a: str, b: str, delta: float):
        if a == b: return
        w = max(0.0, self._adj[a].get(b, 0.0) + delta)
        self._adj[a][b] = self._adj[b][a] = w
        self._save()

    def neighbors(self, node: str) -> list[tuple[str, float]]:
        return list(self._adj.get(node, {}).items())

    def random_walk(self, start: str, steps: int) -> list[str]:
        path, cur = [start], start
        for _ in range(steps):
            nb = self.neighbors(cur)
            if not nb: break
            ids, ws = zip(*nb)
            cur = random.choices(ids, weights=ws)[0]
            path.append(cur)
        return path

    def node_count(self) -> int: return len(self._adj)
    def edge_count(self) -> int: return sum(len(v) for v in self._adj.values()) // 2

    def _save(self):
        with open(self._path, 'wb') as f: pickle.dump(dict(self._adj), f)

    def _load(self):
        if os.path.exists(self._path):
            with open(self._path, 'rb') as f:
                self._adj = defaultdict(dict, pickle.load(f))
