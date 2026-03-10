# theSpaceDB

> Blocks in free space. Distance evolves. Clusters emerge. Personalities form.
> Nothing fixed. Everything evolving. We chase infinity.

A self-evolving cognitive database where memory blocks float in space,
learn from experience, and form personalities over time.

[![Tests](https://github.com/nihar840/theSpaceDB/actions/workflows/ci.yml/badge.svg)](https://github.com/nihar840/theSpaceDB/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is theSpaceDB?

Traditional databases store data at fixed addresses.
**theSpaceDB stores data in free space.**

- Blocks drift closer when they are reinforced together
- Clusters emerge naturally from related blocks
- Clusters strong enough become **personalities**
- When idle, the system **thinks** — random walks form new connections
- The more it learns, the smarter it gets

---

## Installation

```bash
pip install thespacedb
```

**Optional extras:**

```bash
# Auto-embed text via sentence-transformers (~1 GB model download)
pip install thespacedb[embeddings]

# Development / testing
pip install thespacedb[dev]
```

> **Core package requires only `numpy`.**
> You can use your own embedding model and pass raw vectors — no heavy downloads needed.

---

## Quick Start

### With raw vectors (recommended — zero extra dependencies)

```python
import numpy as np
from spacedb import SpaceClient

client = SpaceClient("./my_data", dim=32)
mind = client["demo"]

# Ingest raw vectors
v1 = np.random.randn(32).astype(np.float32)
v2 = np.random.randn(32).astype(np.float32)
block_a = mind.ingest(v1, sensory_type="text")
block_b = mind.ingest(v2, sensory_type="vision")

# Query
results = mind.query(v1).within(ms=200).limit(5).fetch()

# Reinforce connection
mind.reinforce(block_a, block_b)

# Start cognitive drift
mind.drift.start()
print(mind.status())
```

### With auto-embedding (requires `thespacedb[embeddings]`)

```python
from spacedb import SpaceClient

client = SpaceClient("./my_data")   # dim=384 by default (MiniLM)
mind = client["demo"]

block = mind.ingest("apple tastes sweet")
results = mind.query("fruit").within(ms=300).limit(10).fetch()
```

---

## Python API

### SpaceClient

```python
client = SpaceClient(path, dim=384, silent=False)

client["mind_name"]      # open / create a Space
client.list_spaces()     # list all spaces
client.drop_space(name, confirm=True)
client.status()
```

### Space

```python
# Ingest (str auto-embedded, np.ndarray used directly)
block = space.ingest(content, sensory_type="text")

# Chainable query
results = space.query(content).within(ms=500).limit(10).fetch()

# Reinforce / decay connections
space.reinforce(block_a, block_b, strength=0.01)
space.decay(block_a, block_b, strength=0.001)

# Clusters
cid = space.create_cluster([b1, b2, b3], name="food")
space.clusters.all()
space.clusters.personalities()

# Cognitive drift
space.drift.start(idle_seconds=10)
space.drift.stop()
space.drift.status()
```

### Query results

Each result dict contains:

```python
{
    "id":            "uuid-string",
    "token":         "original text or <raw:type>",
    "score":         0.1234,
    "cluster":       "cluster-id or None",
    "sensory_type":  "text",
    "reinforcement": 1.0,
}
```

---

## How it Works

```
Input (text or vector)
    |
    v
MemoryBlock created  →  BlockStore (append-only log)
    |
    v
Vector stored  →  VectorStore (raw embeddings, fixed forever)
    |
    v
Connected to nearest neighbors  →  GraphStore (weighted adjacency)
    |
    v
W Matrix updated  →  DistanceEngine (learned Mahalanobis distance)
    |
    v
Clusters recomputed  →  ClusterRegistry (spirit size tracked)
    |
    v  (when idle)
Cognitive Drift  →  random walk → new connections → cross-cluster links
```

### The W Matrix (Soul of theSpaceDB)

Distance between blocks is not fixed — it is **learned**:

```
dist(A, B) = sqrt( (A - B)^T  W  (A - B) )
```

`W` starts as an identity matrix (standard Euclidean).
Every `reinforce()` call trains `W` to pull two blocks closer.
Every `decay()` call pushes them apart.

---

## Custom Embeddings

Bring your own encoder — just produce `np.ndarray` vectors:

```python
# OpenAI
resp = openai.embeddings.create(input=["hello"], model="text-embedding-3-small")
vec = np.array(resp.data[0].embedding, dtype=np.float32)

# HuggingFace
from transformers import AutoModel, AutoTokenizer
vec = model(**tokenizer("hello", return_tensors="pt")).last_hidden_state.mean(1).detach().numpy()[0]

# Any model — just match the dim
client = SpaceClient("./data", dim=len(vec))
```

See [`examples/custom_embeddings.py`](examples/custom_embeddings.py) for a full working example.

---

## Exceptions

All exceptions inherit from `SpaceDBError`:

| Exception | When |
|---|---|
| `BlockNotFoundError` | Block ID doesn't exist |
| `VectorDimensionError` | Vector dim ≠ space dim |
| `StoreCorruptedError` | Persisted data is corrupt |
| `ClusterNotFoundError` | Cluster ID doesn't exist |
| `EmbedderNotAvailableError` | `sentence-transformers` not installed and text was passed |

```python
from spacedb import SpaceDBError, VectorDimensionError
```

---

## Shell (CLI)

```bash
spacesh ./my_data
```

```
space> use demo
space[demo]> ingest "apple tastes sweet"
space[demo]> query "fruit"
space[demo]> drift on
space[demo]> status
```

---

## Development

```bash
git clone https://github.com/nihar840/theSpaceDB.git
cd theSpaceDB
pip install -e ".[dev]"
pytest
```

---

## Project Structure

```
theSpaceDB/
├── spacedb/
│   ├── __init__.py         ← public exports
│   ├── client.py           ← SpaceClient (entry point)
│   ├── space.py            ← Space (per-mind operations)
│   ├── query.py            ← QueryBuilder (chainable)
│   ├── drift.py            ← DriftController
│   ├── shell.py            ← spacesh CLI shell
│   └── _core/
│       ├── engine.py           ← SpaceEngine (orchestrator)
│       ├── models.py           ← MemoryBlock, ClusterData
│       ├── block_store.py      ← append-only block log
│       ├── vector_store.py     ← raw embeddings
│       ├── distance_engine.py  ← W matrix (the soul)
│       ├── graph_store.py      ← adjacency + random walk
│       ├── cluster_registry.py ← spirit size + personalities
│       ├── _exceptions.py      ← exception hierarchy
│       └── _version.py         ← data format version
├── tests/                  ← 53 pytest tests
├── examples/
│   ├── hello_space.py
│   └── custom_embeddings.py
├── pyproject.toml
└── LICENSE
```

---

## Requirements

- Python 3.10+
- numpy ≥ 1.24
- Any OS (Windows, macOS, Linux)

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built as part of the Draeghir — Cognitive Mind Architecture (CMA) project.*
*Nothing fixed. Everything evolving. [inf]*
