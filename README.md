# theSpaceDB

> Blocks in free space. Distance evolves. Clusters emerge. Personalities form.
> Nothing fixed. Everything evolving. We chase infinity.

A self-evolving cognitive database where memory blocks float in space,
learn from experience, and form personalities over time.

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

### Step 1 — Download

Clone or download this repository to your machine:

```
git clone https://github.com/nihar840/theSpaceDB.git
```

Or download the ZIP from GitHub and extract it anywhere (e.g. `D:\SpaceDB`).

---

### Step 2 — Run the Installer

Navigate to the `installer` folder and double-click:

```
installer\install.bat
```

- Click **Yes** when Windows asks for admin rights
- The installer will automatically:
  - Check for Python 3.11+ (installs via winget if missing)
  - Create a virtual environment
  - Install all dependencies (numpy, scikit-learn, sentence-transformers)
  - Install theSpaceDB
  - Add `spacesh` command to your PATH
  - Create a desktop shortcut

> The first install takes **3-5 minutes** — sentence-transformers is chunky.
> Go drink some water. Seriously.

---

### Step 3 — Verify Installation

After the installer finishes, double-click:

```
installer\verify.bat
```

You should see all **[PASS]** results:

```
  [PASS] Install directory found
  [PASS] Virtual environment found
  [PASS] spacesh launcher found
  [PASS] spacedb package importable  v0.1.0
  [PASS] sentence-transformers installed
  [PASS] spacesh command found in PATH
  [PASS] Desktop shortcut found
```

If anything fails, re-run `install.bat` and it will fix it.

---

### Step 4 — Launch

Open a **new terminal** and type:

```bash
spacesh C:\my_mind
```

Or double-click the **spacesh** shortcut on your Desktop.

---

## Usage

```
  +============================================================+
  |    [inf]  theSpaceDB Shell  v0.1.0                        |
  |           Blocks in free space. We chase infinity.        |
  +============================================================+

  Type 'help' to see available commands.
```

### Commands

| Command | What it does |
|---|---|
| `use <name>` | Open or create a space (like a database) |
| `ingest <text>` | Store a memory block |
| `query <text>` | Find related memories (default: 100ms budget) |
| `query <text> --time 500` | Deeper search with 500ms time budget |
| `reinforce <id1> <id2>` | Pull two blocks closer together |
| `decay <id1> <id2>` | Push two blocks apart |
| `clusters` | List all clusters |
| `personalities` | List personality clusters |
| `drift on` | Start cognitive drift (the DB starts thinking) |
| `drift off` | Stop drift |
| `status` | Show space health |
| `help` | Show all commands |
| `exit` | Exit the shell |

### Quick Demo

```
space> use demo
space[demo]> ingest "apple tastes sweet"
space[demo]> ingest "kitchen smells like food"
space[demo]> ingest "mother cooking in the kitchen"
space[demo]> query "fruit"
space[demo]> drift on
space[demo]> status
```

### Python API

```python
from spacedb import SpaceClient

client = SpaceClient("C:/my_mind")
mind   = client["demo"]

block   = mind.ingest("apple tastes sweet")
results = mind.query("fruit").within(ms=300).limit(10).fetch()

mind.reinforce(block_a, block_b)
mind.drift.start()

print(mind.status())
```

---

## How it Works

```
Input text
    |
    v
Auto-embed (sentence-transformers)
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
    v
Personality check  (spirit_size > threshold → personality emerges)
    |
    v  (when idle)
Cognitive Drift  →  random walk → new connections → cross-cluster links
```

### The W Matrix (Soul of theSpaceDB)

Distance between blocks is not fixed — it is **learned**:

```
dist(A, B) = sqrt( (A - B)^T  W  (A - B) )
```

`W` starts as an identity matrix (pure cosine space).
Every `reinforce()` call trains `W` to pull two blocks closer.
Every `decay()` call pushes them apart.
Rate of separation is proportional to input rate — more inputs = faster drift.

---

## Uninstall

Double-click `installer\uninstall.bat` to remove theSpaceDB completely.

---

## Requirements

- Windows 10 / 11
- Python 3.11+ (auto-installed if missing)
- ~1 GB disk space (sentence-transformers model)
- Internet connection for first install

---

## Project Structure

```
theSpaceDB/
├── installer/
│   ├── install.bat       ← double-click to install
│   ├── install.ps1       ← installer logic
│   ├── verify.bat        ← check installation
│   └── uninstall.bat     ← remove everything
├── spacedb/
│   ├── client.py         ← SpaceClient (entry point)
│   ├── space.py          ← Space (per-mind operations)
│   ├── query.py          ← QueryBuilder (chainable)
│   ├── drift.py          ← DriftController
│   ├── shell.py          ← spacesh CLI shell
│   └── _core/
│       ├── engine.py         ← SpaceEngine (unified internal API)
│       ├── block_store.py    ← append-only block log
│       ├── vector_store.py   ← raw embeddings
│       ├── distance_engine.py← W matrix (the soul)
│       ├── graph_store.py    ← adjacency + random walk
│       └── cluster_registry.py← spirit size + personalities
├── examples/
│   └── hello_space.py
└── pyproject.toml
```

---

*Built as part of the Draeghir — Cognitive Mind Architecture (CMA) project.*
*Nothing fixed. Everything evolving. [inf]*
