"""
hello_space.py — theSpaceDB quickstart (raw vectors, no extra dependencies)

Run:  python examples/hello_space.py
"""

import tempfile
import numpy as np
from spacedb import SpaceClient

DIM = 32  # tiny dimension for demo — use 384/768/1536 in production


def random_vec() -> np.ndarray:
    """Generate a random unit vector."""
    v = np.random.randn(DIM).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


# --1. Connect ----------------------------------------------------------------─
data_dir = tempfile.mkdtemp(prefix="spacedb_demo_")
client = SpaceClient(data_dir, dim=DIM)

# --2. Open a space (like a MongoDB database) --------------------------------─
mind = client["demo_mind"]

# --3. Ingest memories (raw vectors) ------------------------------------------
print("Ingesting memories...")
blocks = []
labels = [
    "apple tastes sweet",
    "kitchen smells like food",
    "mother cooking in the kitchen",
    "writing code at night",
    "debugging a strange error",
    "philosophy of consciousness",
    "what is the meaning of existence",
    "music makes me feel alive",
]
for label in labels:
    b = mind.ingest(random_vec(), sensory_type="text")
    blocks.append(b)
    print(f"  [{b.id[:8]}]  {label}")
print(f"\n  Stored {len(blocks)} blocks.\n")

# --4. Reinforce connections ------------------------------------------------─
mind.reinforce(blocks[0], blocks[1])   # apple <-> kitchen
mind.reinforce(blocks[1], blocks[2])   # kitchen <-> mother
mind.reinforce(blocks[3], blocks[4])   # code <-> debugging
print("Reinforced connections.\n")

# --5. Create named clusters ------------------------------------------------─
mind.create_cluster([blocks[0], blocks[1], blocks[2]], name="food_memories")
mind.create_cluster([blocks[3], blocks[4]],            name="coding_mind")
mind.create_cluster([blocks[5], blocks[6]],            name="philosophy")
print("Created 3 clusters.\n")

# --6. Query ------------------------------------------------------------------
query_vec = random_vec()

print("--Query: random vector (100ms budget, surface memory) --")
results = mind.query(query_vec).within(ms=100).limit(5).fetch()
for r in results:
    print(f"  [{r['id'][:8]}]  {r['token']:<30}  score={r['score']:.4f}")

print()
print("--Query: same vector (800ms budget, deep memory) --")
results = mind.query(query_vec).within(ms=800).limit(10).fetch()
for r in results:
    print(f"  [{r['id'][:8]}]  {r['token']:<30}  score={r['score']:.4f}")

# --7. Status ------------------------------------------------------------------
print()
print("--Status --")
s = mind.status()
for k, v in s.items():
    print(f"  {k:<22} {v}")

# --8. Clusters --------------------------------------------------------------
print()
print("--Clusters --")
for c in mind.clusters.all():
    print(f"  {c['name'] or c['id']:<20}  blocks={c['blocks']}  "
          f"spirit={c['spirit_size']:.4f}  personality={c['personality']}")

# --9. Drift ------------------------------------------------------------------
print()
mind.drift.start(idle_seconds=5)
print(f"Drift active: {mind.drift.active}")
mind.drift.stop()
print(f"Drift active: {mind.drift.active}")

print(f"\nData stored in: {data_dir}")
print("Done!")
