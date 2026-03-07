"""
hello_space.py — theSpaceDB quickstart

Run: python examples/hello_space.py
"""

from spacedb import SpaceClient

# ── 1. Connect ────────────────────────────────────────────────────────────────
client = SpaceClient("D:/SpaceDB/data")

# ── 2. Open a space (like a MongoDB database) ─────────────────────────────────
mind = client["demo_mind"]

# ── 3. Ingest memories ───────────────────────────────────────────────────────
print("Ingesting memories...")
blocks = [
    mind.ingest("apple tastes sweet"),
    mind.ingest("kitchen smells like food"),
    mind.ingest("mother cooking in the kitchen"),
    mind.ingest("writing code at night"),
    mind.ingest("debugging a strange error"),
    mind.ingest("philosophy of consciousness"),
    mind.ingest("what is the meaning of existence"),
    mind.ingest("music makes me feel alive"),
]
print(f"  Stored {len(blocks)} blocks.\n")

# ── 4. Reinforce connections ──────────────────────────────────────────────────
mind.reinforce(blocks[0], blocks[1])   # apple ↔ kitchen
mind.reinforce(blocks[1], blocks[2])   # kitchen ↔ mother
mind.reinforce(blocks[3], blocks[4])   # code ↔ debugging

# ── 5. Create named clusters ──────────────────────────────────────────────────
mind.create_cluster([blocks[0], blocks[1], blocks[2]], name="food_memories")
mind.create_cluster([blocks[3], blocks[4]],            name="coding_mind")
mind.create_cluster([blocks[5], blocks[6]],            name="philosophy")

# ── 6. Query ─────────────────────────────────────────────────────────────────
print("── Query: 'fruit'  (100ms budget, surface memory) ──")
results = mind.query("fruit").within(ms=100).limit(5).fetch()
for r in results:
    print(f"  [{r['id'][:8]}]  {r['token']:<40}  score={r['score']:.4f}")

print()
print("── Query: 'fruit'  (800ms budget, deep memory) ──")
results = mind.query("fruit").within(ms=800).limit(10).fetch()
for r in results:
    print(f"  [{r['id'][:8]}]  {r['token']:<40}  score={r['score']:.4f}")

# ── 7. Status ─────────────────────────────────────────────────────────────────
print()
print("── Status ──")
s = mind.status()
for k, v in s.items():
    print(f"  {k:<22} {v}")

# ── 8. Start drift ───────────────────────────────────────────────────────────
print()
print("── Drift ──")
print("  Starting cognitive drift engine (idle threshold: 5s)")
print("  The mind will start 'thinking' when idle...")
mind.drift.start(idle_seconds=5)
print(f"  Drift active: {mind.drift.active}")
