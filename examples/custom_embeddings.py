"""
custom_embeddings.py — Using theSpaceDB with your own embedding model

This example shows how to bring any embedding model (OpenAI, HuggingFace,
ONNX, etc.) and pass raw numpy vectors to SpaceDB.

Run:  python examples/custom_embeddings.py
"""

import tempfile
import numpy as np
from spacedb import SpaceClient


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Define your embedding function
# ──────────────────────────────────────────────────────────────────────────────
# Replace this with your real encoder. Examples:
#
#   # OpenAI
#   import openai
#   def embed(text: str) -> np.ndarray:
#       resp = openai.embeddings.create(input=[text], model="text-embedding-3-small")
#       return np.array(resp.data[0].embedding, dtype=np.float32)
#
#   # HuggingFace sentence-transformers
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer("all-MiniLM-L6-v2")
#   def embed(text: str) -> np.ndarray:
#       return model.encode([text], normalize_embeddings=True)[0]
#
#   # ONNX / TensorRT / any model
#   def embed(text: str) -> np.ndarray:
#       tokens = tokenizer(text)
#       output = onnx_session.run(None, tokens)
#       return output[0].mean(axis=1).squeeze().astype(np.float32)

# For this demo, we'll use a fake encoder:
DIM = 128

def embed(text: str) -> np.ndarray:
    """Fake embedding: hash-based deterministic vector for demo purposes."""
    rng = np.random.RandomState(hash(text) % (2**31))
    vec = rng.randn(DIM).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Create a SpaceDB client with matching dimension
# ──────────────────────────────────────────────────────────────────────────────
data_dir = tempfile.mkdtemp(prefix="spacedb_custom_")
client = SpaceClient(data_dir, dim=DIM)
mind = client["custom_demo"]


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Ingest with raw vectors
# ──────────────────────────────────────────────────────────────────────────────
texts = [
    "the cat sat on the mat",
    "dogs love to play fetch",
    "quantum mechanics is strange",
    "neural networks learn patterns",
    "the sun rises in the east",
    "cooking pasta requires boiling water",
]

blocks = []
for text in texts:
    vec = embed(text)
    block = mind.ingest(vec, sensory_type="text")
    blocks.append((text, block))
    print(f"  Ingested: {text:<45} -> {block.id[:8]}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Query with raw vectors
# ──────────────────────────────────────────────────────────────────────────────
query_text = "animals playing"
query_vec = embed(query_text)

print(f"\nQuery: '{query_text}'")
results = mind.query(query_vec).within(ms=200).limit(3).fetch()

for i, r in enumerate(results, 1):
    # Find original text from token or blocks
    original = next((t for t, b in blocks if b.id == r["id"]), r["token"])
    print(f"  {i}. [{r['id'][:8]}] score={r['score']:.4f}  {original}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 5: Reinforce + re-query to see distance evolve
# ──────────────────────────────────────────────────────────────────────────────
cat_block = blocks[0][1]   # "the cat sat on the mat"
dog_block = blocks[1][1]   # "dogs love to play fetch"

print(f"\nReinforcing: cat <-> dog  (50 rounds)")
for _ in range(50):
    mind.reinforce(cat_block, dog_block, strength=0.02)

print("Re-querying after reinforcement...")
results = mind.query(query_vec).within(ms=200).limit(3).fetch()
for i, r in enumerate(results, 1):
    original = next((t for t, b in blocks if b.id == r["id"]), r["token"])
    print(f"  {i}. [{r['id'][:8]}] score={r['score']:.4f}  {original}")


print(f"\nData stored in: {data_dir}")
print("Done!")
