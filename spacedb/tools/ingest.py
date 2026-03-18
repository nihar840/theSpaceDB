"""
spacedb.tools.ingest — Bulk knowledge ingestion tool for theSpaceDB

Reads text files from a directory, embeds each line via Ollama,
and ingests them into a SpaceDB space as high-weight memory blocks.

Each non-empty line in a .txt file becomes one memory block with
sensory_type="lesson" and boosted reinforcement score.

Usage:
    python -m spacedb.tools.ingest \\
        --dir ./knowledge \\
        --db ./data \\
        --space my_mind \\
        --model bge-m3 \\
        --dim 1024

    # Or if installed via pip:
    spacesh-ingest --dir ./knowledge --db ./data --space my_mind

Handles:
    - Unicode sanitization (em dashes, smart quotes, etc.)
    - Ollama bge-m3 NaN bug workaround (retry with simplified text)
    - Retry logic (3 attempts per fact)
    - Progress reporting per domain
    - Bulk mode for fast ingestion with re-clustering at the end
"""

import sys
import os
import time
import glob
import argparse
import logging

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spacedb.ingest")


# ── Text Sanitization ───────────────────────────────────────────


def sanitize(text: str) -> str:
    """
    Replace Unicode characters that can cause embedding model issues.

    Some embedding models (notably bge-m3 via Ollama) produce NaN vectors
    for certain Unicode characters. This function replaces them with safe
    ASCII equivalents while preserving meaning.
    """
    return (text
            .replace("\u2014", " - ")   # em dash
            .replace("\u2013", " - ")   # en dash
            .replace("\u2019", "'")     # right single quote
            .replace("\u2018", "'")     # left single quote
            .replace("\u201c", '"')     # left double quote
            .replace("\u201d", '"')     # right double quote
            .replace("\u2026", "...")   # ellipsis
            .replace("\u00d7", "x")     # multiplication sign
            .replace("\u2248", "~")     # approximately equal
            .replace("\u00b2", "^2")    # superscript 2
            .replace("\u00b3", "^3")    # superscript 3
            .replace("\u03c0", "pi")    # pi symbol
            .replace("\u2192", "->")    # right arrow
            .replace("\u2190", "<-")    # left arrow
            .replace("\u2265", ">=")    # greater than or equal
            .replace("\u2264", "<=")    # less than or equal
            .replace("\u00b0", " degrees")  # degree sign
            )


# ── File Loading ─────────────────────────────────────────────────


def load_text_files(directory: str) -> dict:
    """
    Load all .txt files from a directory.

    Each file becomes a domain (named after the filename).
    Each non-empty, non-comment line becomes one fact.

    Returns:
        dict[str, list[str]]: {domain_name: [fact1, fact2, ...]}
    """
    files = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.txt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, "r", encoding="utf-8") as f:
            lines = [
                sanitize(line.strip())
                for line in f.readlines()
                if line.strip() and not line.startswith("#")
            ]
        if lines:
            files[name] = lines
            log.info("  %-30s -> %d entries", name, len(lines))
    return files


# ── Embedding ────────────────────────────────────────────────────


def create_embedder(base_url: str, model: str):
    """
    Create an embedding function that calls Ollama's /api/embed endpoint.

    Handles the bge-m3 NaN bug by retrying with simplified text
    when the model returns NaN embeddings.

    Returns:
        callable: embed(text) -> list[float]
    """
    import requests

    embed_url = f"{base_url}/api/embed"

    def embed(text: str) -> list:
        r = requests.post(embed_url,
                          json={"model": model, "input": text},
                          timeout=60)

        # Handle NaN bug: some models produce NaN for certain inputs
        if r.status_code == 500 and "NaN" in r.text:
            simple = (text.replace(",", "").replace(";", "")
                      .replace("(", "").replace(")", ""))
            r2 = requests.post(embed_url,
                               json={"model": model, "input": simple},
                               timeout=60)
            if r2.status_code == 200:
                return r2.json()["embeddings"][0]
            raise RuntimeError(
                f"Embedding model NaN bug for: {text[:50]}...")

        r.raise_for_status()
        return r.json()["embeddings"][0]

    return embed


# ── Main Ingestion ───────────────────────────────────────────────


def ingest(directory: str, db_path: str, space_name: str,
           model: str = "bge-m3", dim: int = 1024,
           base_url: str = "http://localhost:11434",
           sensory_type: str = "lesson",
           reinforce_boost: float = 3.0,
           max_retries: int = 3):
    """
    Ingest all .txt files from a directory into a SpaceDB space.

    Parameters
    ----------
    directory : str
        Path to directory containing .txt files.
    db_path : str
        Path to SpaceDB data directory.
    space_name : str
        Name of the space to ingest into.
    model : str
        Ollama embedding model name (default: bge-m3).
    dim : int
        Embedding dimension (default: 1024 for bge-m3).
    base_url : str
        Ollama server URL (default: http://localhost:11434).
    sensory_type : str
        Sensory type for ingested blocks (default: "lesson").
    reinforce_boost : float
        Reinforcement score for ingested blocks (default: 3.0).
    max_retries : int
        Max retry attempts per fact (default: 3).

    Returns
    -------
    dict
        Summary with keys: ingested, errors, time_s, domains.
    """
    from spacedb import SpaceClient

    # Load files
    log.info("Loading text files from: %s", directory)
    knowledge = load_text_files(directory)
    total = sum(len(v) for v in knowledge.values())
    log.info("Total: %d entries across %d domains\n", total, len(knowledge))

    if total == 0:
        log.warning("No text files found in %s", directory)
        return {"ingested": 0, "errors": 0, "time_s": 0, "domains": 0}

    # Connect to embedding model
    log.info("Connecting to Ollama (%s @ %s)...", model, base_url)
    embed = create_embedder(base_url, model)

    # Quick test
    try:
        test_vec = embed("test")
        actual_dim = len(test_vec)
        log.info("  Embedding OK (dim=%d)", actual_dim)
        if actual_dim != dim:
            log.warning("  Expected dim=%d but got %d. Using %d.",
                        dim, actual_dim, actual_dim)
            dim = actual_dim
    except Exception as e:
        log.error("Ollama not available: %s", e)
        log.error("Make sure Ollama is running with model '%s'", model)
        return {"ingested": 0, "errors": 0, "time_s": 0, "domains": 0}

    # Connect to SpaceDB
    log.info("Connecting to SpaceDB (db=%s, space=%s)...", db_path, space_name)
    client = SpaceClient(db_path, dim=dim, silent=True)
    space = client[space_name]
    log.info("  Connected: %s\n", space)

    # Ingest
    log.info("--- Starting Ingestion ---\n")
    t0 = time.time()
    ingested = 0
    errors = 0

    space.begin_bulk()

    for domain, facts in knowledge.items():
        log.info("[%s] %d entries...", domain, len(facts))
        domain_t0 = time.time()

        for i, fact in enumerate(facts):
            for attempt in range(max_retries):
                try:
                    raw = embed(fact)
                    vec = np.array(raw, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec /= norm

                    block = space.ingest_fast(
                        vec,
                        sensory_type=sensory_type,
                        raw_input=fact,
                        normalized_content=fact,
                    )

                    # Boost reinforcement for authoritative knowledge
                    try:
                        block.reinforcement_score = reinforce_boost
                        block.metadata["domain"] = domain
                        block.metadata["source"] = "bulk_ingest"
                    except Exception:
                        pass

                    ingested += 1
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.0)
                    else:
                        log.warning("  Skip: %s -- %s", fact[:40], e)
                        errors += 1

            # Breathing room for the embedding server
            time.sleep(0.05)

            if (i + 1) % 10 == 0:
                log.info("    %d/%d", i + 1, len(facts))

        elapsed = time.time() - domain_t0
        log.info("  Done in %.1fs\n", elapsed)

    log.info("Re-clustering (this may take a moment)...")
    space.end_bulk()

    total_time = time.time() - t0
    rate = ingested / total_time if total_time > 0 else 0

    log.info("\n" + "=" * 50)
    log.info("INGESTION COMPLETE")
    log.info("=" * 50)
    log.info("  Ingested:  %d", ingested)
    log.info("  Skipped:   %d", errors)
    log.info("  Time:      %.1fs", total_time)
    log.info("  Rate:      %.1f entries/sec", rate)
    log.info("  SpaceDB:   %s", space)

    return {
        "ingested": ingested,
        "errors": errors,
        "time_s": total_time,
        "domains": len(knowledge),
    }


# ── CLI ──────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        prog="spacesh-ingest",
        description="Bulk ingest text files into a SpaceDB space.",
        epilog=(
            "Example:\n"
            "  spacesh-ingest --dir ./knowledge --db ./data --space my_mind\n"
            "\n"
            "Each .txt file in the directory becomes a domain.\n"
            "Each non-empty line becomes one memory block.\n"
            "Lines starting with # are treated as comments and skipped."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--dir", required=True,
        help="Directory containing .txt knowledge files")
    parser.add_argument(
        "--db", required=True,
        help="Path to SpaceDB data directory")
    parser.add_argument(
        "--space", required=True,
        help="Name of the space to ingest into")
    parser.add_argument(
        "--model", default="bge-m3",
        help="Ollama embedding model (default: bge-m3)")
    parser.add_argument(
        "--dim", type=int, default=1024,
        help="Embedding dimension (default: 1024)")
    parser.add_argument(
        "--url", default="http://localhost:11434",
        help="Ollama server URL (default: http://localhost:11434)")
    parser.add_argument(
        "--type", default="lesson", dest="sensory_type",
        help="Sensory type for blocks (default: lesson)")
    parser.add_argument(
        "--boost", type=float, default=3.0,
        help="Reinforcement score boost (default: 3.0)")
    parser.add_argument(
        "--retries", type=int, default=3,
        help="Max retries per entry (default: 3)")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        log.error("Directory not found: %s", args.dir)
        sys.exit(1)

    result = ingest(
        directory=args.dir,
        db_path=args.db,
        space_name=args.space,
        model=args.model,
        dim=args.dim,
        base_url=args.url,
        sensory_type=args.sensory_type,
        reinforce_boost=args.boost,
        max_retries=args.retries,
    )

    sys.exit(0 if result["errors"] == 0 else 1)


if __name__ == "__main__":
    main()
