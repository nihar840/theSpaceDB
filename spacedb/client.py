"""
client.py — SpaceClient: the entry point to theSpaceDB

Analogous to MongoClient. Manages connection to a SpaceDB data directory
and provides access to individual Space instances (like MongoDB databases).

Usage:
    from spacedb import SpaceClient

    client = SpaceClient("D:/SpaceDB/data")
    mind   = client["my_mind"]          # or client.use("my_mind")

    block  = mind.ingest("apple tastes sweet")
    result = mind.query("fruit").within(ms=200).fetch()

    client.list_spaces()
    client.status()
"""

import os
from typing import Optional
from .space import Space


class SpaceClient:
    """
    Connection to a SpaceDB instance.

    Parameters
    ----------
    path : str
        Root directory where all spaces are stored.
        Each space gets its own sub-directory.
    dim : int
        Embedding dimension. Must match your embedding model.
        Default 384 (all-MiniLM-L6-v2).
    """

    _BANNER = """\033[96m
  ╔══════════════════════════════════════════╗
  ║        theSpaceDB  v0.1.0                ║
  ║        Infinity begins here.             ║
  ╚══════════════════════════════════════════╝\033[0m"""

    def __init__(self, path: str, dim: int = 384, silent: bool = False):
        self._root  = os.path.abspath(path)
        self._dim   = dim
        self._cache: dict[str, Space] = {}
        os.makedirs(self._root, exist_ok=True)
        if not silent:
            print(self._BANNER)
            print(f"\033[90m  Connected to: {self._root}\033[0m\n")

    # ── space access ─────────────────────────────────────────
    def use(self, name: str) -> Space:
        """Open (or create) a Space by name."""
        if name not in self._cache:
            self._cache[name] = Space(name, self._root, self._dim)
        return self._cache[name]

    def __getitem__(self, name: str) -> Space:
        """client["my_mind"]  shorthand for client.use("my_mind")"""
        return self.use(name)

    # ── management ───────────────────────────────────────────
    def list_spaces(self) -> list[str]:
        """List all spaces in the data directory."""
        return [
            d for d in os.listdir(self._root)
            if os.path.isdir(os.path.join(self._root, d))
        ]

    def drop_space(self, name: str, confirm: bool = False):
        """
        Permanently delete a space and all its data.
        Requires confirm=True to prevent accidents.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to drop a space. This is irreversible.")
        import shutil
        path = os.path.join(self._root, name)
        if os.path.exists(path):
            shutil.rmtree(path)
            self._cache.pop(name, None)
            print(f"  Space '{name}' dropped.")
        else:
            print(f"  Space '{name}' not found.")

    def status(self) -> dict:
        """Overall client status."""
        spaces = self.list_spaces()
        return {
            'root':   self._root,
            'spaces': spaces,
            'count':  len(spaces),
        }

    def __repr__(self):
        return f"SpaceClient(root={self._root!r}, spaces={self.list_spaces()})"
