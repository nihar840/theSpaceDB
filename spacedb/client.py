"""
client.py — SpaceClient: the entry point to theSpaceDB

Analogous to MongoClient. Manages connection to a SpaceDB data directory
and provides access to individual Space instances (like MongoDB databases).

Usage::

    from spacedb import SpaceClient

    client = SpaceClient("./data")
    mind   = client["my_mind"]          # or client.use("my_mind")

    block  = mind.ingest("apple tastes sweet")
    result = mind.query("fruit").within(ms=200).fetch()

    client.list_spaces()
    client.status()
"""

import logging
import os
from typing import Optional

from .space import Space

log = logging.getLogger("spacedb.client")


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
    silent : bool
        If ``True``, suppress the startup banner.
    """

    def __init__(self, path: str, dim: int = 384, silent: bool = False,
                 max_size_mb: Optional[float] = None):
        self._root = os.path.abspath(path)
        self._dim  = dim
        self._max_size_mb = max_size_mb   # default quota for new spaces
        self._cache: dict[str, Space] = {}
        os.makedirs(self._root, exist_ok=True)
        if not silent:
            log.info("theSpaceDB connected -> %s", self._root)

    # ── space access ─────────────────────────────────────────
    def use(self, name: str, max_size_mb: Optional[float] = None) -> Space:
        """
        Open (or create) a Space by name.

        Parameters
        ----------
        name : str
            Space name (becomes a subdirectory).
        max_size_mb : float, optional
            Override the client-level default size limit for this space.
            ``None`` inherits the client default. ``0`` means unlimited.
        """
        if name not in self._cache:
            quota = max_size_mb if max_size_mb is not None else self._max_size_mb
            # 0 means explicitly unlimited
            if quota == 0:
                quota = None
            self._cache[name] = Space(name, self._root, self._dim,
                                      max_size_mb=quota)
            log.debug("Opened space %r (quota: %s MB)", name,
                       quota or "unlimited")
        return self._cache[name]

    def __getitem__(self, name: str) -> Space:
        """``client["my_mind"]``  shorthand for ``client.use("my_mind")``."""
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

        Requires ``confirm=True`` to prevent accidents.
        """
        if not confirm:
            raise ValueError(
                "Pass confirm=True to drop a space. This is irreversible."
            )
        import shutil

        path = os.path.join(self._root, name)
        if os.path.exists(path):
            shutil.rmtree(path)
            self._cache.pop(name, None)
            log.info("Space %r dropped.", name)
        else:
            log.warning("Space %r not found.", name)

    def status(self) -> dict:
        """Overall client status."""
        spaces = self.list_spaces()
        return {
            "root":   self._root,
            "spaces": spaces,
            "count":  len(spaces),
        }

    def __repr__(self):
        return f"SpaceClient(root={self._root!r}, spaces={self.list_spaces()})"
