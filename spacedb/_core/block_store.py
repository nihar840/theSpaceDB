import os, struct, pickle, threading, logging
from typing import Optional
from .models import MemoryBlock
from ._exceptions import StoreCorruptedError

logger = logging.getLogger("spacedb.block_store")

_HDR = struct.Struct('>I')


class BlockStore:
    def __init__(self, path: str):
        self._log = os.path.join(path, 'blocks.log')
        self._idx = os.path.join(path, 'blocks.idx')
        self._tomb_path = os.path.join(path, 'blocks.tomb')
        self._offsets: dict[str, int] = {}
        self._tombstones: set[str] = set()
        self._lock = threading.Lock()
        self._load_idx()
        self._load_tombstones()

    def append(self, block: MemoryBlock) -> str:
        with self._lock:
            data = pickle.dumps(block)
            with open(self._log, 'ab') as f:
                offset = f.tell()
                f.write(_HDR.pack(len(data)) + data)
            self._offsets[block.id] = offset
            self._save_idx()
            return block.id

    def update(self, block: MemoryBlock):
        with self._lock:
            data = pickle.dumps(block)
            with open(self._log, 'ab') as f:
                offset = f.tell()
                f.write(_HDR.pack(len(data)) + data)
            self._offsets[block.id] = offset
            self._save_idx()

    def read(self, block_id: str) -> Optional[MemoryBlock]:
        with self._lock:
            if block_id not in self._offsets or block_id in self._tombstones:
                return None
            with open(self._log, 'rb') as f:
                f.seek(self._offsets[block_id])
                length = _HDR.unpack(f.read(_HDR.size))[0]
                raw = f.read(length)
                if len(raw) != length:
                    raise StoreCorruptedError("Block data truncated")
                try:
                    return pickle.loads(raw)
                except (pickle.UnpicklingError, EOFError, struct.error) as exc:
                    raise StoreCorruptedError(
                        f"Corrupt block {block_id!r}: {exc}"
                    ) from exc

    def all_ids(self) -> list[str]:
        return list(self._offsets.keys())

    def alive_ids(self) -> set[str]:
        """Return set of block IDs that are not tombstoned."""
        return set(self._offsets.keys()) - self._tombstones

    def count(self) -> int:
        return len(self._offsets)

    def alive_count(self) -> int:
        """Blocks minus tombstones."""
        return len(self._offsets) - len(self._tombstones)

    # -- deletion support -------------------------------------------------------

    def soft_delete(self, block_id: str):
        """Mark block as deleted (tombstone). O(1). Does not rewrite log."""
        with self._lock:
            if block_id not in self._offsets:
                return
            self._tombstones.add(block_id)
            self._save_tombstones()

    def is_deleted(self, block_id: str) -> bool:
        """Check if block is tombstoned."""
        return block_id in self._tombstones

    def compact(self):
        """Rewrite blocks.log without tombstoned blocks. Offline operation.

        1. Read all alive blocks from current log
        2. Write to blocks.log.tmp
        3. Rebuild offsets (only alive blocks)
        4. Atomic rename tmp -> log
        5. Clear tombstones
        6. Save new index
        """
        with self._lock:
            if not self._tombstones:
                return  # nothing to compact

            alive_ids = set(self._offsets.keys()) - self._tombstones
            tmp_path = self._log + '.tmp'
            new_offsets = {}

            with open(tmp_path, 'wb') as out:
                for bid in alive_ids:
                    block = self._read_unlocked(bid)
                    if block is None:
                        continue
                    data = pickle.dumps(block)
                    offset = out.tell()
                    out.write(_HDR.pack(len(data)) + data)
                    new_offsets[bid] = offset

            # Atomic swap
            if os.path.exists(self._log):
                os.replace(tmp_path, self._log)
            else:
                os.rename(tmp_path, self._log)

            # Update state
            self._offsets = new_offsets
            self._tombstones.clear()
            self._save_idx()
            self._save_tombstones()
            logger.info("Compacted block store: %d blocks remain", len(new_offsets))

    def _read_unlocked(self, block_id: str) -> Optional[MemoryBlock]:
        """Read a block without acquiring lock (caller must hold lock)."""
        if block_id not in self._offsets:
            return None
        with open(self._log, 'rb') as f:
            f.seek(self._offsets[block_id])
            length = _HDR.unpack(f.read(_HDR.size))[0]
            raw = f.read(length)
            if len(raw) != length:
                return None
            try:
                return pickle.loads(raw)
            except Exception:
                return None

    def _load_tombstones(self):
        if os.path.exists(self._tomb_path):
            try:
                with open(self._tomb_path, 'rb') as f:
                    self._tombstones = pickle.load(f)
            except Exception as exc:
                logger.warning("Corrupt tombstone file, resetting: %s", exc)
                self._tombstones = set()

    def _save_tombstones(self):
        with open(self._tomb_path, 'wb') as f:
            pickle.dump(self._tombstones, f)

    def _load_idx(self):
        if os.path.exists(self._idx):
            try:
                with open(self._idx, 'rb') as f:
                    self._offsets = pickle.load(f)
            except (pickle.UnpicklingError, EOFError, Exception) as exc:
                logger.warning("Corrupt block index, resetting: %s", exc)
                self._offsets = {}

    def _save_idx(self):
        with open(self._idx, 'wb') as f:
            pickle.dump(self._offsets, f)
