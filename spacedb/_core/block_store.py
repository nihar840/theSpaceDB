import os, struct, pickle
from typing import Optional
from .models import MemoryBlock

_HDR = struct.Struct('>I')


class BlockStore:
    def __init__(self, path: str):
        self._log = os.path.join(path, 'blocks.log')
        self._idx = os.path.join(path, 'blocks.idx')
        self._offsets: dict[str, int] = {}
        self._load_idx()

    def append(self, block: MemoryBlock) -> str:
        data = pickle.dumps(block)
        with open(self._log, 'ab') as f:
            offset = f.tell()
            f.write(_HDR.pack(len(data)) + data)
        self._offsets[block.id] = offset
        self._save_idx()
        return block.id

    def update(self, block: MemoryBlock):
        data = pickle.dumps(block)
        with open(self._log, 'ab') as f:
            offset = f.tell()
            f.write(_HDR.pack(len(data)) + data)
        self._offsets[block.id] = offset
        self._save_idx()

    def read(self, block_id: str) -> Optional[MemoryBlock]:
        if block_id not in self._offsets:
            return None
        with open(self._log, 'rb') as f:
            f.seek(self._offsets[block_id])
            length = _HDR.unpack(f.read(_HDR.size))[0]
            return pickle.loads(f.read(length))

    def all_ids(self) -> list[str]:
        return list(self._offsets.keys())

    def count(self) -> int:
        return len(self._offsets)

    def _load_idx(self):
        if os.path.exists(self._idx):
            with open(self._idx, 'rb') as f:
                self._offsets = pickle.load(f)

    def _save_idx(self):
        with open(self._idx, 'wb') as f:
            pickle.dump(self._offsets, f)
