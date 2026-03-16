import os, json, math, time, uuid, threading, logging
from typing import Optional
from .models import ClusterData
from ._exceptions import StoreCorruptedError, ClusterNotFoundError

logger = logging.getLogger("spacedb.cluster")


class ClusterRegistry:
    _INIT_THRESH = 5.0
    _MIN_THRESH  = 0.5
    _DECAY       = 0.001

    def __init__(self, path: str):
        self._lock  = threading.Lock()
        self._path  = os.path.join(path, 'clusters.json')
        self._data: dict[str, ClusterData] = {}
        self._exp   = 0
        self._on_personality_change = None   # callback(event_type, cid)
        self._load()

    @property
    def threshold(self) -> float:
        return max(self._INIT_THRESH * math.exp(-self._DECAY * self._exp),
                   self._MIN_THRESH)

    def register(self, block_ids: list[str], name: Optional[str] = None) -> str:
        with self._lock:
            cid = str(uuid.uuid4())[:8]
            self._data[cid] = ClusterData(id=cid, block_ids=list(block_ids), name=name)
            self._save()
            return cid

    def add_block(self, cid: str, bid: str):
        with self._lock:
            c = self._data.get(cid)
            if c and bid not in c.block_ids:
                c.block_ids.append(bid); c.updated_at = time.time(); self._save()

    def remove_block(self, cid: str, bid: str):
        with self._lock:
            c = self._data.get(cid)
            if c and bid in c.block_ids:
                c.block_ids.remove(bid); c.updated_at = time.time(); self._save()

    def dissolve(self, cid: str) -> list[str]:
        with self._lock:
            c = self._data.pop(cid, None); self._save()
            return c.block_ids if c else []

    def update_spirit(self, cid: str, spirit: float):
        with self._lock:
            c = self._data.get(cid)
            if not c: return
            c.spirit_size = spirit; c.updated_at = time.time()
            t = self.threshold
            if not c.is_personality and spirit > t:
                c.is_personality = True
                logger.info("Personality emerged: '%s'  spirit=%.3f", c.name or cid, spirit)
                if self._on_personality_change:
                    self._on_personality_change("promoted", cid)
            elif c.is_personality and spirit < t * 0.5:
                c.is_personality = False
                logger.info("Personality dissolved: '%s'", c.name or cid)
                if self._on_personality_change:
                    self._on_personality_change("demoted", cid)
            self._save()

    def record_experience(self):
        self._exp += 1

    def get(self, cid: str) -> Optional[ClusterData]:
        return self._data.get(cid)

    def all(self) -> list[ClusterData]:
        return list(self._data.values())

    def personalities(self) -> list[ClusterData]:
        return [c for c in self._data.values() if c.is_personality]

    def count(self) -> int:
        return len(self._data)

    def find_by_block(self, bid: str) -> Optional[ClusterData]:
        for c in self._data.values():
            if bid in c.block_ids: return c
        return None

    def _save(self):
        with open(self._path, 'w') as f:
            json.dump({'exp': self._exp,
                       'clusters': {k: vars(v) for k, v in self._data.items()}}, f, indent=2)

    def _load(self):
        if not os.path.exists(self._path): return
        try:
            with open(self._path) as f: d = json.load(f)
            self._exp = d.get('exp', 0)
            for cid, cd in d.get('clusters', {}).items():
                self._data[cid] = ClusterData(**cd)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise StoreCorruptedError(f"Failed to load cluster registry: {e}") from e
