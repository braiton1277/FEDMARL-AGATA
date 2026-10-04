
import fcntl
import hashlib
import pickle
from pathlib import Path
from ultralytics.data.dataset import YOLODataset

_original_load = YOLODataset._load_or_scan_cache

def load_cache_local(self, cache_path, cache_hash):
    root = Path("/tmp/yolo_label_cache")
    root.mkdir(parents=True, exist_ok=True)

    # Separa conjuntos diferentes, inclusive treino e validação.
    key = hashlib.sha256(
        (str(cache_path) + "\n" + str(cache_hash)).encode()
    ).hexdigest()
    local_path = root / (key + ".cache")
    lock_path = root / (key + ".lock")

    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            try:
                return _original_load(self, local_path, cache_hash)
            except (pickle.UnpicklingError, EOFError, ValueError) as error:
                print(
                    f"[CACHE REBUILD] arquivo={local_path} erro={error!r}",
                    flush=True,
                )
                local_path.unlink(missing_ok=True)
                return _original_load(self, local_path, cache_hash)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

YOLODataset._load_or_scan_cache = load_cache_local
