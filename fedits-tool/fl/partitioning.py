# fl/partitioning.py
from __future__ import annotations
import json, os, time, re
from typing import Any, Dict, List
import numpy as np

def veh_id_to_index(veh_id: str, num_veh: int) -> int:
    m = re.search(r"(\d+)$", veh_id or "")
    if m:
        return int(m.group(1)) % max(num_veh, 1)
    return (abs(hash(veh_id)) % max(num_veh, 1))

def ensure_iid_partitions_json(
    path: str,
    num_veh: int,
    seed: int,
    n_train: int = 50_000,
) -> Dict[str, Any]:
    """Create/load IID partitions. Stored as list-of-lists partitions[i]=indices for vehicle i."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {})
        if int(meta.get("num_veh", -1)) == int(num_veh) and int(meta.get("seed", -2)) == int(seed):
            return obj  # reuse
        # else: fallthrough and regenerate

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(n_train)).tolist()
    shards = np.array_split(np.array(perm, dtype=np.int64), int(num_veh))
    partitions: List[List[int]] = [s.tolist() for s in shards]

    obj = {
        "meta": {
            "dataset": "cifar10",
            "scheme": "iid",
            "num_veh": int(num_veh),
            "seed": int(seed),
            "n_train": int(n_train),
            "created_at": int(time.time()),
        },
        "partitions": partitions,
    }

    # atomic write
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
    return obj
