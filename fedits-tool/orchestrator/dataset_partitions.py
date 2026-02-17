# orchestrator/dataset_partitions.py
from __future__ import annotations
import json
import os
import time
import random
from typing import Any, Dict, List


def default_partition_path(out_dir_abs: str, dataset: str, num_veh: int, seed: int, scheme: str) -> str:
    os.makedirs(os.path.join(out_dir_abs, "partitions"), exist_ok=True)
    fname = f"{dataset}_N{num_veh}_seed{seed}_{scheme}.json"
    return os.path.join(out_dir_abs, "partitions", fname)


def ensure_iid_partitions(path: str, dataset: str, num_veh: int, seed: int, n_train: int) -> Dict[str, Any]:
    """
    partitions[i] is a list of sample indices for vehicle i.
    If num_veh > n_train, some partitions will be empty (valid).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {})
        if (
            meta.get("dataset") == dataset
            and meta.get("scheme") == "iid"
            and int(meta.get("num_veh", -1)) == int(num_veh)
            and int(meta.get("seed", -2)) == int(seed)
            and int(meta.get("n_train", -3)) == int(n_train)
        ):
            return obj  # reuse

    idxs = list(range(int(n_train)))
    rnd = random.Random(int(seed))
    rnd.shuffle(idxs)

    base = int(n_train) // int(num_veh)
    rem = int(n_train) % int(num_veh)
    parts: List[List[int]] = []
    cur = 0
    for i in range(int(num_veh)):
        size = base + (1 if i < rem else 0)
        parts.append(idxs[cur : cur + size])
        cur += size

    obj = {
        "meta": {
            "dataset": dataset,
            "scheme": "iid",
            "num_veh": int(num_veh),
            "seed": int(seed),
            "n_train": int(n_train),
            "created_at": int(time.time()),
        },
        "partitions": parts,
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)
    return obj
