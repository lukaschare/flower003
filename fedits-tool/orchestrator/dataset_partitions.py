# orchestrator/dataset_partitions.py
from __future__ import annotations
import json
import os
import time
import random
from typing import Any, Dict, List, Optional

import numpy as np


def default_partition_path(out_dir_abs: str, dataset: str, num_veh: int, seed: int, scheme: str) -> str:
    os.makedirs(os.path.join(out_dir_abs, "partitions"), exist_ok=True)
    fname = f"{dataset}_N{num_veh}_seed{seed}_{scheme}.json"
    return os.path.join(out_dir_abs, "partitions", fname)


def _atomic_dump_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def ensure_iid_partitions(path: str, dataset: str, num_veh: int, seed: int, n_train: int) -> Dict[str, Any]:
    """Equal-size IID split over indices [0..n_train-1]."""
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
            return obj

    idxs = list(range(int(n_train)))
    rnd = random.Random(int(seed))
    rnd.shuffle(idxs)

    base = int(n_train) // int(num_veh)
    rem = int(n_train) % int(num_veh)
    parts: List[List[int]] = []
    cur = 0
    for i in range(int(num_veh)):
        size = base + (1 if i < rem else 0)
        parts.append(idxs[cur:cur + size])
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
    _atomic_dump_json(path, obj)
    return obj


def _load_cifar10_train_labels(data_dir: str) -> np.ndarray:
    # Import locally so orchestrator can still run if someone swaps dataset later
    import torchvision
    import torchvision.transforms as T

    ds = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=T.ToTensor(),
    )
    # torchvision CIFAR10 has `targets` as list[int]
    return np.asarray(ds.targets, dtype=np.int64)


def ensure_dirichlet_partitions(
    path: str,
    dataset: str,
    num_veh: int,
    seed: int,
    data_dir: str,
    alpha: float = 0.5,
    n_train: int = 50_000,
) -> Dict[str, Any]:
    """
    Label-skew non-IID split using Dirichlet(alpha) over class proportions.
    - Supports empty partitions (important when num_veh is large).
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        meta = obj.get("meta", {})
        if (
            meta.get("dataset") == dataset
            and meta.get("scheme") == "dirichlet"
            and int(meta.get("num_veh", -1)) == int(num_veh)
            and int(meta.get("seed", -2)) == int(seed)
            and int(meta.get("n_train", -3)) == int(n_train)
            and abs(float(meta.get("alpha", -9.9)) - float(alpha)) < 1e-12
        ):
            return obj

    if dataset.lower() != "cifar10":
        raise ValueError(f"ensure_dirichlet_partitions currently supports dataset=cifar10 only, got {dataset}")

    labels = _load_cifar10_train_labels(data_dir)
    if int(n_train) != len(labels):
        # keep it strict so your meta always matches the real dataset length
        raise ValueError(f"n_train={n_train} but CIFAR-10 train labels len={len(labels)}")

    n_classes = int(labels.max()) + 1
    rng = np.random.default_rng(int(seed))

    parts: List[List[int]] = [[] for _ in range(int(num_veh))]

    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)

        # Dirichlet proportions across vehicles for this class
        p = rng.dirichlet(alpha=np.full(int(num_veh), float(alpha), dtype=np.float64))

        # convert p into split sizes (keep sum exact)
        counts = (p * len(idx_c)).astype(int)
        diff = len(idx_c) - int(counts.sum())
        if diff > 0:
            # distribute the leftover to top-prob vehicles (deterministic for given seed)
            top = np.argsort(-p)[:diff]
            counts[top] += 1

        # split and append
        cursor = 0
        for i in range(int(num_veh)):
            k = int(counts[i])
            if k > 0:
                parts[i].extend(idx_c[cursor:cursor + k].tolist())
                cursor += k

    # shuffle within each vehicle to avoid class blocks
    for i in range(int(num_veh)):
        rng.shuffle(parts[i])

    obj = {
        "meta": {
            "dataset": dataset,
            "scheme": "dirichlet",
            "alpha": float(alpha),
            "num_veh": int(num_veh),
            "seed": int(seed),
            "n_train": int(n_train),
            "created_at": int(time.time()),
            "data_dir": data_dir,
        },
        "partitions": parts,
    }
    _atomic_dump_json(path, obj)
    return obj
