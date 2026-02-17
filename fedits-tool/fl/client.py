#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import flwr as fl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


# -------------------------
# Helpers
# -------------------------
def env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default))

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def joule_to_kwh(j: float) -> float:
    return j / 3_600_000.0

def derive_slot_id() -> str:
    # useful to debug docker-slot <-> physical client id mapping
    return os.getenv("HOSTNAME", "unknown")

def veh_id_to_index(veh_id: str, num_veh: int) -> int:
    """
    Preferred: veh0..veh(N-1) -> index by suffix.
    Fallback: hash -> modulo.
    """
    if num_veh <= 0:
        return 0
    m = re.search(r"(\d+)$", veh_id or "")
    if m:
        return int(m.group(1)) % num_veh
    return (abs(hash(veh_id)) % num_veh)

def load_partitions(partition_path: str) -> dict:
    with open(partition_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_params_from_model(model: nn.Module) -> List[np.ndarray]:
    # keep the state_dict order stable
    return [v.detach().cpu().numpy() for _, v in model.state_dict().items()]

def set_params_to_model(model: nn.Module, params: List[np.ndarray]) -> None:
    sd = model.state_dict()
    keys = list(sd.keys())
    if len(keys) != len(params):
        raise ValueError(f"Parameter length mismatch: {len(keys)} != {len(params)}")
    new_sd = {}
    for k, arr in zip(keys, params):
        new_sd[k] = torch.tensor(arr)
    model.load_state_dict(new_sd, strict=True)


# -------------------------
# Model
# -------------------------
class SimpleCifarCNN(nn.Module):
    """Tiny CNN for CIFAR-10 (3x32x32)."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 64x8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# Flower Client
# -------------------------
class Cifar10PartitionClient(fl.client.NumPyClient):
    """
    Compute-plane client:
    - Receives veh_id and partition_path from server config (Orchestrator-driven).
    - Loads CIFAR-10 partition for that veh_id and trains a simple CNN.
    - Reports metrics: t_train_s, e_comp_j, co2_comp_g, veh_id (needed by orchestrator).
    """

    def __init__(self) -> None:
        # energy model (placeholder; you can later swap to real measurement)
        self.p_comp_w = env_float("P_COMP_W", 18.0)
        self.ci_g_per_kwh = env_float("CI_G_PER_KWH", 200.0)

        # training hyper-params (can be overridden by fit_config)
        self.epochs = env_int("EPOCHS", 1)
        self.batch_size = env_int("BATCH_SIZE", 64)
        self.lr = env_float("LR", 0.01)

        # dataset path (shared volume recommended)
        self.data_dir = env_str("DATA_DIR", "/app/data")

        # cache partition json by path (because different experiments may pass different file)
        self._partition_cache: Dict[str, dict] = {}
        self._loader_cache: Dict[Tuple[str, str], Tuple[DataLoader, int]] = {}  # (partition_path, veh_id) -> (dl, n)

        # device
        self.device = torch.device(env_str("DEVICE", "cpu"))

        # model
        self.model: nn.Module = SimpleCifarCNN().to(self.device)

        # dataset (download into shared dir)
        transform = T.Compose([
            T.ToTensor(),
            # You can add Normalize later if needed
        ])
        self.trainset = torchvision.datasets.CIFAR10(
            # root=self.data_dir, train=True, download=True, transform=transform
            root=self.data_dir, train=True, download=False, transform=transform
        )

        self.slot_id = derive_slot_id()

    def _get_partition_obj(self, partition_path: str) -> dict:
        if partition_path not in self._partition_cache:
            self._partition_cache[partition_path] = load_partitions(partition_path)
        return self._partition_cache[partition_path]

    def _get_trainloader(self, veh_id: str, partition_path: str) -> Tuple[DataLoader, int]:
        key = (partition_path, veh_id)
        if key in self._loader_cache:
            return self._loader_cache[key]

        pobj = self._get_partition_obj(partition_path)
        meta = pobj.get("meta", {})
        num_veh = int(meta.get("num_veh", env_int("NUM_VEH", 100)))
        parts = pobj["partitions"]

        idx = veh_id_to_index(veh_id, num_veh)
        indices = parts[idx] if 0 <= idx < len(parts) else []

        subset = Subset(self.trainset, indices)
        dl = DataLoader(subset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self._loader_cache[key] = (dl, len(indices))
        return dl, len(indices)

    # Flower NumPyClient APIs
    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_params_from_model(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        # required by orchestrator logic
        veh_id = str(config.get("veh_id", "veh0"))
        server_round = int(config.get("server_round", 0))

        # partition_path should come from orchestrator->server->client config
        # fallback to env (so you can still run before wiring orchestrator change)

        # partition_path = str(config.get("partition_path", "")).strip()
        # if not partition_path:
        #     partition_path = env_str("PARTITION_PATH", "/app/outputs/partitions/cifar10_N100_seed42_iid.json")

        partition_path = str(config.get("partition_path", "")).strip()
        if not partition_path:
            partition_path = env_str("PARTITION_PATH", "").strip()
        if not partition_path:
            raise RuntimeError("Missing partition_path. Set PARTITION_PATH env or orchestrator must inject it.")


        # allow orchestrator override CI
        if "ci_g_per_kwh" in config:
            try:
                self.ci_g_per_kwh = float(config["ci_g_per_kwh"])
            except Exception:
                pass

        # allow server/orchestrator override training knobs
        if "epochs" in config:
            self.epochs = int(config["epochs"])
        if "batch_size" in config:
            self.batch_size = int(config["batch_size"])
        if "lr" in config:
            self.lr = float(config["lr"])

        # load received global params
        set_params_to_model(self.model, parameters)
        self.model.train()

        trainloader, num_examples = self._get_trainloader(veh_id, partition_path)

        # edge case: empty partition (e.g., num_veh > n_train)
        if num_examples <= 0:
            metrics = {
                "veh_id": veh_id,
                "t_train_s": 0.0,
                "e_comp_j": 0.0,
                "co2_comp_g": 0.0,
                "server_round": server_round,
                "slot_id": self.slot_id,
                "partition_path": partition_path,
                "num_examples": 0,
            }
            return get_params_from_model(self.model), 0, metrics

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        t0 = time.time()
        for _ in range(self.epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        t1 = time.time()

        t_train_s = float(t1 - t0)
        e_comp_j = self.p_comp_w * t_train_s
        co2_comp_g = joule_to_kwh(e_comp_j) * self.ci_g_per_kwh

        metrics = {
            "veh_id": veh_id,
            "t_train_s": t_train_s,
            "e_comp_j": e_comp_j,
            "co2_comp_g": co2_comp_g,
            "server_round": server_round,
            "slot_id": self.slot_id,
            "partition_path": partition_path,
            "num_examples": int(num_examples),
        }

        return get_params_from_model(self.model), int(num_examples), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        # MVP: disabled
        return 0.0, 0, {}


def main() -> None:
    server_addr = env_str("SERVER_ADDR", "fl_server:8080")
    client = Cifar10PartitionClient()

    while True:
        try:
            fl.client.start_client(server_address=server_addr, client=client.to_client())
            break
        except Exception as e:
            print(f"[fl_client] connect failed ({e}), retry in 2s...")
            time.sleep(2)


if __name__ == "__main__":
    main()
