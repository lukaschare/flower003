#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import json
import re
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import flwr as fl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

from model_cnn import build_model  # 如果路径不对：按你的工程结构调整 import


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
    return os.getenv("HOSTNAME", "unknown")


def stable_mod(s: str, mod: int) -> int:
    h = hashlib.sha1((s or "").encode("utf-8")).digest()
    v = int.from_bytes(h[:8], "big")
    return v % mod


def veh_id_to_index(veh_id: str, num_veh: int) -> int:
    if num_veh <= 0:
        return 0
    m = re.search(r"(\d+)$", veh_id or "")
    if m:
        return int(m.group(1)) % num_veh
    return stable_mod(veh_id, num_veh)


def load_partitions(partition_path: str) -> dict:
    with open(partition_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_params_from_model(model: nn.Module) -> List[np.ndarray]:
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


def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion_sum(logits, y)
            loss_sum += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))
    if total <= 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total


# -------------------------
# Model
# -------------------------
class SimpleCifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -------------------------
# Flower Client
# -------------------------
class Cifar10PartitionClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        self.kappa = env_float("SIM_KAPPA", 1e-28)
        self.cycles_per_sample = env_float("SIM_CYCLES_PER_SAMPLE", 1e7)
        self.cpu_freq_hz = env_float("SIM_CPU_FREQ", 1e9) 
        self.p_comp_w = env_float("P_COMP_W", 18.0)
        self.ci_g_per_kwh = env_float("CI_G_PER_KWH", 153.0)

        self.epochs = env_int("EPOCHS", 1)
        self.batch_size = env_int("BATCH_SIZE", 32) # CIFAR-10 上过大的 batch size 可能导致显存不足，默认设置为 32
        self.lr = env_float("LR", 0.005) # CIFAR-10 上过大的学习率 可能导致训练不稳定，默认设置为 0.005

        self.data_dir = env_str("DATA_DIR", "/app/data")
        self._partition_cache: Dict[str, dict] = {}
        self._loader_cache: Dict[Tuple[str, str], Tuple[DataLoader, DataLoader, DataLoader, int, int]] = {}

        self.device = torch.device(env_str("DEVICE", "cpu"))
        # self.model: nn.Module = SimpleCifarCNN().to(self.device)

        # allow_download = (env_int("ALLOW_DOWNLOAD", 0) == 1)
        
        # # 1. 训练集 Transform：带数据增强 + Normalize
        # transform_train = T.Compose([
        #     T.RandomCrop(32, padding=4),
        #     T.RandomHorizontalFlip(),
        #     T.ToTensor(),
        #     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # ])
        
        # # 2. 验证/评估集 Transform：仅 Normalize
        # transform_val = T.Compose([
        #     T.ToTensor(),
        #     T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # ])

        # self.dataset_train = torchvision.datasets.CIFAR10(
        #     root=self.data_dir, train=True, download=allow_download, transform=transform_train
        # )
        # self.dataset_val = torchvision.datasets.CIFAR10(
        #     root=self.data_dir, train=True, download=False, transform=transform_val
        # )

        
        # --- 修改开始：动态加载模型和数据集 ---
        self.dataset_name = env_str("DATASET", "cifar10").lower()
        self.model: nn.Module = build_model(self.dataset_name).to(self.device)

        allow_download = (env_int("ALLOW_DOWNLOAD", 0) == 1)

        if self.dataset_name in ("fashionmnist", "fashion-mnist"):
            transform_train = T.Compose([
                T.RandomCrop(28, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            transform_val = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5,), (0.5,))
            ])
            self.dataset_train = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, download=allow_download, transform=transform_train
            )
            self.dataset_val = torchvision.datasets.FashionMNIST(
                root=self.data_dir, train=True, download=False, transform=transform_val
            )
        else:
            transform_train = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_val = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            self.dataset_train = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=allow_download, transform=transform_train
            )
            self.dataset_val = torchvision.datasets.CIFAR10(
                root=self.data_dir, train=True, download=False, transform=transform_val
            )
        # --- 修改结束 ---


        self.slot_id = derive_slot_id()

    def _get_partition_obj(self, partition_path: str) -> dict:
        if partition_path not in self._partition_cache:
            self._partition_cache[partition_path] = load_partitions(partition_path)
        return self._partition_cache[partition_path]

    def _get_loaders(self, veh_id: str, partition_path: str) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
        key = (partition_path, veh_id)
        if key in self._loader_cache:
            return self._loader_cache[key]

        pobj = self._get_partition_obj(partition_path)
        meta = pobj.get("meta", {})
        num_veh = int(meta.get("num_veh", env_int("NUM_VEH", 100)))
        parts = pobj["partitions"]

        idx = veh_id_to_index(veh_id, num_veh)
        indices = parts[idx] if 0 <= idx < len(parts) else []

        n_total = len(indices)
        if n_total <= 0:
            return None, None, None, 0, 0
            
        import random
        rng = random.Random(42 + idx)
        shuffled_indices = list(indices)
        rng.shuffle(shuffled_indices)
        
        split_idx = int(0.8 * n_total)
        train_idx = shuffled_indices[:split_idx]
        val_idx = shuffled_indices[split_idx:]

        # 训练过程 Loader (带增强)
        trainloader = DataLoader(Subset(self.dataset_train, train_idx), batch_size=self.batch_size, shuffle=True, num_workers=0)
        # 训练集评估 Loader (无增强，保证评估稳定)
        train_eval_loader = DataLoader(Subset(self.dataset_val, train_idx), batch_size=self.batch_size, shuffle=False, num_workers=0)
        # 验证集评估 Loader (无增强)
        valloader = DataLoader(Subset(self.dataset_val, val_idx), batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        self._loader_cache[key] = (trainloader, train_eval_loader, valloader, len(train_idx), len(val_idx))
        return trainloader, train_eval_loader, valloader, len(train_idx), len(val_idx)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_params_from_model(self.model)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        veh_id = str(config.get("veh_id", "veh0"))
        server_round = int(config.get("server_round", 0))

        partition_path = str(config.get("partition_path", "")).strip()
        if not partition_path:
            partition_path = env_str("PARTITION_PATH", "").strip()
        if not partition_path:
            raise RuntimeError("Missing partition_path. Set PARTITION_PATH env or orchestrator must inject it.")

        if "sim_cpu_freq" in config:
            self.cpu_freq_hz = float(config["sim_cpu_freq"])
        if "ci_g_per_kwh" in config:
            try:
                self.ci_g_per_kwh = float(config["ci_g_per_kwh"])
            except Exception:
                pass
        if "epochs" in config:
            self.epochs = int(config["epochs"])
        if "batch_size" in config:
            self.batch_size = int(config["batch_size"])
        if "lr" in config:
            self.lr = float(config["lr"])

        set_params_to_model(self.model, parameters)
        self.model.train()

        trainloader, train_eval_loader, valloader, num_train, num_val = self._get_loaders(veh_id, partition_path)
        num_examples_total = num_train + num_val

        # edge case: empty partition
        if num_examples_total <= 0:
            metrics = {
                "veh_id": veh_id, "t_train_s": 0.0, "e_comp_j": 0.0, "co2_comp_g": 0.0,
                "train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None,
                "server_round": server_round, "slot_id": self.slot_id,
                "partition_path": partition_path, "num_examples": 0, "num_train": 0, "num_val": 0,
            }
            return get_params_from_model(self.model), 0, metrics

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)

        t0 = time.time()
        for _ in range(self.epochs):
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

        # 虚拟时间和能耗只根据 num_train 计算
        total_cpu_cycles = self.epochs * num_train * self.cycles_per_sample
        t_train_virtual = total_cpu_cycles / self.cpu_freq_hz
        p_comp_virtual = self.kappa * (self.cpu_freq_hz ** 3)
        e_comp_virtual = p_comp_virtual * t_train_virtual
        co2_comp_virtual = joule_to_kwh(e_comp_virtual) * self.ci_g_per_kwh

        # 使用无增强的 loader 评估
        train_loss, train_acc = eval_on_loader(self.model, train_eval_loader, self.device) if num_train > 0 else (None, None)
        val_loss, val_acc = eval_on_loader(self.model, valloader, self.device) if num_val > 0 else (None, None)

        metrics = {
            "veh_id": veh_id,
            "t_train_s": float(t_train_virtual),
            "e_comp_j": float(e_comp_virtual),
            "co2_comp_g": float(co2_comp_virtual),
            "sim_cpu_freq": float(self.cpu_freq_hz),
            "train_loss": float(train_loss) if train_loss is not None else None,
            "train_acc": float(train_acc) if train_acc is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_acc": float(val_acc) if val_acc is not None else None,
            "server_round": server_round,
            "slot_id": self.slot_id,
            "partition_path": partition_path,
            "num_examples": int(num_examples_total),
            "num_train": int(num_train),
            "num_val": int(num_val),
        }

        # 注意：返回给 Flower Server 的 weight 设置为 num_train，聚合才准确
        return get_params_from_model(self.model), int(num_train), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
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