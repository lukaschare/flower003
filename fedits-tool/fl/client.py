#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import re
from typing import Dict, List, Tuple

import numpy as np
import flwr as fl


def env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default))

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return default

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default

def joule_to_kwh(j: float) -> float:
    return j / 3_600_000.0

def derive_client_id() -> str:
    # 1) explicit env wins
    cid = os.getenv("CLIENT_ID", "").strip()
    if cid:
        return cid

    # 2) docker scale hostname like: fl_client-1 / fl_client_1 / <hash>
    hn = os.getenv("HOSTNAME", "clientX")
    m = re.search(r"(\d+)$", hn)
    if m:
        i = int(m.group(1))
        n = max(i - 1, 0)  # 0-based
        return f"client{n:02d}" if n <= 99 else f"client{n:03d}"
    return "clientX"


class SyntheticClient(fl.client.NumPyClient):
    """
    Minimal compute-plane client:
    - model: vector length D
    - local training: SGD on synthetic regression
    - metrics: t_train_s, e_comp_j, co2_comp_g, veh_id (for control-plane use)
    """

    def __init__(self, cid: str, dim: int = 10) -> None:
        self.cid = cid
        self.dim = dim

        # compute/energy params (replace later with real measurement)
        self.p_comp_w = env_float("P_COMP_W", 18.0)
        self.ci_g_per_kwh = env_float("CI_G_PER_KWH", 200.0)

        # heterogeneity knob
        self.sleep_scale = env_float("SLEEP_SCALE", 0.0)

        self.rnd = random.Random(hash(cid) & 0xFFFFFFFF)

        # synthetic dataset
        self.n = env_int("N_SAMPLES", 200)
        self.X = self._randn(self.n, self.dim)
        w_true = np.array([self.rnd.uniform(-1, 1) for _ in range(self.dim)], dtype=np.float32)
        noise = 0.1 * self._randn(self.n, 1).reshape(-1)
        self.y = (self.X @ w_true) + noise

    def _randn(self, n: int, d: int) -> np.ndarray:
        rng = np.random.RandomState(hash((self.cid, n, d)) & 0xFFFFFFFF)
        return rng.randn(n, d).astype(np.float32)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [np.zeros((self.dim,), dtype=np.float32)]

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        w = parameters[0].astype(np.float32).copy()

        lr = float(config.get("lr", env_float("LR", 0.05)))
        steps = int(config.get("steps", env_int("STEPS", 20)))

        # IMPORTANT: veh_id comes from Orchestrator -> server -> client config
        veh_id = str(config.get("veh_id", ""))
        server_round = int(config.get("server_round", 0))

        # Orchestrator may pass CI; if present override
        if "ci_g_per_kwh" in config:
            try:
                self.ci_g_per_kwh = float(config["ci_g_per_kwh"])
            except Exception:
                pass

        t0 = time.time()

        for _ in range(steps):
            pred = self.X @ w
            grad = (self.X.T @ (pred - self.y)) / float(self.n)
            w -= lr * grad.astype(np.float32)

        if self.sleep_scale > 0:
            time.sleep(self.sleep_scale * self.rnd.uniform(0.0, 1.0))

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
        }

        return [w], self.n, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        return 0.0, self.n, {}


def main() -> None:
    server_addr = env_str("SERVER_ADDR", "fl_server:8080")
    cid = derive_client_id()

    client = SyntheticClient(cid=cid, dim=10)

    while True:
        try:
            # fl.client.start_numpy_client(server_address=server_addr, client=client)
            # 使用 .to_client() 转换，并调用 start_client
            fl.client.start_client(server_address=server_addr, client=client.to_client())
            break
        except Exception as e:
            print(f"[{cid}] connect failed ({e}), retry in 2s...")
            time.sleep(2)


if __name__ == "__main__":
    main()
