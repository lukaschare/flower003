#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
from typing import Dict, List, Tuple

import numpy as np
import flwr as fl
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays


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


class SyntheticClient(fl.client.NumPyClient):
    """
    Minimal client:
    - model is a vector of length D
    - local training: do a few gradient steps on synthetic linear regression
    - reports t_train_s, e_comp_j, co2_comp_g in metrics
    """

    def __init__(self, cid: str, dim: int = 10) -> None:
        self.cid = cid
        self.dim = dim

        self.p_comp_w = env_float("P_COMP_W", 18.0)          # mock compute power
        self.ci_g_per_kwh = env_float("CI_G_PER_KWH", 200.0) # can be overwritten by server later

        # heterogeneity knob: extra sleep to emulate slow devices
        self.sleep_scale = env_float("SLEEP_SCALE", 0.0)     # e.g., 0.5 makes slower

        # fixed local dataset seed by cid
        self.rnd = random.Random(hash(cid) & 0xFFFFFFFF)

        # synthetic dataset
        self.n = env_int("N_SAMPLES", 200)
        self.X = self._randn(self.n, self.dim)
        # per-client true weight
        w_true = np.array([self.rnd.uniform(-1, 1) for _ in range(self.dim)], dtype=np.float32)
        noise = 0.1 * self._randn(self.n, 1).reshape(-1)
        self.y = (self.X @ w_true) + noise

    def _randn(self, n: int, d: int) -> np.ndarray:
        # deterministic numpy RNG per client
        rng = np.random.RandomState(hash((self.cid, n, d)) & 0xFFFFFFFF)
        return rng.randn(n, d).astype(np.float32)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        # initial params = zeros
        return [np.zeros((self.dim,), dtype=np.float32)]

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        w = parameters[0].astype(np.float32).copy()

        # local training hyperparams
        lr = float(config.get("lr", env_float("LR", 0.05)))
        steps = int(config.get("steps", env_int("STEPS", 20)))
        veh_id = str(config.get("veh_id", ""))
        server_round = int(config.get("server_round", 0))

        t0 = time.time()

        # simple SGD on MSE: grad = X^T (Xw - y)/n
        for _ in range(steps):
            pred = self.X @ w
            grad = (self.X.T @ (pred - self.y)) / float(self.n)
            w -= lr * grad.astype(np.float32)

        # optional sleep to emulate slower compute
        if self.sleep_scale > 0:
            time.sleep(self.sleep_scale * self.rnd.uniform(0.0, 1.0))

        t1 = time.time()
        t_train_s = float(t1 - t0)

        # compute energy/carbon (mock; later you replace with real measurement)
        e_comp_j = self.p_comp_w * t_train_s
        co2_comp_g = joule_to_kwh(e_comp_j) * self.ci_g_per_kwh

        metrics = {
            "cid": self.cid,
            "veh_id": veh_id,
            "t_train_s": t_train_s,
            "e_comp_j": e_comp_j,
            "co2_comp_g": co2_comp_g,
            "server_round": server_round,
        }

        return [w], self.n, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        # MVP: no evaluation
        return 0.0, self.n, {}


def main() -> None:
    server_addr = env_str("SERVER_ADDR", "fl_server:8080")
    cid = env_str("CLIENT_ID", "clientX")

    client = SyntheticClient(cid=cid, dim=10)

    # Robust connect loop (server may not be ready)
    while True:
        try:
            fl.client.start_numpy_client(server_address=server_addr, client=client)
            break
        except Exception as e:
            print(f"[{cid}] connect failed ({e}), retry in 2s...")
            time.sleep(2)


if __name__ == "__main__":
    main()
