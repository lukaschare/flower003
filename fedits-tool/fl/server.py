#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import urllib.request
from typing import Dict, List, Tuple, Optional, Any
import time

import numpy as np
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager


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

def params_l2_norm(nds: List[np.ndarray]) -> float:
    s = 0.0
    for a in nds:
        aa = a.astype(np.float64, copy=False)
        s += float(np.sum(aa * aa))
    return float(s ** 0.5)

def aggregate_weighted(params_and_weights: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    """
    Multi-layer weighted average (FedAvg):
      agg[layer] = sum_i w_i * p_i[layer] / sum_i w_i
    """
    total = float(sum(w for _, w in params_and_weights))
    if total <= 0:
        # return first as fallback (should not happen if weights>0)
        return params_and_weights[0][0]

    n_layers = len(params_and_weights[0][0])
    agg = [np.zeros_like(params_and_weights[0][0][k]) for k in range(n_layers)]

    for params, w in params_and_weights:
        wf = float(w) / total
        for k in range(n_layers):
            agg[k] += params[k] * wf

    return agg


# -------------------------
# Minimal HTTP client
# -------------------------
class OrchestratorHttpClient:
    """Minimal HTTP client for Orchestrator control-plane service."""
    def __init__(self, base_url: str, timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            text = resp.read().decode("utf-8").strip()
            return json.loads(text) if text else {}

    def get(self, path: str) -> Dict[str, Any]:
        url = self.base_url + path
        with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
            text = resp.read().decode("utf-8").strip()
            return json.loads(text) if text else {}


# -------------------------
# Model init (must match client CNN exactly)
# -------------------------
def build_cifar10_cnn_params() -> List[np.ndarray]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class SimpleCifarCNN(nn.Module):
        def __init__(self, num_classes: int = 10) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    m = SimpleCifarCNN()
    # stable order: state_dict items
    return [v.detach().cpu().numpy() for _, v in m.state_dict().items()]


# -------------------------
# Proxy Strategy (strict separation)
# -------------------------
class ProxyOrchestratedFedAvg(fl.server.strategy.Strategy):
    """
    STRICT separation:
    - Orchestrator (control plane): selection, Veins down/up simulation, commit/drop, logs, t_sim advance
    - Flower server (data plane): only dispatches fit and aggregates committed updates
    - Flower clients (compute plane): real training + comp energy/carbon metrics

    For each round:
    1) configure_fit -> ask Orchestrator who trains + fit_config per selected physical client_id
    2) aggregate_fit -> send client train metrics to Orchestrator, get committed client_ids
    3) aggregate only committed updates (multi-layer FedAvg)
    4) report global_model_norm back to Orchestrator finalize (server_round.csv, advance t_sim)
    """

    def __init__(
        self,
        orch_url: str,
        min_fit_clients: int = 1,
        min_available_clients: int = 10,
        wait_timeout: int = 60,
    ) -> None:
        super().__init__()
        self.orch = OrchestratorHttpClient(orch_url, timeout_s=90.0)

        # CNN init params
        self.init_params_nd = build_cifar10_cnn_params()

        self.min_fit_clients = int(min_fit_clients)
        self.min_available_clients = int(min_available_clients)
        self.wait_timeout = int(wait_timeout)

        try:
            h = self.orch.get("/health")
            print(f"[server] Orchestrator health: {h}")
        except Exception as e:
            print(f"[server] Orchestrator not reachable yet: {e}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters(self.init_params_nd)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        # Wait for physical clients
        wait_num = self.min_available_clients if server_round == 1 else self.min_fit_clients
        print(f"[round {server_round}] Waiting for {wait_num} clients (timeout={self.wait_timeout}s)...")

        success = client_manager.wait_for(num_clients=wait_num, timeout=self.wait_timeout)
        if not success:
            print(f"[round {server_round}] Timeout! Only {len(client_manager.all())} clients connected. Skipping round.")
            return []

        all_clients = list(client_manager.all().values())
        available_ids = [cp.cid for cp in all_clients]

        print(f"[Round {server_round}] Step 1: Physical connected clients = {len(available_ids)}")
        print(f"2. Local Physical IDs (Scope sent to Orch): {available_ids}")

        # Ask Orchestrator (logical selection)
        resp = self.orch.post(
            "/v1/round/configure_fit",
            {
                "server_round": int(server_round),
                "available_client_ids": available_ids,
            },
        )
        if not resp.get("ok", False):
            print(f"[server] configure_fit failed: {resp}")
            return []

        assignments = resp.get("train_assignments", [])

        # If Orchestrator selected nobody, force time advance (common when SUMO time is still early)
        if not assignments:
            print(f"[Round {server_round}] No clients selected (Orch empty). Forcing time advance...")
            self.orch.post(
                "/v1/round/finalize",
                {"server_round": int(server_round), "global_model_norm": 0.0},
            )
            return []

        print(f"[Round {server_round}] Step 2: Orchestrator selected = {len(assignments)} "
              f"(Filtered out {len(available_ids) - len(assignments)})")

        # Dispatch fit tasks
        by_cid = {cp.cid: cp for cp in all_clients}
        fit_instructions: List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]] = []

        for a in assignments:
            cid = a.get("client_id", "")
            cfg = a.get("fit_config", {})  # should include veh_id, ci, partition_path, etc.
            cp = by_cid.get(cid)
            if cp is None:
                continue
            fit_instructions.append((cp, FitIns(parameters, cfg)))

        print(f"[round {server_round}] dispatch_fit={len(fit_instructions)} (from assignments={len(assignments)})")
        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        # Send fit metrics to Orchestrator for commit/drop decision
        fit_results_payload: List[dict] = []
        for cp, fitres in results:
            m = dict(fitres.metrics) if fitres.metrics else {}
            fit_results_payload.append(
                {
                    "client_id": cp.cid,
                    "veh_id": str(m.get("veh_id", "")),
                    "t_train_s": float(m.get("t_train_s", 0.0)),
                    "e_comp_j": float(m.get("e_comp_j", 0.0)),
                    "co2_comp_g": float(m.get("co2_comp_g", 0.0)),
                    "num_examples": int(fitres.num_examples) if fitres.num_examples is not None else int(m.get("num_examples", 0)),
                }
            )

        dec = self.orch.post(
            "/v1/round/decide_commit",
            {
                "server_round": int(server_round),
                "fit_results": fit_results_payload,
            },
        )

        if not dec.get("ok", False):
            print(f"[server] decide_commit failed: {dec}")
            committed = set()
        else:
            committed = set(dec.get("committed_client_ids", []))

        # Aggregate only committed updates (multi-layer FedAvg)
        params_and_weights: List[Tuple[List[np.ndarray], int]] = []
        for cp, fitres in results:
            if cp.cid not in committed:
                continue
            nds = parameters_to_ndarrays(fitres.parameters)
            w = int(fitres.num_examples) if fitres.num_examples is not None else 0
            if w <= 0:
                # empty partition -> ignore
                continue
            params_and_weights.append((nds, w))

        if not params_and_weights:
            new_params = None
            global_norm = float("nan")
        else:
            agg_nds = aggregate_weighted(params_and_weights)
            new_params = ndarrays_to_parameters(agg_nds)
            global_norm = params_l2_norm(agg_nds)

        # Finalize round in Orchestrator (server_round.csv + advance t_sim happens there)
        fin = self.orch.post(
            "/v1/round/finalize",
            {"server_round": int(server_round), "global_model_norm": float(global_norm) if global_norm == global_norm else 0.0},
        )
        if not fin.get("ok", False):
            print(f"[server] finalize failed: {fin}")

        return new_params, {"m_committed": len(committed), "global_model_norm": global_norm}

    # MVP: disable evaluate
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Any]]]:
        return None


def main() -> None:
    SERVER_ADDR = env_str("SERVER_ADDR", "0.0.0.0:8080")
    ROUNDS = env_int("ROUNDS", 10)
    ORCH_URL = env_str("ORCH_URL", "http://orchestrator:7070")

    wait_timeout = env_int("WAIT_TIMEOUT", 60)
    min_fit_clients = env_int("MIN_FIT_CLIENTS", 1)
    min_available_clients = env_int("MIN_AVAILABLE_CLIENTS", 10)

    strat = ProxyOrchestratedFedAvg(
        orch_url=ORCH_URL,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        wait_timeout=wait_timeout,
    )

    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strat,
    )


if __name__ == "__main__":
    main()
