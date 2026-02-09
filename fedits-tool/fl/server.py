#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import urllib.request
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
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

def vec_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))

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
# Proxy Strategy (strict separation)
# -------------------------

class ProxyOrchestratedFedAvg(fl.server.strategy.Strategy):
    """
    STRICT separation:
      - Orchestrator (control plane): selection, Veins down/up simulation, commit/drop, CSV logs, t_sim advance
      - Flower server (data plane): only dispatches fit and aggregates committed updates
      - Flower clients (compute plane): real training + comp energy/carbon metrics

    For each round:
      1) configure_fit -> ask Orchestrator who trains (dl_ok only) + fit_config per client
      2) aggregate_fit -> send client train metrics to Orchestrator, get committed client_ids
      3) aggregate only committed updates
      4) report global_model_norm back to Orchestrator finalize (server_round.csv, advance t_sim)
    """

    def __init__(self, orch_url: str) -> None:
        super().__init__()
        self.orch = OrchestratorHttpClient(orch_url, timeout_s=90.0)
        self.init_nd = np.zeros((10,), dtype=np.float32)

        # optional quick sanity
        try:
            h = self.orch.get("/health")
            print(f"[server] Orchestrator health: {h}")
        except Exception as e:
            print(f"[server] Orchestrator not reachable yet: {e}")

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters([self.init_nd])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:

        # All currently connected clients
        all_clients = list(client_manager.all().values())
        available_ids = [cp.cid for cp in all_clients]

        resp = self.orch.post("/v1/round/configure_fit", {
            "server_round": int(server_round),
            "available_client_ids": available_ids,
        })
        if not resp.get("ok", False):
            print(f"[server] configure_fit failed: {resp}")
            return []

        # Orchestrator returns only dl_ok assignments for training
        assignments = resp.get("train_assignments", [])
        by_cid = {cp.cid: cp for cp in all_clients}

        fit_instructions: List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]] = []
        for a in assignments:
            cid = a.get("client_id", "")
            cfg = a.get("fit_config", {})
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

        # build payload to Orchestrator (control plane uses it to compute uplink start & commit/drop)
        fit_results_payload: List[dict] = []
        for cp, fitres in results:
            m = dict(fitres.metrics) if fitres.metrics else {}
            fit_results_payload.append({
                "client_id": cp.cid,
                "veh_id": str(m.get("veh_id", "")),
                "t_train_s": float(m.get("t_train_s", 0.0)),
                "e_comp_j": float(m.get("e_comp_j", 0.0)),
                "co2_comp_g": float(m.get("co2_comp_g", 0.0)),
                "num_examples": int(fitres.num_examples) if fitres.num_examples is not None else 1,
            })

        dec = self.orch.post("/v1/round/decide_commit", {
            "server_round": int(server_round),
            "fit_results": fit_results_payload,
        })
        if not dec.get("ok", False):
            print(f"[server] decide_commit failed: {dec}")
            committed = set()
        else:
            committed = set(dec.get("committed_client_ids", []))

        # Aggregate only committed
        if not committed:
            new_params = None
            global_norm = float("nan")
        else:
            weights: List[int] = []
            vecs: List[np.ndarray] = []
            for cp, fitres in results:
                if cp.cid not in committed:
                    continue
                nds = parameters_to_ndarrays(fitres.parameters)
                v = nds[0].astype(np.float32)
                n = int(fitres.num_examples) if fitres.num_examples is not None else 1
                weights.append(n)
                vecs.append(v * n)

            denom = float(sum(weights)) if weights else 1.0
            agg = np.sum(vecs, axis=0) / denom
            new_params = ndarrays_to_parameters([agg])
            global_norm = vec_norm(agg)

        # finalize round in Orchestrator (server_round.csv + advance t_sim happens there)
        fin = self.orch.post("/v1/round/finalize", {
            "server_round": int(server_round),
            "global_model_norm": float(global_norm),
        })
        if not fin.get("ok", False):
            print(f"[server] finalize failed: {fin}")

        return new_params, {"m_committed": len(committed), "global_model_norm": global_norm}

    # MVP: disable evaluate
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        return None, {}


def main() -> None:
    SERVER_ADDR = env_str("SERVER_ADDR", "0.0.0.0:8080")
    ROUNDS = env_int("ROUNDS", 10)
    ORCH_URL = env_str("ORCH_URL", "http://orchestrator:7070")

    strat = ProxyOrchestratedFedAvg(orch_url=ORCH_URL)

    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strat,
    )


if __name__ == "__main__":
    main()
