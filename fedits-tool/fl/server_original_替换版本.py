#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

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

# Reuse your Veins client abstraction
from orchestrator.veins_client import (
    MockVeinsClient,
    RPCVeinsClient,
    DLResult,
    ULResult,
    VeinsState,
)

# -------------------------
# Helpers
# -------------------------

def run_id_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def joule_to_kwh(j: float) -> float:
    return j / 3_600_000.0

def comm_energy_j(p_rx_w: float, p_tx_w: float, t_down_s: float, t_up_s: float) -> float:
    return p_rx_w * max(t_down_s, 0.0) + p_tx_w * max(t_up_s, 0.0)

def comm_carbon_g(e_comm_j: float, ci_g_per_kwh: float) -> float:
    return joule_to_kwh(e_comm_j) * ci_g_per_kwh

def vec_norm(x: np.ndarray) -> float:
    return float(np.sqrt(np.sum(x * x)))

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

def env_str(name: str, default: str) -> str:
    return str(os.getenv(name, default))

# -------------------------
# CSV Loggers (append mode)
# -------------------------

class CsvAppender:
    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        ensure_dir(os.path.dirname(path))
        self._init_file()

    def _init_file(self) -> None:
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=self.fieldnames)
                w.writeheader()

    def append_rows(self, rows: List[dict]) -> None:
        if not rows:
            return
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            for r in rows:
                w.writerow(r)

# -------------------------
# Strategy with Orchestrator logic
# -------------------------

class OrchestratedFedAvg(fl.server.strategy.Strategy):
    """
    Flower server strategy embedding your orchestrator semantics:

    For each round:
      select (based on Veins in-range at t_sim)
      simulate downlink in Veins -> dl_done times
      train in real clients (Docker) -> returns t_train_s, e_comp_j, co2_comp_g
      simulate uplink in Veins starting at dl_done + t_train_s
      commit/drop based on uplink success + deadline
      aggregate only committed updates
      log clients_round.csv + server_round.csv
      advance virtual time: t_sim = deadline
    """

    def __init__(
        self,
        *,
        veins_mode: str,
        veins_host: str,
        veins_port: int,
        map_size_m: float,
        num_vehicles: int,
        rsu_x_m: float,
        rsu_y_m: float,
        rsu_radius_m: float,
        seed: int,
        rounds: int,
        m: int,
        deadline_s: float,
        model_down_bytes: int,
        model_up_bytes: int,
        ci_g_per_kwh: float,
        p_rx_w: float,
        p_tx_w: float,
        out_dir: str,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.rounds = rounds
        self.m = m
        self.deadline_s = deadline_s
        self.model_down_bytes = model_down_bytes
        self.model_up_bytes = model_up_bytes
        self.ci_g_per_kwh = ci_g_per_kwh
        self.p_rx_w = p_rx_w
        self.p_tx_w = p_tx_w

        # virtual simulation time
        self.t_sim = 0.0

        # Veins client
        if veins_mode == "rpc":
            self.veins = RPCVeinsClient(host=veins_host, port=veins_port, timeout_s=15.0)
        else:
            self.veins = MockVeinsClient(
                map_size_m=map_size_m,
                num_vehicles=num_vehicles,
                rsu_x_m=rsu_x_m,
                rsu_y_m=rsu_y_m,
                rsu_radius_m=rsu_radius_m,
                seed=seed,
            )

        # output
        self.run_id = run_id_now()
        self.run_dir = os.path.join(out_dir, "runs", self.run_id)
        ensure_dir(self.run_dir)

        # snapshot config
        with open(os.path.join(self.run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": self.run_id,
                    "veins_mode": veins_mode,
                    "rsu": {"x": rsu_x_m, "y": rsu_y_m, "r": rsu_radius_m},
                    "map_size_m": map_size_m,
                    "num_vehicles": num_vehicles,
                    "seed": seed,
                    "rounds": rounds,
                    "m": m,
                    "deadline_s": deadline_s,
                    "model_down_bytes": model_down_bytes,
                    "model_up_bytes": model_up_bytes,
                    "ci_g_per_kwh": ci_g_per_kwh,
                    "p_rx_w": p_rx_w,
                    "p_tx_w": p_tx_w,
                },
                f,
                indent=2,
            )

        # CSV schemas
        self.clients_fields = [
            "run_id","round","client_id","veh_id",
            "selected","committed","drop_reason",
            "t_round_start","t_deadline",
            "t_dl_done","t_ul_start","t_ul_done",
            "t_down","t_train","t_up",
            "dl_ok","ul_ok",
            "dl_goodput_mbps","ul_goodput_mbps",
            "dl_rtt_ms","ul_rtt_ms",
            "e_comp_j","co2_comp_g",
            "e_comm_j","co2_comm_g","co2_total_g",
        ]
        self.server_fields = [
            "run_id","round",
            "m_selected","m_committed","dropout_rate",
            "global_model_norm",
            "co2_committed_g","co2_dropped_g","co2_total_g",
            "t_round_start","t_deadline",
        ]
        self.clients_csv = CsvAppender(os.path.join(self.run_dir, "clients_round.csv"), self.clients_fields)
        self.server_csv = CsvAppender(os.path.join(self.run_dir, "server_round.csv"), self.server_fields)

        # round context store
        self.ctx: Dict[int, dict] = {}

        # init model (10 dims)
        self.init_nd = np.zeros((10,), dtype=np.float32)

        print(f"[server] run_id={self.run_id} out={self.run_dir} veins_mode={veins_mode}")

    # ----- Strategy required methods -----

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters([self.init_nd])

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:

        t_round_start = self.t_sim
        t_deadline = t_round_start + self.deadline_s

        # (1) Query Veins state
        state: VeinsState = self.veins.get_state(t=t_round_start)
        candidates_veh = [vid for vid, v in state.vehicles.items() if v.in_range]

        # (2) Choose veh_ids (random)
        rnd = np.random.RandomState(self.seed + server_round)
        if len(candidates_veh) <= self.m:
            selected_veh = candidates_veh
        else:
            selected_veh = list(rnd.choice(candidates_veh, size=self.m, replace=False))

        # (3) Map veh -> Flower clients (we just pick available clients 1-to-1)
        # We sample exactly len(selected_veh) Flower clients.
        available = list(client_manager.all().values())
        if len(available) == 0:
            return []
        if len(available) < len(selected_veh):
            selected_veh = selected_veh[:len(available)]

        rnd2 = np.random.RandomState(self.seed + 999 + server_round)
        chosen_clients = list(rnd2.choice(available, size=len(selected_veh), replace=False))

        veh_by_client: Dict[str, str] = {}
        for cp, veh in zip(chosen_clients, selected_veh):
            veh_by_client[cp.cid] = veh

        # (4) Downlink simulation in Veins (veh domain)
        dl = self.veins.simulate_downlink(
            t_now=t_round_start,
            veh_ids=selected_veh,
            size_bytes=self.model_down_bytes,
            deadline=t_deadline,
        )

        # Determine which veh got model successfully
        dl_ok_veh = [veh for veh in selected_veh if dl.get(veh) and dl[veh].ok]

        # Keep only clients whose mapped veh is dl_ok
        fit_instructions = []
        for cp in chosen_clients:
            veh = veh_by_client[cp.cid]
            dlr = dl.get(veh, DLResult.fail())
            if not dlr.ok:
                continue

            config = {
                "server_round": server_round,
                "veh_id": veh,
                "t_round_start": t_round_start,
                "t_deadline": t_deadline,
                # optional: you can pass model sizes for client-side logging
                "model_down_bytes": self.model_down_bytes,
                "model_up_bytes": self.model_up_bytes,
            }
            fit_instructions.append((cp, FitIns(parameters, config)))

        # Store ctx for aggregate_fit
        self.ctx[server_round] = {
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
            "veh_by_client": veh_by_client,
            "dl": dl,
            "selected_veh": selected_veh,
            "selected_client_cids": [cp.cid for cp in chosen_clients],
        }

        print(
            f"[round {server_round}] t={t_round_start:.2f} candVeh={len(candidates_veh)} "
            f"selVeh={len(selected_veh)} dlOK={len(fit_instructions)}"
        )

        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:

        ctx = self.ctx.get(server_round, {})
        t_round_start = float(ctx.get("t_round_start", self.t_sim))
        t_deadline = float(ctx.get("t_deadline", t_round_start + self.deadline_s))
        veh_by_client = ctx.get("veh_by_client", {})
        dl: Dict[str, DLResult] = ctx.get("dl", {})

        # Parse client results
        # Each client returns: parameters + metrics(t_train_s, e_comp_j, co2_comp_g)
        train_metrics: Dict[str, dict] = {}
        params_by_client: Dict[str, List[np.ndarray]] = {}

        for cp, fitres in results:
            cid = cp.cid
            veh = veh_by_client.get(cid, "")
            m = dict(fitres.metrics) if fitres.metrics else {}
            t_train_s = float(m.get("t_train_s", 0.0))
            e_comp_j = float(m.get("e_comp_j", 0.0))
            co2_comp_g = float(m.get("co2_comp_g", 0.0))
            train_metrics[cid] = {
                "veh": veh,
                "t_train_s": t_train_s,
                "e_comp_j": e_comp_j,
                "co2_comp_g": co2_comp_g,
            }
            params_by_client[cid] = parameters_to_ndarrays(fitres.parameters)

        # Schedule uplink start times in veh domain
        ul_start_times_veh: Dict[str, float] = {}
        for cid, tm in train_metrics.items():
            veh = tm["veh"]
            dlr = dl.get(veh, DLResult.fail())
            if not dlr.ok:
                continue
            ul_start_times_veh[veh] = float(dlr.t_done) + float(tm["t_train_s"])

        # Simulate uplink in Veins
        ul: Dict[str, ULResult] = self.veins.simulate_uplink(
            start_times=ul_start_times_veh,
            veh_ids=list(ul_start_times_veh.keys()),
            size_bytes=self.model_up_bytes,
            deadline=t_deadline,
        )

        # Commit/drop
        committed_clients: List[str] = []
        dropped_clients: Dict[str, str] = {}
        co2_committed = 0.0
        co2_dropped = 0.0

        client_rows: List[dict] = []

        # For accounting, we also want to log selected but missing result clients
        selected_client_cids = ctx.get("selected_client_cids", [])
        all_considered = set(selected_client_cids) | set(train_metrics.keys())

        for cid in sorted(all_considered):
            veh = veh_by_client.get(cid, "")
            selected = 1

            dlr = dl.get(veh, DLResult.fail("dl_failed"))
            dl_ok = 1 if dlr.ok else 0

            tm = train_metrics.get(cid, None)
            if tm is None:
                # did not return fit result
                t_train_s = 0.0
                e_comp_j = 0.0
                co2_comp_g = 0.0
            else:
                t_train_s = float(tm["t_train_s"])
                e_comp_j = float(tm["e_comp_j"])
                co2_comp_g = float(tm["co2_comp_g"])

            t_dl_done = float(dlr.t_done) if dlr.ok else -1.0
            t_ul_start = (t_dl_done + t_train_s) if (dlr.ok and tm is not None) else -1.0

            ulr = ul.get(veh, ULResult.fail("ul_not_scheduled"))
            ul_ok = 1 if ulr.ok else 0
            t_ul_done = float(ulr.t_done) if ulr.ok else -1.0

            t_down = (t_dl_done - t_round_start) if dlr.ok else 0.0
            t_up = (t_ul_done - t_ul_start) if (ulr.ok and t_ul_start >= 0) else 0.0

            # Commit rule: uplink ok and <= deadline and has fit params
            if dlr.ok and (tm is not None) and ulr.ok and (t_ul_done <= t_deadline) and (cid in params_by_client):
                committed = 1
                drop_reason = ""
                committed_clients.append(cid)
            else:
                committed = 0
                if not dlr.ok:
                    drop_reason = dlr.reason or "dl_failed"
                elif tm is None:
                    drop_reason = "fit_missing"
                elif not ulr.ok:
                    drop_reason = ulr.reason or "ul_failed"
                else:
                    drop_reason = "deadline_miss"
                dropped_clients[cid] = drop_reason

            # comm energy/carbon
            e_comm_j = comm_energy_j(self.p_rx_w, self.p_tx_w, t_down, t_up)
            co2_comm_g = comm_carbon_g(e_comm_j, self.ci_g_per_kwh)
            co2_total_g = co2_comp_g + co2_comm_g

            if committed == 1:
                co2_committed += co2_total_g
            else:
                co2_dropped += co2_total_g

            client_rows.append({
                "run_id": self.run_id,
                "round": server_round,
                "client_id": cid,
                "veh_id": veh,
                "selected": selected,
                "committed": committed,
                "drop_reason": drop_reason,
                "t_round_start": t_round_start,
                "t_deadline": t_deadline,
                "t_dl_done": t_dl_done,
                "t_ul_start": t_ul_start,
                "t_ul_done": t_ul_done,
                "t_down": t_down,
                "t_train": t_train_s,
                "t_up": t_up,
                "dl_ok": dl_ok,
                "ul_ok": ul_ok,
                "dl_goodput_mbps": float(dlr.goodput_mbps),
                "ul_goodput_mbps": float(ulr.goodput_mbps),
                "dl_rtt_ms": float(dlr.rtt_ms),
                "ul_rtt_ms": float(ulr.rtt_ms),
                "e_comp_j": e_comp_j,
                "co2_comp_g": co2_comp_g,
                "e_comm_j": e_comm_j,
                "co2_comm_g": co2_comm_g,
                "co2_total_g": co2_total_g,
            })

        # Aggregate only committed updates (FedAvg weighted by num_examples if provided)
        if len(committed_clients) == 0:
            new_params = None
            global_norm = float("nan")
        else:
            # Each client sends a single ndarray (model vector)
            # We do simple mean (or weighted mean if 'num_examples' is provided)
            weights = []
            vecs = []
            for cp, fitres in results:
                cid = cp.cid
                if cid not in committed_clients:
                    continue
                nds = parameters_to_ndarrays(fitres.parameters)
                v = nds[0].astype(np.float32)
                num_ex = int(fitres.num_examples) if fitres.num_examples is not None else 1
                weights.append(num_ex)
                vecs.append(v * num_ex)

            denom = float(sum(weights)) if weights else 1.0
            agg = np.sum(vecs, axis=0) / denom
            new_params = ndarrays_to_parameters([agg])
            global_norm = vec_norm(agg)

        # Log
        self.clients_csv.append_rows(client_rows)

        m_selected = len(ctx.get("selected_client_cids", []))
        m_committed = len(committed_clients)
        dropout_rate = 0.0 if m_selected == 0 else (m_selected - m_committed) / float(m_selected)

        self.server_csv.append_rows([{
            "run_id": self.run_id,
            "round": server_round,
            "m_selected": m_selected,
            "m_committed": m_committed,
            "dropout_rate": dropout_rate,
            "global_model_norm": global_norm,
            "co2_committed_g": co2_committed,
            "co2_dropped_g": co2_dropped,
            "co2_total_g": co2_committed + co2_dropped,
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
        }])

        print(
            f"[round {server_round}] selectedClients={m_selected} committed={m_committed} "
            f"dropout={dropout_rate:.2f} model_norm={global_norm:.3f}"
        )

        # Advance virtual time to deadline
        self.t_sim = t_deadline

        return new_params, {"committed": m_committed, "dropout_rate": dropout_rate}

    # Disable evaluation for MVP
    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        return []

    def aggregate_evaluate(self, server_round: int, results, failures):
        return None, {}


def main() -> None:
    # ---- env config (easy to tweak from docker-compose) ----
    OUT_DIR = env_str("OUT_DIR", "outputs")

    VEINS_MODE = env_str("VEINS_MODE", "mock")   # mock | rpc
    VEINS_HOST = env_str("VEINS_HOST", "127.0.0.1")
    VEINS_PORT = env_int("VEINS_PORT", 9999)

    MAP_SIZE_M = env_float("MAP_SIZE_M", 1000.0)
    NUM_VEH = env_int("NUM_VEH", 100)
    RSU_X = env_float("RSU_X_M", 500.0)
    RSU_Y = env_float("RSU_Y_M", 500.0)
    RSU_R = env_float("RSU_R_M", 300.0)

    SEED = env_int("SEED", 42)
    ROUNDS = env_int("ROUNDS", 5)
    M = env_int("M", 10)
    DEADLINE_S = env_float("DEADLINE_S", 25.0)

    MODEL_DOWN = env_int("MODEL_DOWN_BYTES", 2_000_000)
    MODEL_UP = env_int("MODEL_UP_BYTES", 2_000_000)

    CI = env_float("CI_G_PER_KWH", 200.0)
    P_RX = env_float("P_RX_W", 1.0)
    P_TX = env_float("P_TX_W", 1.5)

    SERVER_ADDR = env_str("SERVER_ADDR", "0.0.0.0:8080")

    strat = OrchestratedFedAvg(
        veins_mode=VEINS_MODE,
        veins_host=VEINS_HOST,
        veins_port=VEINS_PORT,
        map_size_m=MAP_SIZE_M,
        num_vehicles=NUM_VEH,
        rsu_x_m=RSU_X,
        rsu_y_m=RSU_Y,
        rsu_radius_m=RSU_R,
        seed=SEED,
        rounds=ROUNDS,
        m=M,
        deadline_s=DEADLINE_S,
        model_down_bytes=MODEL_DOWN,
        model_up_bytes=MODEL_UP,
        ci_g_per_kwh=CI,
        p_rx_w=P_RX,
        p_tx_w=P_TX,
        out_dir=OUT_DIR,
    )

    # Start Flower server (deprecated warning is OK for MVP)
    fl.server.start_server(
        server_address=SERVER_ADDR,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strat,
    )


if __name__ == "__main__":
    main()
