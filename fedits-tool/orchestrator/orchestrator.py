#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orchestrator: round state machine
select → downlink(sim) → train(real/mock) → uplink(sim) → commit/drop → aggregate → log

Default mode is MOCK so you can run immediately without Veins/Flower integration.
Later, switch to --mode rpc and implement the Veins ControlServer side.

Outputs:
- outputs/runs/<run_id>/clients_round.csv
- outputs/runs/<run_id>/server_round.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Make local imports work even when run as a script
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from veins_client import MockVeinsClient, RPCVeinsClient, VeinsState, DLResult, ULResult
from flwr_adapter import MockFlwrAdapter, TrainResult


# -----------------------------
# Config + Data Structures
# -----------------------------

@dataclass
class RsuConfig:
    x_m: float = 500.0
    y_m: float = 500.0
    radius_m: float = 300.0


@dataclass
class SimConfig:
    map_size_m: float = 1000.0             # 1km x 1km
    num_vehicles: int = 100
    rsu: RsuConfig = RsuConfig()

    rounds: int = 10
    clients_per_round: int = 10            # m
    round_deadline_s: float = 25.0         # virtual deadline length

    # payload sizes (bytes) - adjust later based on model
    model_down_bytes: int = 2_000_000      # ~2MB
    model_up_bytes: int = 2_000_000        # ~2MB

    # carbon-intensity for mock (gCO2/kWh). Replace with your CI trace later.
    ci_g_per_kwh: float = 200.0

    # simple radio power model for comm carbon (Watts)
    p_rx_w: float = 1.0
    p_tx_w: float = 1.5

    # selection policy
    selection: str = "random"              # for now: random from in-range candidates

    seed: int = 42


@dataclass
class ClientRoundRow:
    run_id: str
    round: int
    veh_id: str

    selected: int
    committed: int
    drop_reason: str

    # virtual time stamps
    t_round_start: float
    t_dl_done: float
    t_ul_start: float
    t_ul_done: float

    # durations
    t_down: float
    t_train: float
    t_up: float

    # link metrics (from Veins)
    dl_ok: int
    ul_ok: int
    dl_goodput_mbps: float
    ul_goodput_mbps: float
    dl_rtt_ms: float
    ul_rtt_ms: float

    # energy/carbon accounting
    e_comp_j: float
    co2_comp_g: float

    e_comm_j: float
    co2_comm_g: float

    co2_total_g: float


@dataclass
class ServerRoundRow:
    run_id: str
    round: int

    m_selected: int
    m_committed: int
    dropout_rate: float

    # basic model value (toy)
    global_model_norm: float

    # carbon summary
    co2_committed_g: float
    co2_dropped_g: float
    co2_total_g: float


# -----------------------------
# Helpers
# -----------------------------

def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def joule_to_kwh(j: float) -> float:
    # 1 kWh = 3.6e6 Joules
    return j / 3_600_000.0


def comm_energy_j(p_rx_w: float, p_tx_w: float, t_down_s: float, t_up_s: float) -> float:
    return p_rx_w * max(t_down_s, 0.0) + p_tx_w * max(t_up_s, 0.0)


def comm_carbon_g(e_comm_j: float, ci_g_per_kwh: float) -> float:
    return joule_to_kwh(e_comm_j) * ci_g_per_kwh


def vec_norm(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def aggregate_updates(global_model: List[float], updates: Dict[str, List[float]]) -> List[float]:
    """Toy aggregator: global_model + mean(update)"""
    if not updates:
        return global_model
    dim = len(global_model)
    mean_upd = [0.0] * dim
    for upd in updates.values():
        for i in range(dim):
            mean_upd[i] += upd[i]
    n = float(len(updates))
    mean_upd = [x / n for x in mean_upd]
    return [global_model[i] + mean_upd[i] for i in range(dim)]


# -----------------------------
# Selection
# -----------------------------

def select_clients(cfg: SimConfig, candidates: List[str]) -> List[str]:
    # for MVP: random sampling from candidates
    import random
    rnd = random.Random(cfg.seed + len(candidates))
    if len(candidates) <= cfg.clients_per_round:
        return candidates[:]
    return rnd.sample(candidates, cfg.clients_per_round)


# -----------------------------
# Main Orchestrator
# -----------------------------

def run(cfg: SimConfig, mode: str, host: str, port: int, out_root: str) -> None:
    run_id = now_run_id()
    print(f"[orchestrator] run_id={run_id} mode={mode}")

    run_dir = os.path.join(out_root, "runs", run_id)
    ensure_dir(run_dir)

    # Save config snapshot for reproducibility
    cfg_path = os.path.join(run_dir, "config_snapshot.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "SimConfig": asdict(cfg),
        }, f, indent=2)
    print(f"[orchestrator] saved config: {cfg_path}")

    # Veins client
    if mode == "rpc":
        veins = RPCVeinsClient(host=host, port=port, timeout_s=10.0)
    else:
        veins = MockVeinsClient(
            map_size_m=cfg.map_size_m,
            num_vehicles=cfg.num_vehicles,
            rsu_x_m=cfg.rsu.x_m,
            rsu_y_m=cfg.rsu.y_m,
            rsu_radius_m=cfg.rsu.radius_m,
            seed=cfg.seed,
        )

    # Flower adapter (mock for now)
    fl = MockFlwrAdapter(seed=cfg.seed)

    # A toy global model vector (10 dims)
    global_model = [0.0] * 10

    clients_rows: List[dict] = []
    server_rows: List[dict] = []

    # Virtual simulation time
    t_sim = 0.0

    for r in range(1, cfg.rounds + 1):
        t_round_start = t_sim
        t_deadline = t_round_start + cfg.round_deadline_s

        # 1) query Veins state (positions + inRange)
        state: VeinsState = veins.get_state(t=t_round_start)
        candidates = [vid for vid, v in state.vehicles.items() if v.in_range]
        selected = select_clients(cfg, candidates)

        # 2) simulate downlink
        dl: Dict[str, DLResult] = veins.simulate_downlink(
            t_now=t_round_start,
            veh_ids=selected,
            size_bytes=cfg.model_down_bytes,
            deadline=t_deadline,
        )

        # clients that successfully received model before deadline
        dl_ok_ids = [vid for vid in selected if dl.get(vid) and dl[vid].ok]

        # 3) train (real/mock)
        train_res: Dict[str, TrainResult] = fl.train_clients(
            round_idx=r,
            client_ids=dl_ok_ids,
            global_model=global_model,
            ci_g_per_kwh=cfg.ci_g_per_kwh,
        )

        # 4) schedule uplink start times based on dl_done + T_train
        ul_start_times: Dict[str, float] = {}
        for vid in dl_ok_ids:
            t_dl_done = dl[vid].t_done
            t_train = train_res[vid].t_train_s
            ul_start_times[vid] = t_dl_done + t_train

        # 5) simulate uplink
        ul: Dict[str, ULResult] = veins.simulate_uplink(
            start_times=ul_start_times,
            veh_ids=dl_ok_ids,
            size_bytes=cfg.model_up_bytes,
            deadline=t_deadline,
        )

        # 6) commit/drop + aggregate
        committed_updates: Dict[str, List[float]] = {}
        co2_committed = 0.0
        co2_dropped = 0.0

        for vid in selected:
            dlr = dl.get(vid, DLResult.fail())
            tr = train_res.get(vid)  # only exists if dl ok
            ulr = ul.get(vid, ULResult.fail(reason="not_scheduled"))

            selected_flag = 1
            dl_ok = 1 if dlr.ok else 0

            # compute times
            t_dl_done = dlr.t_done if dlr.ok else -1.0
            t_train = tr.t_train_s if tr else 0.0

            if dlr.ok and tr:
                t_ul_start = ul_start_times[vid]
            else:
                t_ul_start = -1.0

            t_ul_done = ulr.t_done if ulr.ok else -1.0
            ul_ok = 1 if ulr.ok else 0

            # duration
            t_down = (t_dl_done - t_round_start) if dlr.ok else 0.0
            t_up = (t_ul_done - t_ul_start) if (ulr.ok and t_ul_start >= 0) else 0.0

            # commit rule (MVP): ul ok AND arrives before deadline
            if dlr.ok and tr and ulr.ok and (t_ul_done <= t_deadline):
                committed = 1
                drop_reason = ""
                committed_updates[vid] = tr.update_vec
            else:
                committed = 0
                if not dlr.ok:
                    drop_reason = dlr.reason or "dl_failed"
                elif tr is None:
                    drop_reason = "train_missing"
                elif not ulr.ok:
                    drop_reason = ulr.reason or "ul_failed"
                else:
                    drop_reason = "deadline_miss"

            # energy/carbon
            e_comp_j = tr.e_comp_j if tr else 0.0
            co2_comp_g = tr.co2_comp_g if tr else 0.0

            e_comm_j = comm_energy_j(cfg.p_rx_w, cfg.p_tx_w, t_down, t_up)
            co2_comm_g = comm_carbon_g(e_comm_j, cfg.ci_g_per_kwh)
            co2_total_g = co2_comp_g + co2_comm_g

            # bookkeep committed vs dropped carbon (total carbon happened; but "useful" is committed)
            if committed == 1:
                co2_committed += co2_total_g
            else:
                co2_dropped += co2_total_g

            row = ClientRoundRow(
                run_id=run_id,
                round=r,
                veh_id=vid,
                selected=selected_flag,
                committed=committed,
                drop_reason=drop_reason,
                t_round_start=t_round_start,
                t_dl_done=t_dl_done,
                t_ul_start=t_ul_start,
                t_ul_done=t_ul_done,
                t_down=t_down,
                t_train=t_train,
                t_up=t_up,
                dl_ok=dl_ok,
                ul_ok=ul_ok,
                dl_goodput_mbps=dlr.goodput_mbps,
                ul_goodput_mbps=ulr.goodput_mbps,
                dl_rtt_ms=dlr.rtt_ms,
                ul_rtt_ms=ulr.rtt_ms,
                e_comp_j=e_comp_j,
                co2_comp_g=co2_comp_g,
                e_comm_j=e_comm_j,
                co2_comm_g=co2_comm_g,
                co2_total_g=co2_total_g,
            )
            clients_rows.append(asdict(row))

        # aggregate
        global_model = aggregate_updates(global_model, committed_updates)

        m_selected = len(selected)
        m_committed = len(committed_updates)
        dropout_rate = 0.0 if m_selected == 0 else (m_selected - m_committed) / float(m_selected)

        srv = ServerRoundRow(
            run_id=run_id,
            round=r,
            m_selected=m_selected,
            m_committed=m_committed,
            dropout_rate=dropout_rate,
            global_model_norm=vec_norm(global_model),
            co2_committed_g=co2_committed,
            co2_dropped_g=co2_dropped,
            co2_total_g=co2_committed + co2_dropped,
        )
        server_rows.append(asdict(srv))

        # advance virtual time: end of round at deadline (sync FL)
        t_sim = t_deadline

        print(
            f"[round {r}] candidates={len(candidates)} selected={m_selected} "
            f"committed={m_committed} dropout={dropout_rate:.2f} "
            f"model_norm={srv.global_model_norm:.3f}"
        )

    # write logs
    clients_csv = os.path.join(run_dir, "clients_round.csv")
    server_csv = os.path.join(run_dir, "server_round.csv")

    # field order
    client_fields = list(asdict(ClientRoundRow(
        run_id="", round=0, veh_id="", selected=0, committed=0, drop_reason="",
        t_round_start=0, t_dl_done=0, t_ul_start=0, t_ul_done=0,
        t_down=0, t_train=0, t_up=0,
        dl_ok=0, ul_ok=0,
        dl_goodput_mbps=0, ul_goodput_mbps=0,
        dl_rtt_ms=0, ul_rtt_ms=0,
        e_comp_j=0, co2_comp_g=0,
        e_comm_j=0, co2_comm_g=0,
        co2_total_g=0
    )).keys())

    server_fields = list(asdict(ServerRoundRow(
        run_id="", round=0, m_selected=0, m_committed=0, dropout_rate=0.0,
        global_model_norm=0.0,
        co2_committed_g=0.0, co2_dropped_g=0.0, co2_total_g=0.0
    )).keys())

    write_csv(clients_csv, clients_rows, client_fields)
    write_csv(server_csv, server_rows, server_fields)

    print(f"[orchestrator] wrote: {clients_csv}")
    print(f"[orchestrator] wrote: {server_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mock", "rpc"], default="mock", help="Veins mode")
    p.add_argument("--host", default="127.0.0.1", help="Veins RPC host")
    p.add_argument("--port", type=int, default=9999, help="Veins RPC port")
    p.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    p.add_argument("--m", type=int, default=10, help="Clients per round")
    p.add_argument("--deadline", type=float, default=25.0, help="Round deadline (virtual seconds)")
    p.add_argument("--out", default="outputs", help="Output root folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimConfig()
    cfg.rounds = args.rounds
    cfg.clients_per_round = args.m
    cfg.round_deadline_s = args.deadline
    run(cfg=cfg, mode=args.mode, host=args.host, port=args.port, out_root=args.out)


if __name__ == "__main__":
    main()
