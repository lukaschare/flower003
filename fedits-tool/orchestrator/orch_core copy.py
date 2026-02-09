#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, os, csv, random, re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from orchestrator.veins_client import MockVeinsClient, RPCVeinsClient, VeinsState, DLResult, ULResult, BaseVeinsClient


def run_id_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def joule_to_kwh(j: float) -> float:
    return j / 3_600_000.0

def comm_energy_j(p_rx_w: float, p_tx_w: float, t_down_s: float, t_up_s: float) -> float:
    return p_rx_w * max(t_down_s, 0.0) + p_tx_w * max(t_up_s, 0.0)

def comm_carbon_g(e_comm_j: float, ci_g_per_kwh: float) -> float:
    return joule_to_kwh(e_comm_j) * ci_g_per_kwh


class CsvAppender:
    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        ensure_dir(os.path.dirname(path))
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


@dataclass
class OrchestratorConfig:
    # sim
    map_size_m: float = 1000.0
    num_vehicles: int = 100
    rsu_x_m: float = 500.0
    rsu_y_m: float = 500.0
    rsu_radius_m: float = 300.0

    # fl
    rounds: int = 10
    clients_per_round: int = 10
    deadline_s: float = 25.0
    model_down_bytes: int = 2_000_000
    model_up_bytes: int = 2_000_000

    # carbon + radio power
    ci_g_per_kwh: float = 200.0
    p_rx_w: float = 1.0
    p_tx_w: float = 1.5

    # misc
    seed: int = 42
    out_dir: str = "outputs"
    veins_mode: str = "mock"   # mock | rpc
    veins_host: str = "127.0.0.1"
    veins_port: int = 9999


class OrchestratorCore:
    """
    Strict control-plane core:
    - Talks to Veins (mock or rpc) for reachability + downlink/uplink timing + success reason.
    - Talks to Flower server via HTTP service (handled in orch_service.py).
    - Does NOT aggregate model weights (server does), but decides commit/drop and logs.
    """

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.run_id = run_id_now()
        self.t_sim = 0.0  # virtual sim time

        # output
        self.run_dir = os.path.join(cfg.out_dir, "runs", self.run_id)
        ensure_dir(self.run_dir)
        with open(os.path.join(self.run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "cfg": cfg.__dict__}, f, indent=2)

        # csv schema (match your existing fields)
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

        # veins client
        if cfg.veins_mode == "rpc":
            self.veins: BaseVeinsClient = RPCVeinsClient(cfg.veins_host, cfg.veins_port, timeout_s=30.0)
        else:
            self.veins = MockVeinsClient(
                map_size_m=cfg.map_size_m,
                num_vehicles=cfg.num_vehicles,
                rsu_x_m=cfg.rsu_x_m,
                rsu_y_m=cfg.rsu_y_m,
                rsu_radius_m=cfg.rsu_radius_m,
                seed=cfg.seed,
            )

        # per-round context (control-plane memory)
        self.ctx: Dict[int, Dict[str, Any]] = {}

        print(f"[orch-core] run_id={self.run_id} out={self.run_dir} veins_mode={cfg.veins_mode}")

    # ---------- binding: veh_id -> client_id ----------
    def bind_veh_to_client(self, veh_id: str) -> str:
        # default: veh17 -> client17 (2-digit) ; if >=100 use 3-digit
        m = re.search(r"(\d+)", veh_id)
        if m:
            n = int(m.group(1))
        else:
            n = abs(hash(veh_id)) % max(self.cfg.num_vehicles, 1)
        if n <= 99:
            return f"client{n:02d}"
        return f"client{n:03d}"

    # ---------- round APIs used by Flower Strategy ----------
    def configure_fit(self, server_round: int, available_client_ids: List[str]) -> Dict[str, Any]:
        t_round_start = self.t_sim
        t_deadline = t_round_start + self.cfg.deadline_s

        state: VeinsState = self.veins.get_state(t=t_round_start)
        candidates_veh = [vid for vid, v in state.vehicles.items() if v.in_range]

        available = set(available_client_ids)
        candidate_pairs: List[Tuple[str,str]] = []  # (cid, veh)
        for veh in candidates_veh:
            cid = self.bind_veh_to_client(veh)
            if cid in available:
                candidate_pairs.append((cid, veh))

        rnd = random.Random(self.cfg.seed + server_round)
        if len(candidate_pairs) <= self.cfg.clients_per_round:
            selected_pairs = candidate_pairs[:]
        else:
            selected_pairs = rnd.sample(candidate_pairs, self.cfg.clients_per_round)

        veh_ids = [veh for cid, veh in selected_pairs]

        dl = self.veins.simulate_downlink(
            t_now=t_round_start,
            veh_ids=veh_ids,
            size_bytes=self.cfg.model_down_bytes,
            deadline=t_deadline,
        )

        # only dl_ok clients actually train
        assignments: List[dict] = []
        for cid, veh in selected_pairs:
            dlr = dl.get(veh, DLResult.fail("dl_no_result"))
            if dlr.ok:
                assignments.append({
                    "client_id": cid,
                    "veh_id": veh,
                    "fit_config": {
                        "veh_id": veh,
                        "ci_g_per_kwh": self.cfg.ci_g_per_kwh,
                        "t_round_start": t_round_start,
                        "t_deadline": t_deadline,
                        "server_round": server_round,
                    }
                })

        # store ctx for later commit decision & logging
        self.ctx[server_round] = {
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
            "selected_pairs": selected_pairs,   # includes dl_fail
            "dl": dl,
            "pending_server_row": None,
        }

        return {
            "ok": True,
            "run_id": self.run_id,
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
            "selected": [{"client_id": cid, "veh_id": veh} for cid, veh in selected_pairs],
            "train_assignments": assignments,   # only dl_ok
        }

    def decide_commit(self, server_round: int, fit_results: List[dict]) -> Dict[str, Any]:
        if server_round not in self.ctx:
            return {"ok": False, "error": f"no ctx for round {server_round}"}
        ctx = self.ctx[server_round]
        t_round_start = float(ctx["t_round_start"])
        t_deadline = float(ctx["t_deadline"])
        selected_pairs: List[Tuple[str,str]] = ctx["selected_pairs"]
        dl: Dict[str, DLResult] = ctx["dl"]

        # map metrics by client_id
        m_by_cid: Dict[str, dict] = {}
        for r in fit_results:
            cid = str(r.get("client_id", ""))
            m_by_cid[cid] = r

        # uplink start times are per-veh (veh_id)
        start_times: Dict[str, float] = {}
        ul_veh_ids: List[str] = []

        for cid, veh in selected_pairs:
            dlr = dl.get(veh, DLResult.fail("dl_no_result"))
            mr = m_by_cid.get(cid)
            if dlr.ok and mr:
                t_train = float(mr.get("t_train_s", 0.0))
                start = float(dlr.t_done) + t_train
                start_times[veh] = start
                ul_veh_ids.append(veh)

        ul = self.veins.simulate_uplink(
            start_times=start_times,
            veh_ids=ul_veh_ids,
            size_bytes=self.cfg.model_up_bytes,
            deadline=t_deadline,
        )

        committed_cids: List[str] = []
        client_rows: List[dict] = []

        co2_committed = 0.0
        co2_dropped = 0.0

        for cid, veh in selected_pairs:
            dlr = dl.get(veh, DLResult.fail("dl_no_result"))
            mr = m_by_cid.get(cid)
            ulr = ul.get(veh, ULResult.fail("not_scheduled"))

            selected = 1
            dl_ok = 1 if dlr.ok else 0

            t_dl_done = float(dlr.t_done) if dlr.ok else -1.0
            t_train = float(mr.get("t_train_s", 0.0)) if mr else 0.0

            if dlr.ok and mr:
                t_ul_start = float(start_times.get(veh, -1.0))
            else:
                t_ul_start = -1.0

            ul_ok = 1 if ulr.ok else 0
            t_ul_done = float(ulr.t_done) if ulr.ok else -1.0

            # durations
            t_down = (t_dl_done - t_round_start) if dlr.ok else 0.0
            t_up = (t_ul_done - t_ul_start) if (ulr.ok and t_ul_start >= 0.0) else 0.0

            # energy/carbon
            e_comp_j = float(mr.get("e_comp_j", 0.0)) if mr else 0.0
            co2_comp_g = float(mr.get("co2_comp_g", 0.0)) if mr else 0.0

            e_comm_j = comm_energy_j(self.cfg.p_rx_w, self.cfg.p_tx_w, t_down, t_up)
            co2_comm_g = comm_carbon_g(e_comm_j, self.cfg.ci_g_per_kwh)
            co2_total = co2_comp_g + co2_comm_g

            # commit/drop semantics
            committed = 0
            drop_reason = ""

            if not dlr.ok:
                drop_reason = dlr.reason or "dl_failed"
            elif mr is None:
                drop_reason = "train_missing"
            elif not ulr.ok:
                drop_reason = ulr.reason or "ul_failed"
            elif float(ulr.t_done) > t_deadline:
                drop_reason = "ul_deadline_miss"
            else:
                committed = 1
                committed_cids.append(cid)

            if committed == 1:
                co2_committed += co2_total
            else:
                co2_dropped += co2_total

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
                "t_train": t_train,
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
                "co2_total_g": co2_total,
            })

        # write per-client rows now (strictly control-plane output)
        self.clients_csv.append_rows(client_rows)

        m_selected = len(selected_pairs)
        m_committed = len(committed_cids)
        dropout_rate = 0.0 if m_selected == 0 else (m_selected - m_committed) / float(m_selected)

        # store pending server row (needs global_model_norm from server finalize)
        ctx["pending_server_row"] = {
            "run_id": self.run_id,
            "round": server_round,
            "m_selected": m_selected,
            "m_committed": m_committed,
            "dropout_rate": dropout_rate,
            "global_model_norm": float("nan"),
            "co2_committed_g": co2_committed,
            "co2_dropped_g": co2_dropped,
            "co2_total_g": co2_committed + co2_dropped,
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
        }

        return {"ok": True, "committed_client_ids": committed_cids}

    def finalize_round(self, server_round: int, global_model_norm: float) -> Dict[str, Any]:
        if server_round not in self.ctx:
            return {"ok": False, "error": f"no ctx for round {server_round}"}
        ctx = self.ctx[server_round]
        row = ctx.get("pending_server_row")
        if not row:
            return {"ok": False, "error": f"no pending_server_row for round {server_round}"}

        row["global_model_norm"] = float(global_model_norm)
        self.server_csv.append_rows([row])

        # advance virtual time: end at deadline
        self.t_sim = float(ctx["t_deadline"])

        return {"ok": True, "t_sim": self.t_sim, "run_id": self.run_id}
