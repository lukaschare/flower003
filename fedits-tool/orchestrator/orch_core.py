#orch_core.py: OrchestratorCore class implementing the control-plane logic for FL orchestration with Veins as the judge for communication and timing. It handles client selection, simulates downlink/uplink, and decides commit/drop based on results and deadlines. It also logs detailed per-client and per-round metrics to CSV files.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import csv
import random
import urllib.parse
import urllib.request
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Any

from orchestrator.veins_client import (
    MockVeinsClient,
    RPCVeinsClient,
    VeinsState,
    DLResult,
    ULResult,
    BaseVeinsClient,
)

# NEW: dataset partitioning
from orchestrator.dataset_partitions import (
    ensure_iid_partitions,
    ensure_dirichlet_partitions,
    default_partition_path,
)


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

def normalize_drop_reason(raw: str) -> str:
    r = (raw or "").strip().lower()

    # 1) out_of_map / left_map
    if r in {"left_map", "veh_missing", "mobility_missing"} or "missing" in r or "left_map" in r:
        return "left_map"

    # 2) out_of_range
    if r in {"out_of_range", "left_coverage", "dl_left_range", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
        return "out_of_range"
    if "coverage" in r or "out_of_range" in r or "left_range" in r:
        return "out_of_range"

    # 3) bad_signal / deadline / others
    return "bad_signal"

# --- 请将这部分代码贴在文件的顶部区域（例如 normalize_drop_reason 后面） ---

def reason_group(raw_reason: str) -> str:
    r = (raw_reason or "").strip().lower()
    if r in {"veh_missing", "mobility_missing", "veh_not_found", "left_map", "out_of_map", "veh_gone", "no_host"}:
        return "out_of_map"
    if "missing" in r or "left_map" in r or "out_of_map" in r:
        return "out_of_map"
    if r in {"left_coverage", "out_of_range", "dl_left_range", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
        return "out_of_range"
    if "coverage" in r or "out_of_range" in r or "left_range" in r:
        return "out_of_range"
    if r in {"deadline", "dl_deadline_miss", "ul_deadline_miss", "ul_start_after_deadline", "link_down", "ul_link_down", "bad_signal"}:
        return "bad_signal_or_deadline"
    if "deadline" in r or "signal" in r or "link" in r or "per" in r or "pdr" in r:
        return "bad_signal_or_deadline"
    return "bad_signal_or_deadline"

def _get_attr(obj, name: str, default: float = float("nan")) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float(default)

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
    # map_size_m: float = 1000.0
    # num_vehicles: int = 100
    # rsu_x_m: float = 500.0
    # rsu_y_m: float = 500.0
    # rsu_radius_m: float = 300.0

    # sim 
    map_size_m: float = 2500.0  # 改为 2500
    num_vehicles: int = 100
    rsu_x_m: float = 1250.0     # 改为 1250
    rsu_y_m: float = 1250.0     # 改为 1250
    rsu_radius_m: float = 500.0 # 改为 500

    # fl
    rounds: int = 10
    clients_per_round: int = 10
    deadline_s: float = 25.0
    model_down_bytes: int = 2_000_000
    model_up_bytes: int = 2_000_000

    # carbon + radio power
    ci_g_per_kwh: float = 153.0
    p_rx_w: float = 1.0
    p_tx_w: float = 1.5

    # === 新增配置 ===
    ci_mode: str = "fixed"       # fixed | electricitymaps
    emaps_token: str = ""        # auth-token
    emaps_zone: str = "ES"       # Zone key
    ci_cache_s: int = 300        # Cache TTL
    ci_timeout_s: float = 5.0    # Timeout
    # ================

    # misc
    seed: int = 42
    out_dir: str = "outputs"
    # veins_mode: str = "mock"  # mock | rpc
    veins_mode: str = "rpc"  # mock | rpc
    veins_host: str = "127.0.0.1"
    veins_port: int = 9999

    # dataset/partition (NEW)
    dataset: str = "cifar10"
    data_dir: str = "/app/data"
    dataset_train_size: int = 50_000  # CIFAR-10 train

    # partition_scheme: str = "iid"
    # partition_scheme: str = "non-iid" # "iid" | "non-iid"
    # PARTITION_SCHEME: str = "dirichlet"

    # "iid" | "dirichlet" (Dirichlet label-skew non-IID)
    # partition_scheme: str = "iid"
    partition_scheme: str = "dirichlet"   # default non-iid
    partition_path: str = ""          # optional override
    dirichlet_alpha: float = 0.5



# 3. 新增 Provider 类 (放在 OrchestratorConfig 后面，OrchestratorCore 前面)
class CarbonIntensityProvider:
    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.mode = (cfg.ci_mode or "fixed").lower().strip()
        self.fixed_ci = float(cfg.ci_g_per_kwh)
        self.token = (cfg.emaps_token or "").strip()
        self.zone = (cfg.emaps_zone or "ES").strip()
        self.base_url = "https://api.electricitymap.org"
        self.cache_s = int(cfg.ci_cache_s) if cfg.ci_cache_s else 300
        self.timeout_s = float(cfg.ci_timeout_s) if cfg.ci_timeout_s else 5.0
        
        self._cache_ts: float = 0.0
        self._cache_ci: float = self.fixed_ci

    def get_ci_g_per_kwh(self) -> float:
        # 如果不是动态模式或没 Token，直接返回固定值
        if self.mode != "electricitymaps" or not self.token:
            return self.fixed_ci

        now = time.time()
        # 检查缓存
        if self._cache_ts > 0 and (now - self._cache_ts) < self.cache_s:
            return float(self._cache_ci)

        # 构造请求
        params = {"zone": self.zone}
        url = f"{self.base_url}/v3/carbon-intensity/latest?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers={"auth-token": self.token, "accept": "application/json"}
        )

        try:
            print(f"[orch-core] Fetching CI from Electricity Maps ({self.zone})...")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
                data = json.loads(resp.read().decode("utf-8"))
            
            # 提取数据
            ci = float(data.get("carbonIntensity", self.fixed_ci))
            print(f"[orch-core] API Success: CI={ci}")
            
            # 更新缓存
            self._cache_ts = now
            self._cache_ci = ci
            return ci
            
        except Exception as e:
            print(f"[orch-core] WARN: CI API failed ({e}), fallback to fixed={self.fixed_ci}")
            # 失败时也更新缓存时间，防止下一秒立刻重试卡死系统，而是等待 cache_s 后再试
            self._cache_ts = now
            self._cache_ci = self.fixed_ci
            return self.fixed_ci




class OrchestratorCore:
    """
    Strict control-plane core:
    - Veins is the judge for reachability + downlink/uplink timing + failure reason.
    - Flower server is data plane; Orchestrator does NOT aggregate weights.
    - FIXED: We do NOT assume veh_id -> client_id mapping. We bind vehicles to actual cp.cid
      provided by Flower server (available_client_ids) and keep mapping stable when possible.
    """

    def __init__(self, cfg: OrchestratorConfig) -> None:
        self.cfg = cfg
        self.run_id = run_id_now()
        self.t_sim = 0.0  # virtual sim time (round barrier time)
        # === 初始化 Provider ===
        self.ci_provider = CarbonIntensityProvider(cfg)

        # output
        self.run_dir = os.path.join(cfg.out_dir, "runs", self.run_id)
        ensure_dir(self.run_dir)
        with open(os.path.join(self.run_dir, "config_snapshot.json"), "w", encoding="utf-8") as f:
            json.dump({"run_id": self.run_id, "cfg": cfg.__dict__}, f, indent=2)

        # --- dataset partitions (run once at start) ---
        out_dir_abs = os.path.abspath(cfg.out_dir)

        scheme = (cfg.partition_scheme or "iid").lower()
        if scheme in ("iid", "i.i.d"):
            scheme_tag = "iid"
            self.partition_path = cfg.partition_path or default_partition_path(
                out_dir_abs=out_dir_abs,
                dataset=cfg.dataset,
                num_veh=cfg.num_vehicles,
                seed=cfg.seed,
                scheme=scheme_tag,
            )
            ensure_iid_partitions(
                path=self.partition_path,
                dataset=cfg.dataset,
                num_veh=cfg.num_vehicles,
                seed=cfg.seed,
                n_train=cfg.dataset_train_size,
            )

        elif scheme in ("non-iid", "noniid", "dirichlet"):
            scheme_tag = f"dirichlet_a{cfg.dirichlet_alpha}"
            self.partition_path = cfg.partition_path or default_partition_path(
                out_dir_abs=out_dir_abs,
                dataset=cfg.dataset,
                num_veh=cfg.num_vehicles,
                seed=cfg.seed,
                scheme=scheme_tag,
            )
            ensure_dirichlet_partitions(
                path=self.partition_path,
                dataset=cfg.dataset,
                num_veh=cfg.num_vehicles,
                seed=cfg.seed,
                data_dir=cfg.data_dir,
                alpha=cfg.dirichlet_alpha,
                n_train=cfg.dataset_train_size,
            )
        else:
            raise ValueError(f"Unsupported partition_scheme={cfg.partition_scheme}")

        print(f"[orch-core] partitions ready: {self.partition_path}")



        # csv schema
        self.clients_fields = [
            "run_id", "round", "client_id", "veh_id",
            "selected", "committed", "drop_reason",
            "drop_group", "drop_stage", "dl_reason", "ul_reason",
            "dl_dist_start_m", "dl_dist_end_m", "ul_dist_start_m", "ul_dist_end_m",
            "dl_rx_power_dbm", "dl_sinr_db", "dl_per", "ul_rx_power_dbm", "ul_sinr_db", "ul_per",
            "t_round_start", "t_deadline",
            "t_dl_done", "t_ul_start", "t_ul_done",
            "t_down", "t_train", "t_up",
            "dl_ok", "ul_ok",
            "dl_goodput_mbps", "ul_goodput_mbps",
            "dl_rtt_ms", "ul_rtt_ms",
            "e_comp_j", "co2_comp_g",
            "e_comm_j", "co2_comm_g", "co2_total_g",
        ]
        self.server_fields = [
            "run_id", "round",
            "m_selected", "m_committed", "dropout_rate",
            "global_model_norm",
            "co2_committed_g", "co2_dropped_g", "co2_total_g",
            "ci_g_per_kwh_used",
            "t_round_start", "t_deadline",
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

        # per-round context
        self.ctx: Dict[int, Dict[str, Any]] = {}

        # IMPORTANT: dynamic binding between SUMO vehicle id and Flower cp.cid
        self.veh_to_cid: Dict[str, str] = {}
        self.cid_to_veh: Dict[str, str] = {}

        print(f"[orch-core] run_id={self.run_id} out={self.run_dir} veins_mode={cfg.veins_mode}")



    # def finalize_round(self, server_round: int, global_model_norm: float) -> Dict[str, Any]:
    # ...
    # # advance virtual time: end at deadline
    # self.t_sim = float(ctx["t_deadline"])

    # # NEW: keep Veins sim-time aligned even when no clients were selected
    # if self.cfg.veins_mode == "rpc":
    #     try:
    #         # use existing API to force Veins advance to deadline (empty UL)
    #         self.veins.simulate_uplink(
    #             start_times={},
    #             veh_ids=[],
    #             size_bytes=0,
    #             deadline=self.t_sim,
    #         )
    #     except Exception as e:
    #         print(f"[WARN] Veins advance (idle) failed: {e}")

    # return {"ok": True, "t_sim": self.t_sim, "run_id": self.run_id}



    # -----------------------------
    # Binding helpers (veh <-> cp.cid)
    # -----------------------------

    def _refresh_bindings(self, available_client_ids: List[str]) -> None:
        """Drop bindings whose cid is no longer connected."""
        alive = set(available_client_ids)
        # remove dead cids
        dead_cids = [cid for cid in self.cid_to_veh.keys() if cid not in alive]
        for cid in dead_cids:
            veh = self.cid_to_veh.pop(cid, None)
            if veh is not None:
                # only delete if still mapped to same cid
                if self.veh_to_cid.get(veh) == cid:
                    self.veh_to_cid.pop(veh, None)

    def _alloc_cid_for_vehicle(self, veh_id: str, free_cids: List[str], used_cids: set) -> str | None:
        """
        Ensure veh_id has a cid bound.
        Prefer existing binding if still available; else assign a free cid.
        """
        # keep existing if possible
        cid0 = self.veh_to_cid.get(veh_id)
        if cid0 and (cid0 in free_cids or cid0 in used_cids):
            # if already used this round, cannot reuse
            if cid0 in used_cids:
                return None
            # otherwise accept it (and remove from free list if present)
            if cid0 in free_cids:
                free_cids.remove(cid0)
            used_cids.add(cid0)
            self.veh_to_cid[veh_id] = cid0
            self.cid_to_veh[cid0] = veh_id
            return cid0

        # allocate new
        for i, cid in enumerate(list(free_cids)):
            if cid in used_cids:
                continue
            # clear old mapping if any
            old_veh = self.cid_to_veh.get(cid)
            if old_veh is not None and self.veh_to_cid.get(old_veh) == cid:
                self.veh_to_cid.pop(old_veh, None)

            # assign
            free_cids.pop(i)
            used_cids.add(cid)
            self.veh_to_cid[veh_id] = cid
            self.cid_to_veh[cid] = veh_id
            return cid

        return None

    # -----------------------------
    # Round APIs called by server via HTTP
    # -----------------------------

    def configure_fit(self, server_round: int, available_client_ids: List[str]) -> Dict[str, Any]:
        """
        Decide which cp.cid should train for which veh_id this round.
        FIXED: Use available_client_ids directly (cp.cid), NOT synthetic "clientXX".
        """
        t_round_start = self.t_sim
        t_deadline = t_round_start + self.cfg.deadline_s

        # === 1. 获取本轮 CI ===
        ci_now = self.ci_provider.get_ci_g_per_kwh()

        # refresh binding with currently alive cids
        self._refresh_bindings(available_client_ids)

        # === [新增日志 1] ===
        print(f"--- [DEBUG] Round {server_round} Start (t={t_round_start}) ---")
        print(f"--- [DEBUG] Asking Veins for vehicles at t={t_round_start}...")


        # query Veins for in-range candidates
        state: VeinsState = self.veins.get_state(t=t_round_start)
        candidates_veh = [vid for vid, v in state.vehicles.items() if v.in_range]

        # 【新增这行打印】
        print(f"[Round {server_round}] Debug: Veins_in_range={len(candidates_veh)}, Flower_connected={len(available_client_ids)}")

        # === [新增日志 2] ===
        print(f"--- [DEBUG] Veins response: Total Vehicles={len(state.vehicles)}, In-Range (Radius={self.cfg.rsu_radius_m})={len(candidates_veh)}")
        if len(state.vehicles) > 0:
            # 打印第一辆车的坐标，看看是不是真的很远
            first_veh = list(state.vehicles.values())[0]
            # print(f"--- [DEBUG] Sample Vehicle Pos: x={first_veh.x}, y={first_veh.y}")
            print(f"--- [DEBUG] Sample Vehicle Pos: x={first_veh.x_m}, y={first_veh.y_m}")
        

        # If no candidates or no clients, nothing to do
        if not candidates_veh or not available_client_ids:
            self.ctx[server_round] = {
                "t_round_start": t_round_start,
                "t_deadline": t_deadline,
                "selected_pairs": [],
                "dl": {},
                "pending_server_row": None,
            }
            return {
                "ok": True,
                "run_id": self.run_id,
                "t_round_start": t_round_start,
                "t_deadline": t_deadline,
                "selected": [],
                "train_assignments": [],
            }

        # deterministic selection of vehicles (but randomized w/ seed+round)
        rnd = random.Random(self.cfg.seed + int(server_round))
        candidates_veh = list(candidates_veh)
        rnd.shuffle(candidates_veh)

        # cap by both: desired m, number of in-range vehicles, number of available clients
        m_cap = min(self.cfg.clients_per_round, len(candidates_veh), len(available_client_ids))
        selected_veh = candidates_veh[:m_cap]

        # stable binding: try keep existing veh->cid mapping
        free_cids = sorted(list(available_client_ids))
        used_cids: set = set()

        selected_pairs: List[Tuple[str, str]] = []
        for veh in selected_veh:
            cid = self._alloc_cid_for_vehicle(veh, free_cids, used_cids)
            if cid is None:
                continue
            selected_pairs.append((cid, veh))

        veh_ids = [veh for _, veh in selected_pairs]

        # simulate downlink in Veins (DL early-stop should be implemented in ControlServer)
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
                        # "ci_g_per_kwh": self.cfg.ci_g_per_kwh,
                        "ci_g_per_kwh": ci_now, # <--- 这里注入当前轮的 CI
                        "t_round_start": t_round_start,
                        "t_deadline": t_deadline,
                        "server_round": server_round,

                        # NEW: dataset partition routing
                        "partition_path": self.partition_path,
                        "dataset": self.cfg.dataset,
                        "partition_scheme": self.cfg.partition_scheme,
                        "num_veh": self.cfg.num_vehicles,
                        "dirichlet_alpha": self.cfg.dirichlet_alpha,
                    },
                })

        # store ctx for decide_commit/finalize and logging
        self.ctx[server_round] = {
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
            "ci_g_per_kwh": ci_now,  # <--- 存入 ctx
            "selected_pairs": selected_pairs,  # includes dl_fail pairs
            "dl": dl,
            "pending_server_row": None,
        }

        return {
            "ok": True,
            "run_id": self.run_id,
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
            "selected": [{"client_id": cid, "veh_id": veh} for cid, veh in selected_pairs],
            "train_assignments": assignments,  # only dl_ok
        }

    def decide_commit(self, server_round: int, fit_results: List[dict]) -> Dict[str, Any]:
        """
        Using client train results (t_train_s, e_comp_j, co2_comp_g) and DL timings,
        compute UL start_times per veh and ask Veins to simulate uplink.
        Then commit/drop based on UL success and deadline.
        """
        if server_round not in self.ctx:
            return {"ok": False, "error": f"no ctx for round {server_round}"}

        ctx = self.ctx[server_round]
        # 读取当轮存储的 CI，用于计算通信能耗
        ci_round = float(ctx.get("ci_g_per_kwh", self.cfg.ci_g_per_kwh))

        t_round_start = float(ctx["t_round_start"])
        t_deadline = float(ctx["t_deadline"])
        selected_pairs: List[Tuple[str, str]] = ctx["selected_pairs"]
        dl: Dict[str, DLResult] = ctx["dl"]

        # map metrics by client_id and also by veh_id as fallback (robust)
        m_by_cid: Dict[str, dict] = {}
        m_by_veh: Dict[str, dict] = {}
        for r in fit_results:
            cid = str(r.get("client_id", ""))
            veh = str(r.get("veh_id", ""))
            if cid:
                m_by_cid[cid] = r
            if veh:
                m_by_veh[veh] = r

        # uplink start times are per-veh
        start_times: Dict[str, float] = {}
        ul_veh_ids: List[str] = []

        for cid, veh in selected_pairs:
            dlr = dl.get(veh, DLResult.fail("dl_no_result"))
            mr = m_by_cid.get(cid) or m_by_veh.get(veh)
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
            mr = m_by_cid.get(cid) or m_by_veh.get(veh)
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
            # co2_comm_g = comm_carbon_g(e_comm_j, self.cfg.ci_g_per_kwh)
            # 计算通信碳排时使用 ci_round
            co2_comm_g = comm_carbon_g(e_comm_j, ci_round)
            
            co2_total = co2_comp_g + co2_comm_g

            # commit/drop semantics
            # committed = 0
            # drop_reason = ""

            # if not dlr.ok:
            #     drop_reason = dlr.reason or "dl_failed"
            # elif mr is None:
            #     drop_reason = "train_missing"
            # elif not ulr.ok:
            #     drop_reason = ulr.reason or "ul_failed"
            # elif float(ulr.t_done) > t_deadline:
            #     drop_reason = "ul_deadline_miss"
            # else:
            #     committed = 1
            #     committed_cids.append(cid)

            #     dl_reason = str(dlr.reason or "")
            #     ul_reason = str(ulr.reason or "")
            #     if committed == 1:
            #         drop_stage = "OK"
            #         drop_group = "committed"
            #     else:
            #         if not dlr.ok:
            #             drop_stage = "DL"
            #         elif mr is None:
            #             drop_stage = "TRAIN"
            #         else:
            #             drop_stage = "UL"
            #         drop_group = reason_group(drop_reason)

            # if committed == 1:
            #     co2_committed += co2_total
            # else:
            #     co2_dropped += co2_total

            # if committed == 0:
            #     drop_reason = normalize_drop_reason(drop_reason)
            # else:
            #     drop_reason = ""

            
            # ================= 替换从这里开始 =================
            # commit/drop semantics
            committed = 0
            drop_reason = ""
            dl_reason = str(dlr.reason or "")
            ul_reason = str(ulr.reason or "")
            drop_stage = "UNKNOWN"
            drop_group = "unknown"

            if not dlr.ok:
                drop_reason = dlr.reason or "dl_failed"
                drop_stage = "DL"
            elif mr is None:
                drop_reason = "train_missing"
                drop_stage = "TRAIN"
            elif not ulr.ok:
                drop_reason = ulr.reason or "ul_failed"
                drop_stage = "UL"
            elif float(ulr.t_done) > t_deadline:
                drop_reason = "ul_deadline_miss"
                drop_stage = "UL"
            else:
                committed = 1
                committed_cids.append(cid)
                drop_stage = "OK"
                drop_group = "committed"

            if committed == 1:
                co2_committed += co2_total
            else:
                co2_dropped += co2_total
                drop_reason = normalize_drop_reason(drop_reason)
                drop_group = reason_group(drop_reason)
            # ================= 替换到这里结束 =================
            

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
                "drop_group": drop_group,
                "drop_stage": drop_stage,
                "dl_reason": dl_reason,
                "ul_reason": ul_reason,

                "dl_dist_start_m": _get_attr(dlr, "dist_start_m"),
                "dl_dist_end_m": _get_attr(dlr, "dist_end_m"),
                "ul_dist_start_m": _get_attr(ulr, "dist_start_m"),
                "ul_dist_end_m": _get_attr(ulr, "dist_end_m"),
                "dl_rx_power_dbm": _get_attr(dlr, "rx_power_dbm"),
                "dl_sinr_db": _get_attr(dlr, "sinr_db"),
                "dl_per": _get_attr(dlr, "per"),
                "ul_rx_power_dbm": _get_attr(ulr, "rx_power_dbm"),
                "ul_sinr_db": _get_attr(ulr, "sinr_db"),
                "ul_per": _get_attr(ulr, "per"),
            })

        # write per-client rows
        self.clients_csv.append_rows(client_rows)

        m_selected = len(selected_pairs)
        m_committed = len(committed_cids)
        dropout_rate = 0.0 if m_selected == 0 else (m_selected - m_committed) / float(m_selected)

        # store pending server row (needs global_model_norm at finalize)
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
            "ci_g_per_kwh_used": ci_round,
            "t_round_start": t_round_start,
            "t_deadline": t_deadline,
        }

        return {"ok": True, "committed_client_ids": committed_cids}

    # def normalize_reason(r: str) -> str:
    #     r = (r or "").strip().lower()
    #     if r in {"left_map", "veh_missing", "mobility_missing"}:
    #         return "left_map"
    #     if r in {"out_of_range", "left_coverage", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
    #         return "out_of_range"
    #     # 其余全部算 bad_signal（deadline / link_down / train_missing / ul_deadline_miss / dl_failed / ul_failed ...)
    #     return "bad_signal"

    #     drop_reason = normalize_reason(drop_reason)


    # def reason_group(raw_reason: str) -> str:
    #     r = (raw_reason or "").strip().lower()

    #     if r in {"veh_missing", "mobility_missing", "veh_not_found", "left_map", "out_of_map", "veh_gone", "no_host"}:
    #         return "out_of_map"
    #     if "missing" in r or "left_map" in r or "out_of_map" in r:
    #         return "out_of_map"

    #     if r in {"left_coverage", "out_of_range", "dl_left_range", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
    #         return "out_of_range"
    #     if "coverage" in r or "out_of_range" in r or "left_range" in r:
    #         return "out_of_range"

    #     if r in {"deadline", "dl_deadline_miss", "ul_deadline_miss", "ul_start_after_deadline", "link_down", "ul_link_down", "bad_signal"}:
    #         return "bad_signal_or_deadline"
    #     if "deadline" in r or "signal" in r or "link" in r or "per" in r or "pdr" in r:
    #         return "bad_signal_or_deadline"

    #     return "bad_signal_or_deadline"


    # def _get_attr(obj, name: str, default: float = float("nan")) -> float:
    #     try:
    #         return float(getattr(obj, name))
    #     except Exception:
    #         return float(default)


    # def finalize_round(self, server_round: int, global_model_norm: float) -> Dict[str, Any]:
    #     if server_round not in self.ctx:
    #         return {"ok": False, "error": f"no ctx for round {server_round}"}
    #     ctx = self.ctx[server_round]
    #     row = ctx.get("pending_server_row")

    #     # ================= [修改代码 START] =================
    #     # 如果没有 row (说明这轮没人被选中)，我们依然要推进时间，不能报错返回
    #     if row:
    #         row["global_model_norm"] = float(global_model_norm)
    #         self.server_csv.append_rows([row])
    #     # ================= [修改代码 END] =================
    #     # advance virtual time: end at deadline
    #     self.t_sim = float(ctx["t_deadline"])  # <--- 这行代码得以执行，时间就动了！

    #     return {"ok": True, "t_sim": self.t_sim, "run_id": self.run_id}

    #     # if not row:
    #     #     return {"ok": False, "error": f"no pending_server_row for round {server_round}"}

    #     # row["global_model_norm"] = float(global_model_norm)
    #     # self.server_csv.append_rows([row])

    #     # # advance virtual time: end at deadline
    #     # self.t_sim = float(ctx["t_deadline"])

    #     # return {"ok": True, "t_sim": self.t_sim, "run_id": self.run_id}

    def finalize_round(self, server_round: int, global_model_norm: float) -> Dict[str, Any]:
        if server_round not in self.ctx:
            return {"ok": False, "error": f"no ctx for round {server_round}"}

        ctx = self.ctx[server_round]
        row = ctx.get("pending_server_row")

        # 1) 这轮有 server_row 才写日志；没人选中就跳过（但仍要推进时间）
        if row:
            row["global_model_norm"] = float(global_model_norm)
            self.server_csv.append_rows([row])

        # 2) 推进 orchestrator 的虚拟时间到 deadline
        self.t_sim = float(ctx["t_deadline"])

        # 3) 关键：即使没人选中，也要把 Veins 推进到 deadline（否则 Veins 永远停在 t=0）
        if getattr(self.cfg, "veins_mode", "") == "rpc":
            try:
                # 空 uplink：让 ControlServer 进入 RUN_XFER 并 tick 到 deadline
                self.veins.simulate_uplink(
                    start_times={},
                    veh_ids=[],
                    size_bytes=0,
                    deadline=self.t_sim,
                )
            except Exception as e:
                print(f"[WARN] Veins idle-advance failed in finalize_round: {e}")

        return {"ok": True, "t_sim": self.t_sim, "run_id": self.run_id}

    def reason_group(raw_reason: str) -> str:
        r = (raw_reason or "").strip().lower()

        # 1) out_of_map
        if r in {"veh_missing", "mobility_missing", "veh_not_found", "left_map", "out_of_map", "veh_gone", "no_host"}:
            return "out_of_map"
        if "missing" in r or "left_map" in r or "out_of_map" in r:
            return "out_of_map"

        # 2) out_of_range
        if r in {"left_coverage", "out_of_range", "dl_left_range", "ul_left_range", "ul_out_of_range", "dl_out_of_range"}:
            return "out_of_range"
        if "coverage" in r or "out_of_range" in r or "left_range" in r:
            return "out_of_range"

        # 3) bad_signal_or_deadline
        if r in {"deadline", "dl_deadline_miss", "ul_deadline_miss", "ul_start_after_deadline", "link_down", "ul_link_down", "bad_signal"}:
            return "bad_signal_or_deadline"
        if "deadline" in r or "signal" in r or "link" in r or "per" in r or "pdr" in r:
            return "bad_signal_or_deadline"

        return "bad_signal_or_deadline"
