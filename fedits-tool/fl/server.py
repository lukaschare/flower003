#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

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
    """Multi-layer weighted average (FedAvg)."""
    total = float(sum(w for _, w in params_and_weights))
    if total <= 0:
        return params_and_weights[0][0]

    n_layers = len(params_and_weights[0][0])
    agg = [np.zeros_like(params_and_weights[0][0][k]) for k in range(n_layers)]
    for params, w in params_and_weights:
        wf = float(w) / total
        for k in range(n_layers):
            agg[k] += params[k] * wf
    return agg


# -------------------------
# Run Logger (JSONL events + CSV metrics)
# -------------------------
@dataclass
class RunPaths:
    run_dir: str
    events_path: str
    metrics_path: str


class RunLogger:
    """Append-only run logger: JSONL events + CSV round metrics (for Streamlit)."""

    def __init__(self, base_dir: str, run_id: Optional[str] = None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, run_id)

        os.makedirs(run_dir, exist_ok=True)

        self.paths = RunPaths(
            run_dir=run_dir,
            events_path=os.path.join(run_dir, "events.jsonl"),
            metrics_path=os.path.join(run_dir, "round_metrics.csv"),
        )
        self._metrics_header_written = os.path.exists(self.paths.metrics_path)

    def log_event(self, event_type: str, server_round: int, payload: Dict[str, Any]) -> None:
        rec = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": event_type,
            "round": int(server_round),
            "payload": payload,
        }
        with open(self.paths.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_round_metrics(
        self,
        server_round: int,
        selected: int,
        received: int,
        kept: int,
        dropped: int,
        agg_scalar: Optional[float],
        train_loss: Optional[float],
        train_acc: Optional[float],
    ) -> None:
        import csv

        row = {
            "round": int(server_round),
            "selected": int(selected),
            "received": int(received),
            "kept": int(kept),
            "dropped": int(dropped),
            "agg_scalar": "" if agg_scalar is None else f"{float(agg_scalar):.8f}",
            "train_loss": "" if train_loss is None else f"{float(train_loss):.6f}",
            "train_acc": "" if train_acc is None else f"{float(train_acc):.6f}",
        }
        write_header = not self._metrics_header_written
        with open(self.paths.metrics_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._metrics_header_written = True
            w.writerow(row)

    def write_manifest(self, extra: Optional[Dict[str, Any]] = None) -> None:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": self.paths.run_dir,
        }
        if extra:
            manifest.update(extra)
        path = os.path.join(self.paths.run_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


# -------------------------
# Optional terminal TUI (rich)
# -------------------------
def optional_progress_board(num_rounds: int, title: str):
    try:
        from progress_ui import ProgressBoard
        return ProgressBoard(num_rounds=num_rounds, title=title)
    except Exception:
        return None


# -------------------------
# Minimal HTTP client
# -------------------------
class OrchestratorHttpClient:
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
    return [v.detach().cpu().numpy() for _, v in m.state_dict().items()]


# -------------------------
# Proxy Strategy
# -------------------------
class ProxyOrchestratedFedAvg(fl.server.strategy.Strategy):
    def __init__(
        self,
        orch_url: str,
        min_fit_clients: int = 1,
        min_available_clients: int = 10,
        wait_timeout: int = 60,
        board: Optional[Any] = None,
        logger: Optional[RunLogger] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.orch = OrchestratorHttpClient(orch_url, timeout_s=90.0)

        self.init_params_nd = build_cifar10_cnn_params()
        self.min_fit_clients = int(min_fit_clients)
        self.min_available_clients = int(min_available_clients)
        self.wait_timeout = int(wait_timeout)

        self.board = board
        self.logger = logger
        self.verbose = bool(verbose)
        self._selected_per_round: Dict[int, List[str]] = {}

        try:
            h = self.orch.get("/health")
            self._log(f"[server] Orchestrator health: {h}")
        except Exception as e:
            self._log(f"[server] Orchestrator not reachable yet: {e}")

    def _log(self, msg: str) -> None:
        if self.board is not None and hasattr(self.board, "log"):
            try:
                self.board.log(msg)
                return
            except Exception:
                pass
        if self.verbose:
            print(msg)

    def _log_event(self, t: str, r: int, payload: Dict[str, Any]) -> None:
        if self.logger is None:
            return
        try:
            self.logger.log_event(t, r, payload)
        except Exception as e:
            self._log(f"[logger] log_event failed: {t} r={r} err={e}")

    def _log_round_metrics(
        self,
        r: int,
        selected: int,
        received: int,
        kept: int,
        dropped: int,
        agg_scalar: Optional[float],
        train_loss: Optional[float],
        train_acc: Optional[float],
    ) -> None:
        if self.logger is None:
            return
        try:
            self.logger.log_round_metrics(r, selected, received, kept, dropped, agg_scalar, train_loss, train_acc)
        except Exception as e:
            self._log(f"[logger] log_round_metrics failed: r={r} err={e}")

    def _board_call(self, name: str, *args) -> None:
        if self.board is None:
            return
        if hasattr(self.board, name):
            try:
                getattr(self.board, name)(*args)
            except Exception:
                pass

    def _get_all_clients(self, client_manager: ClientManager) -> Dict[str, fl.server.client_proxy.ClientProxy]:
        if hasattr(client_manager, "all"):
            return client_manager.all()  # type: ignore
        n = client_manager.num_available()
        sampled = client_manager.sample(n, n)
        return {c.cid: c for c in sampled}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        return ndarrays_to_parameters(self.init_params_nd)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]]:
        self._board_call("on_round_start", server_round)
        self._log_event("ROUND_START", server_round, {})

        wait_num = self.min_available_clients if server_round == 1 else self.min_fit_clients
        self._log(f"[round {server_round}] waiting for >= {wait_num} clients (timeout={self.wait_timeout}s) ...")

        t0 = time.time()
        last_log = 0.0
        while client_manager.num_available() < wait_num:
            now = time.time()
            if now - t0 > self.wait_timeout:
                self._log(f"[round {server_round}] timeout: available={client_manager.num_available()} < {wait_num}, skip round.")
                self._log_event("WAIT_TIMEOUT", server_round, {"available": client_manager.num_available(), "required": wait_num})
                self._board_call("on_round_end")
                self._log_event("ROUND_END", server_round, {"note": "wait_timeout"})
                return []
            if now - last_log >= 2.0:
                self._log_event("WAIT_AVAILABLE", server_round, {"available": client_manager.num_available(), "required": wait_num})
                last_log = now
            time.sleep(0.5)

        all_clients = self._get_all_clients(client_manager)
        available_ids = list(all_clients.keys())

        resp = self.orch.post(
            "/v1/round/configure_fit",
            {"server_round": int(server_round), "available_client_ids": available_ids},
        )
        if not resp.get("ok", False):
            self._log(f"[server] configure_fit failed: {resp}")
            self._log_event("ERROR", server_round, {"stage": "configure_fit", "resp": resp})
            self._board_call("on_round_end")
            self._log_event("ROUND_END", server_round, {"note": "configure_fit_failed"})
            return []

        assignments = resp.get("train_assignments", []) or []
        selected = [a.get("client_id", "") for a in assignments if a.get("client_id", "")]
        self._selected_per_round[server_round] = list(selected)

        self._board_call("on_select", selected)
        self._log_event("SELECT", server_round, {"selected": selected, "available": available_ids})

        if not assignments:
            self._log(f"[round {server_round}] no clients selected (orch empty). finalize to advance time.")
            try:
                self.orch.post("/v1/round/finalize", {"server_round": int(server_round), "global_model_norm": 0.0})
            except Exception as e:
                self._log(f"[server] finalize (empty) failed: {e}")
            # still log a metrics row (optional but nice)
            self._log_round_metrics(
                r=server_round,
                selected=len(selected),
                received=0,
                kept=0,
                dropped=0,
                agg_scalar=None,
                train_loss=None,
                train_acc=None,
            )
            self._board_call("on_round_end")
            self._log_event("ROUND_END", server_round, {"note": "empty_selection"})
            return []

        fit_instructions: List[Tuple[fl.server.client_proxy.ClientProxy, FitIns]] = []
        for a in assignments:
            cid = a.get("client_id", "")
            cfg = a.get("fit_config", {})
            cp = all_clients.get(cid)
            if cp is None:
                continue
            fit_instructions.append((cp, FitIns(parameters, cfg)))

        self._log(f"[round {server_round}] dispatch_fit={len(fit_instructions)} selected={len(selected)}")
        return fit_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        received = [cp.cid for cp, _ in results]
        self._board_call("on_recv", received)
        self._log_event("RECV", server_round, {"received": received, "num_failures": len(failures) if failures else 0})

        # payload to orchestrator (decision uses timing/energy/carbon; keep/drop decided there)
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

        try:
            dec = self.orch.post(
                "/v1/round/decide_commit",
                {"server_round": int(server_round), "fit_results": fit_results_payload},
            )
        except Exception as e:
            dec = {"ok": False, "error": str(e)}

        if not dec.get("ok", False):
            self._log(f"[server] decide_commit failed: {dec}")
            committed = set()
        else:
            committed = set(dec.get("committed_client_ids", []))

        keep = sorted(list(committed))
        drop = [cid for cid in received if cid not in committed]

        self._board_call("on_verdict", keep, drop)
        self._log_event("VERDICT", server_round, {"keep": keep, "drop": drop, "reason": dec.get("reason", {})})

        # --- NEW: aggregate train_loss/train_acc over committed only ---
        w_sum = 0
        loss_w_sum = 0.0
        acc_w_sum = 0.0
        has_loss = False
        has_acc = False

        for cp, fitres in results:
            if cp.cid not in committed:
                continue
            m = dict(fitres.metrics) if fitres.metrics else {}
            w = int(fitres.num_examples) if fitres.num_examples is not None else int(m.get("num_examples", 0))
            if w <= 0:
                continue

            tl = m.get("train_loss", None)
            ta = m.get("train_acc", None)

            if tl is not None:
                loss_w_sum += float(tl) * w
                has_loss = True
            if ta is not None:
                acc_w_sum += float(ta) * w
                has_acc = True
            w_sum += w

        train_loss = (loss_w_sum / w_sum) if (w_sum > 0 and has_loss) else None
        train_acc = (acc_w_sum / w_sum) if (w_sum > 0 and has_acc) else None

        # Aggregate only committed updates (multi-layer FedAvg)
        params_and_weights: List[Tuple[List[np.ndarray], int]] = []
        for cp, fitres in results:
            if cp.cid not in committed:
                continue
            nds = parameters_to_ndarrays(fitres.parameters)
            w = int(fitres.num_examples) if fitres.num_examples is not None else 0
            if w <= 0:
                continue
            params_and_weights.append((nds, w))

        if not params_and_weights:
            new_params = None
            agg_scalar = None
        else:
            agg_nds = aggregate_weighted(params_and_weights)
            new_params = ndarrays_to_parameters(agg_nds)
            agg_scalar = float(params_l2_norm(agg_nds))

        self._board_call("on_agg", agg_scalar)
        self._log_event("AGG", server_round, {"agg_scalar": agg_scalar, "train_loss": train_loss, "train_acc": train_acc})

        selected_cnt = len(self._selected_per_round.get(server_round, []))
        self._log_round_metrics(
            r=server_round,
            selected=selected_cnt,
            received=len(received),
            kept=len(keep),
            dropped=len(drop),
            agg_scalar=agg_scalar,
            train_loss=train_loss,
            train_acc=train_acc,
        )

        # finalize (advance sim time + orchestrator writes server_round.csv)
        try:
            self.orch.post(
                "/v1/round/finalize",
                {"server_round": int(server_round), "global_model_norm": float(agg_scalar) if agg_scalar is not None else 0.0},
            )
        except Exception as e:
            self._log(f"[server] finalize failed: {e}")

        self._board_call("on_round_end")
        self._log_event("ROUND_END", server_round, {"kept": len(keep), "dropped": len(drop), "agg_scalar": agg_scalar, "train_loss": train_loss, "train_acc": train_acc})

        return new_params, {"kept": len(keep), "dropped": len(drop), "agg_scalar": agg_scalar, "train_loss": train_loss, "train_acc": train_acc}

    # disable evaluate (we already log train metrics via clients)
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

    # --- NEW: align run_id with orchestrator, and log to /app/outputs/runs/<run_id>/ ---
    orch = OrchestratorHttpClient(ORCH_URL, timeout_s=10.0)
    run_id: Optional[str] = None
    try:
        h = orch.get("/health")
        if h.get("ok") and h.get("run_id"):
            run_id = str(h["run_id"])
    except Exception:
        run_id = None

    outputs_dir = env_str("OUTPUTS_DIR", "/app/outputs/runs")

    try:
        logger = RunLogger(base_dir=outputs_dir, run_id=run_id)
        logger.write_manifest(
            {
                "server_address": SERVER_ADDR,
                "num_rounds": ROUNDS,
                "orch_url": ORCH_URL,
                "wait_timeout": wait_timeout,
                "min_fit_clients": min_fit_clients,
                "min_available_clients": min_available_clients,
            }
        )
    except Exception as e:
        print(f"[server] FATAL: cannot write outputs_dir={outputs_dir}. "
              f"Fix docker volume permissions or remove :ro. err={e}")
        raise

    run_dir = logger.paths.run_dir
    print(f"[server] logging enabled: {run_dir}")
    # Host-side tip
    print(f"[server] streamlit: choose Run directory (host): outputs/runs/{os.path.basename(run_dir)}")

    board = None
    if env_int("ENABLE_TUI", 0) == 1:
        board = optional_progress_board(num_rounds=ROUNDS, title=f"Run: {os.path.basename(run_dir)}")

    strat = ProxyOrchestratedFedAvg(
        orch_url=ORCH_URL,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        wait_timeout=wait_timeout,
        board=board,
        logger=logger,
        verbose=True,
    )

    config = fl.server.ServerConfig(num_rounds=ROUNDS)

    if board is not None:
        with board as b:
            strat.board = b
            b.log(f"Logging to: {run_dir}")
            b.log(f"Starting Flower server at {SERVER_ADDR}")
            fl.server.start_server(server_address=SERVER_ADDR, config=config, strategy=strat)
    else:
        fl.server.start_server(server_address=SERVER_ADDR, config=config, strategy=strat)


if __name__ == "__main__":
    main()
