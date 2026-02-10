#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#orch_service.py: Orchestrator service exposing REST API for control-plane interactions with clients and Veins

from __future__ import annotations

import json, os
import sys

# ==========================================
# 修复导入路径 (关键步骤)
# ==========================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__)) # /app/orchestrator
PARENT_DIR = os.path.dirname(THIS_DIR)                # /app
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
# ==========================================




import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from orchestrator.orch_core import OrchestratorCore, OrchestratorConfig


def env_str(k: str, d: str) -> str:
    return str(os.getenv(k, d))

def env_int(k: str, d: int) -> int:
    try: return int(os.getenv(k, d))
    except: return d

def env_float(k: str, d: float) -> float:
    try: return float(os.getenv(k, d))
    except: return d


def load_cfg_from_env() -> OrchestratorConfig:
    return OrchestratorConfig(
        map_size_m=env_float("MAP_SIZE_M", 1000.0),
        num_vehicles=env_int("NUM_VEH", 100),
        rsu_x_m=env_float("RSU_X_M", 500.0),
        rsu_y_m=env_float("RSU_Y_M", 500.0),
        rsu_radius_m=env_float("RSU_R_M", 300.0),
        rounds=env_int("ROUNDS", 10),
        clients_per_round=env_int("M", 10),
        deadline_s=env_float("DEADLINE_S", 25.0),
        model_down_bytes=env_int("MODEL_DOWN_BYTES", 2_000_000),
        model_up_bytes=env_int("MODEL_UP_BYTES", 2_000_000),
        ci_g_per_kwh=env_float("CI_G_PER_KWH", 200.0),
        p_rx_w=env_float("P_RX_W", 1.0),
        p_tx_w=env_float("P_TX_W", 1.5),
        seed=env_int("SEED", 42),
        out_dir=env_str("OUT_DIR", "outputs"),
        veins_mode=env_str("VEINS_MODE", "mock"),
        veins_host=env_str("VEINS_HOST", "127.0.0.1"),
        veins_port=env_int("VEINS_PORT", 9999),
    )


class Handler(BaseHTTPRequestHandler):
    core: OrchestratorCore = None  # type: ignore

    def _json(self, code: int, obj: Dict[str, Any]) -> None:
        b = (json.dumps(obj) + "\n").encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "run_id": self.core.run_id, "t_sim": self.core.t_sim})
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8")
            req = json.loads(raw) if raw.strip() else {}
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})

        if self.path == "/v1/round/configure_fit":
            resp = self.core.configure_fit(
                server_round=int(req.get("server_round", 0)),
                available_client_ids=list(req.get("available_client_ids", [])),
            )
            return self._json(200, resp)

        if self.path == "/v1/round/decide_commit":
            resp = self.core.decide_commit(
                server_round=int(req.get("server_round", 0)),
                fit_results=list(req.get("fit_results", [])),
            )
            return self._json(200, resp)

        if self.path == "/v1/round/finalize":
            resp = self.core.finalize_round(
                server_round=int(req.get("server_round", 0)),
                global_model_norm=float(req.get("global_model_norm", float("nan"))),
            )
            return self._json(200, resp)

        return self._json(404, {"ok": False, "error": "not found"})


def main() -> None:
    host = env_str("ORCH_HOST", "0.0.0.0")
    port = env_int("ORCH_PORT", 7070)

    cfg = load_cfg_from_env()
    core = OrchestratorCore(cfg)
    Handler.core = core

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[orch-service] listening on http://{host}:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
