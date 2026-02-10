#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Veins client abstraction.

- MockVeinsClient: runs immediately without any OMNeT++ changes.
- RPCVeinsClient: TCP/JSON-lines client to a future ControlServer inside Veins.

JSON-lines (one request per line) suggested contract:
Request:
  {"cmd":"get_state","t":12.3}
  {"cmd":"simulate_downlink","t_now":0.0,"veh_ids":["veh0"],"size_bytes":2000000,"deadline":25.0}
  {"cmd":"simulate_uplink","veh_ids":["veh0"],"size_bytes":2000000,"deadline":25.0,"start_times":{"veh0":4.2}}

Response:
  {"ok":true, ... payload ...}
"""
#veins_client.py
from __future__ import annotations

import json
import math
import random
import socket
from dataclasses import dataclass
from typing import Dict, List, Optional


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class VehicleState:
    x_m: float
    y_m: float
    in_range: bool


@dataclass
class VeinsState:
    vehicles: Dict[str, VehicleState]


@dataclass
class DLResult:
    ok: bool
    t_done: float
    goodput_mbps: float
    rtt_ms: float
    reason: str = ""

    @staticmethod
    def fail(reason: str = "dl_failed") -> "DLResult":
        return DLResult(ok=False, t_done=-1.0, goodput_mbps=0.0, rtt_ms=0.0, reason=reason)


@dataclass
class ULResult:
    ok: bool
    t_done: float
    goodput_mbps: float
    rtt_ms: float
    reason: str = ""

    @staticmethod
    def fail(reason: str = "ul_failed") -> "ULResult":
        return ULResult(ok=False, t_done=-1.0, goodput_mbps=0.0, rtt_ms=0.0, reason=reason)


# -----------------------------
# Base Interface
# -----------------------------

class BaseVeinsClient:
    def get_state(self, t: float) -> VeinsState:
        raise NotImplementedError

    def simulate_downlink(self, t_now: float, veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, DLResult]:
        raise NotImplementedError

    def simulate_uplink(self, start_times: Dict[str, float], veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, ULResult]:
        raise NotImplementedError


# -----------------------------
# Mock Veins (immediate runnable)
# -----------------------------

class MockVeinsClient(BaseVeinsClient):
    """
    Very lightweight mobility + link model:
    - Vehicles move with constant velocity in a torus [0, map_size)
    - in_range determined by distance to RSU <= rsu_radius
    - goodput decreases with distance; if out_of_range => fail
    - transfer fails if it would finish after deadline or leaves range during transfer (rough check)
    """

    def __init__(
        self,
        map_size_m: float,
        num_vehicles: int,
        rsu_x_m: float,
        rsu_y_m: float,
        rsu_radius_m: float,
        seed: int = 42,
    ) -> None:
        self.map_size_m = float(map_size_m)
        self.rsu_x_m = float(rsu_x_m)
        self.rsu_y_m = float(rsu_y_m)
        self.rsu_radius_m = float(rsu_radius_m)

        rnd = random.Random(seed)
        self.veh_init = {}
        self.veh_vel = {}
        for i in range(num_vehicles):
            vid = f"veh{i}"
            x0 = rnd.uniform(0, self.map_size_m)
            y0 = rnd.uniform(0, self.map_size_m)
            # speed 5~20 m/s, random direction
            spd = rnd.uniform(5.0, 20.0)
            ang = rnd.uniform(0, 2.0 * math.pi)
            vx = spd * math.cos(ang)
            vy = spd * math.sin(ang)
            self.veh_init[vid] = (x0, y0)
            self.veh_vel[vid] = (vx, vy)

        self.rnd = random.Random(seed + 999)

    def _pos(self, vid: str, t: float) -> tuple[float, float]:
        x0, y0 = self.veh_init[vid]
        vx, vy = self.veh_vel[vid]
        x = (x0 + vx * t) % self.map_size_m
        y = (y0 + vy * t) % self.map_size_m
        return x, y

    def _dist(self, x: float, y: float) -> float:
        return math.hypot(x - self.rsu_x_m, y - self.rsu_y_m)

    def _in_range(self, x: float, y: float) -> bool:
        return self._dist(x, y) <= self.rsu_radius_m

    def _goodput_mbps(self, dist: float) -> float:
        """
        Simple distance-based goodput:
        near -> up to ~25 Mbps
        at edge -> ~2 Mbps
        """
        if dist <= 1.0:
            return 25.0
        if dist >= self.rsu_radius_m:
            return 0.0
        # smooth decay
        frac = dist / self.rsu_radius_m
        return max(2.0, 25.0 * (1.0 - frac) ** 1.2)

    def _rtt_ms(self, dist: float) -> float:
        # baseline + distance-based
        base = 5.0
        return base + 0.05 * dist + self.rnd.uniform(0.0, 2.0)

    def get_state(self, t: float) -> VeinsState:
        vehicles: Dict[str, VehicleState] = {}
        for vid in self.veh_init.keys():
            x, y = self._pos(vid, t)
            vehicles[vid] = VehicleState(x_m=x, y_m=y, in_range=self._in_range(x, y))
        return VeinsState(vehicles=vehicles)

    def simulate_downlink(self, t_now: float, veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, DLResult]:
        res: Dict[str, DLResult] = {}
        for vid in veh_ids:
            x, y = self._pos(vid, t_now)
            dist = self._dist(x, y)
            if dist > self.rsu_radius_m:
                res[vid] = DLResult.fail("out_of_range")
                continue

            gp = self._goodput_mbps(dist)
            if gp <= 0.0:
                res[vid] = DLResult.fail("link_down")
                continue

            # transfer time (seconds)
            bytes_per_s = gp * 1_000_000 / 8.0
            t_tx = size_bytes / bytes_per_s
            # add small jitter
            t_tx *= self.rnd.uniform(1.0, 1.2)

            t_done = t_now + t_tx
            if t_done > deadline:
                res[vid] = DLResult.fail("dl_deadline_miss")
                continue

            # rough: if vehicle leaves range by t_done -> fail
            x2, y2 = self._pos(vid, t_done)
            if not self._in_range(x2, y2):
                res[vid] = DLResult.fail("dl_left_range")
                continue

            res[vid] = DLResult(ok=True, t_done=t_done, goodput_mbps=gp, rtt_ms=self._rtt_ms(dist))
        return res

    def simulate_uplink(self, start_times: Dict[str, float], veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, ULResult]:
        res: Dict[str, ULResult] = {}
        for vid in veh_ids:
            t_start = start_times.get(vid, None)
            if t_start is None:
                res[vid] = ULResult.fail("no_start_time")
                continue
            if t_start > deadline:
                res[vid] = ULResult.fail("ul_start_after_deadline")
                continue

            x, y = self._pos(vid, t_start)
            dist = self._dist(x, y)
            if dist > self.rsu_radius_m:
                res[vid] = ULResult.fail("ul_out_of_range")
                continue

            gp = self._goodput_mbps(dist)
            if gp <= 0.0:
                res[vid] = ULResult.fail("ul_link_down")
                continue

            bytes_per_s = gp * 1_000_000 / 8.0
            t_tx = size_bytes / bytes_per_s
            # contention-like jitter (slightly larger uplink variance)
            t_tx *= self.rnd.uniform(1.05, 1.35)

            t_done = t_start + t_tx
            if t_done > deadline:
                res[vid] = ULResult.fail("ul_deadline_miss")
                continue

            x2, y2 = self._pos(vid, t_done)
            if not self._in_range(x2, y2):
                res[vid] = ULResult.fail("ul_left_range")
                continue

            res[vid] = ULResult(ok=True, t_done=t_done, goodput_mbps=gp, rtt_ms=self._rtt_ms(dist))
        return res


# -----------------------------
# RPC Veins Client (for real integration later)
# -----------------------------

class RPCVeinsClient(BaseVeinsClient):
    """
    TCP/JSON-lines client. You implement a corresponding ControlServer in Veins.
    """

    def __init__(self, host: str, port: int, timeout_s: float = 10.0) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _call(self, payload: dict) -> dict:
        # === [新增] 打印发送日志 ===
        print(f"--- [RPC DEBUG] Connecting to {self.host}:{self.port} ...")
        print(f"--- [RPC DEBUG] Sending: {json.dumps(payload)}")
        # ==========================

        msg = (json.dumps(payload) + "\n").encode("utf-8")
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
            s.sendall(msg)
            s.shutdown(socket.SHUT_WR)
            data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                data += chunk
        text = data.decode("utf-8").strip()
        # === [新增] 打印接收日志 ===
        print(f"--- [RPC DEBUG] Received raw: {text}")
        # ==========================
        if not text:
            raise RuntimeError("Empty response from Veins ControlServer")
        return json.loads(text)

    def get_state(self, t: float) -> VeinsState:
        resp = self._call({"cmd": "get_state", "t": t})
        if not resp.get("ok", False):
            raise RuntimeError(f"Veins get_state failed: {resp}")
        vehicles = {}
        for vid, v in resp["vehicles"].items():
            vehicles[vid] = VehicleState(
                x_m=float(v["x_m"]),
                y_m=float(v["y_m"]),
                in_range=bool(v["in_range"]),
            )
        return VeinsState(vehicles=vehicles)

    def simulate_downlink(self, t_now: float, veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, DLResult]:
        resp = self._call({
            "cmd": "simulate_downlink",
            "t_now": t_now,
            "veh_ids": veh_ids,
            "size_bytes": int(size_bytes),
            "deadline": float(deadline),
        })
        if not resp.get("ok", False):
            raise RuntimeError(f"Veins simulate_downlink failed: {resp}")
        out = {}
        for vid, r in resp["results"].items():
            out[vid] = DLResult(
                ok=bool(r["ok"]),
                t_done=float(r.get("t_done", -1.0)),
                goodput_mbps=float(r.get("goodput_mbps", 0.0)),
                rtt_ms=float(r.get("rtt_ms", 0.0)),
                reason=str(r.get("reason", "")),
            )
        return out

    def simulate_uplink(self, start_times: Dict[str, float], veh_ids: List[str], size_bytes: int, deadline: float) -> Dict[str, ULResult]:
        resp = self._call({
            "cmd": "simulate_uplink",
            "veh_ids": veh_ids,
            "start_times": start_times,
            "size_bytes": int(size_bytes),
            "deadline": float(deadline),
        })
        if not resp.get("ok", False):
            raise RuntimeError(f"Veins simulate_uplink failed: {resp}")
        out = {}
        for vid, r in resp["results"].items():
            out[vid] = ULResult(
                ok=bool(r["ok"]),
                t_done=float(r.get("t_done", -1.0)),
                goodput_mbps=float(r.get("goodput_mbps", 0.0)),
                rtt_ms=float(r.get("rtt_ms", 0.0)),
                reason=str(r.get("reason", "")),
            )
        return out
