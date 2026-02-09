#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import urllib.request
from typing import Any, Dict


class OrchestratorHttpClient:
    def __init__(self, base_url: str, timeout_s: float = 30.0) -> None:
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
