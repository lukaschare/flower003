#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Flower adapter layer.

For now: MockFlwrAdapter to run end-to-end immediately.
Later: replace with real Flower/Docker integration.

Contract:
train_clients(round_idx, client_ids, global_model, ci_g_per_kwh) -> Dict[client_id, TrainResult]
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrainResult:
    t_train_s: float
    e_comp_j: float
    co2_comp_g: float
    update_vec: List[float]


class MockFlwrAdapter:
    """
    Mock training:
    - t_train depends on client id and round (random but stable with seed)
    - update is small random vector
    - compute energy: P_comp * t_train
    - compute carbon: E(kWh) * CI
    """

    def __init__(self, seed: int = 42, p_comp_w: float = 18.0) -> None:
        self.seed = seed
        self.p_comp_w = float(p_comp_w)  # average compute power (W) for mock

    @staticmethod
    def _joule_to_kwh(j: float) -> float:
        return j / 3_600_000.0

    def train_clients(
        self,
        round_idx: int,
        client_ids: List[str],
        global_model: List[float],
        ci_g_per_kwh: float,
    ) -> Dict[str, TrainResult]:
        out: Dict[str, TrainResult] = {}

        for cid in client_ids:
            # deterministic randomness per (seed, round, client)
            rnd = random.Random(hash((self.seed, round_idx, cid)) & 0xFFFFFFFF)

            # training time: 2~8 seconds (mock)
            t_train = rnd.uniform(2.0, 8.0)

            # compute energy
            e_comp_j = self.p_comp_w * t_train

            # compute carbon
            co2_comp_g = self._joule_to_kwh(e_comp_j) * float(ci_g_per_kwh)

            # toy update vector: small perturbation
            dim = len(global_model)
            update = [rnd.uniform(-0.05, 0.05) for _ in range(dim)]

            out[cid] = TrainResult(
                t_train_s=t_train,
                e_comp_j=e_comp_j,
                co2_comp_g=co2_comp_g,
                update_vec=update,
            )

        return out
