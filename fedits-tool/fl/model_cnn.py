# fl/model_cnn.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import numpy as np

class SimpleCifarCNN(nn.Module):
    """Tiny CNN for CIFAR-10 (3x32x32)."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))   # 64x8x8
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_ndarrays(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]

def set_ndarrays(model: nn.Module, params: List[np.ndarray]) -> None:
    sd = model.state_dict()
    keys = list(sd.keys())
    assert len(keys) == len(params), f"param len mismatch: {len(keys)} vs {len(params)}"
    new_sd = {}
    for k, v in zip(keys, params):
        new_sd[k] = torch.tensor(v)
    model.load_state_dict(new_sd, strict=True)

def l2_norm(params: List[np.ndarray]) -> float:
    s = 0.0
    for a in params:
        s += float((a.astype(np.float64) ** 2).sum())
    return float(s ** 0.5)
