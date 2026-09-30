"""Frozen local RAFT flow, implemented through the installed torchvision API.

No model download path is called here. Checkpoints must be packaged beforehand.
"""
from pathlib import Path
import os

import numpy as np
import torch


class LocalRaft:
    def __init__(self, checkpoint: str | Path, architecture="raft_large", width=384,
                 updates=8, every_n_frames=3, device=None):
        from torchvision.models.optical_flow import raft_large, raft_small
        if architecture not in ("raft_large", "raft_small"):
            raise ValueError(f"Unknown RAFT architecture: {architecture}")
        if device is None:
            device = os.environ.get("BLACKBOX_DEVICE")
            if device in (None, "", "auto"):
                device = None
        if device is None:
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = torch.device(device)
        self.width = int(width)
        self.every_n_frames = max(1, int(every_n_frames))
        self.updates = int(updates)
        builder = raft_large if architecture == "raft_large" else raft_small
        self.model = builder(weights=None, progress=False)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        if self.device.type == "cpu":
            torch.set_num_threads(2)

    def __call__(self, previous, current):
        # Input images are uint8 RGB; torchvision RAFT expects [-1, 1].
        pair = np.stack((previous, current))
        tensor = torch.from_numpy(pair).permute(0, 3, 1, 2).to(self.device, dtype=torch.float32)
        tensor = tensor/127.5-1.
        with torch.inference_mode():
            flow = self.model(tensor[:1], tensor[1:], num_flow_updates=self.updates)[-1]
        return flow[0].permute(1, 2, 0).cpu().numpy()
