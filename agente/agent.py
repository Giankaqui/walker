"""Adaptador de entrega: carga `policy.pt` (MLP + normalización) y la ejecuta.

`policy.pt` lo genera `python -m training.export runs/<exp> --to agente`; este archivo no cambia entre experimentos.
"""

import numpy as np
import torch

from walker_lab.agent import BaseAgent

from .architecture import load_policy


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        weights = self.model_dir / "policy.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Falta {weights}. Genérelo con: python -m training.export runs/<exp> --to agente")
        self.policy = load_policy(weights, self.device)

    def reset(self):
        pass  # política sin memoria: no hay estado que reiniciar

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(observation, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy(obs)[0].cpu().numpy()
        return np.clip(np.nan_to_num(action), -1.0, 1.0).astype(np.float32)
