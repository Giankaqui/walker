"""Agente híbrido de entrega: SAC + detector de pista difícil + PPO.

- `policy_sac.pt`: SAC (`sac_max`), el más rápido y casi perfecto en pistas normales, pero cae en obstáculos hardcore.
- `policy_ppo.pt`: PPO (soup de ramas `x_all3`), el más robusto en pistas hardcore / hardcore-plus.
- `detector.pt`: MLP pequeña (24-64-64-1) entrenada con carreras de SAC en pistas de entrenamiento (semillas
  distintas de las de evaluación) que da, para cada observación, el log-odds de "esta pista es hardcore".

La carrera empieza con SAC. Si la media del log-odds en los últimos `W` pasos supera `thr` (algo que en las pistas
normales casi nunca pasa), el agente cambia a PPO y sigue con PPO hasta el final (cambio en un solo sentido).
Anti-atasco (igual que la entrega anterior): si la velocidad horizontal es ~0 durante 100 pasos, se suma ruido
N(0, 0.3) a las acciones durante 30 pasos; el generador se reinicia con semilla fija en reset() (determinista).
Todo se carga con torch.load(weights_only=True); solo depende de torch y numpy.
"""

from collections import deque

import numpy as np
import torch
from torch import nn

from walker_lab.agent import BaseAgent

from .architecture import load_policy

STALL_WINDOW = 100
STALL_SPEED = 0.02
NOISE_STEPS = 30
NOISE_SIGMA = 0.3
NOISE_SEED = 0


def _to_tensor(x, device):
    try:
        return torch.as_tensor(x, device=device)
    except (RuntimeError, TypeError):  # torch antiguo sin puente con NumPy 2.x
        return torch.tensor(x.tolist(), dtype=torch.float32, device=device)


def _to_numpy(t):
    try:
        return t.numpy()
    except (RuntimeError, TypeError):
        return np.array(t.tolist(), dtype=np.float32)


class _Detector(nn.Module):
    def __init__(self, blob):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.net.load_state_dict(blob["state"])
        self.register_buffer("mu", blob["mu"].float())
        self.register_buffer("sd", blob["sd"].float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        for f in ("policy_sac.pt", "policy_ppo.pt", "detector.pt"):
            if not (self.model_dir / f).is_file():
                raise FileNotFoundError(f"Falta {self.model_dir / f}")
        self.sac = load_policy(self.model_dir / "policy_sac.pt", self.device)
        self.ppo = load_policy(self.model_dir / "policy_ppo.pt", self.device)
        blob = torch.load(self.model_dir / "detector.pt", map_location=self.device, weights_only=True)
        self.window, self.thr = int(blob["W"]), float(blob["thr"])
        self.detector = _Detector(blob).to(self.device).eval()
        self.reset()

    def reset(self):
        self._logits = deque(maxlen=self.window)
        self.switched = False
        self._speeds = deque(maxlen=STALL_WINDOW)
        self._noise_left = 0
        self._rng = np.random.default_rng(NOISE_SEED)

    def act(self, observation: np.ndarray) -> np.ndarray:
        x = np.array(observation, dtype=np.float32).reshape(-1)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        xt = _to_tensor(x, self.device).unsqueeze(0)
        with torch.no_grad():
            if not self.switched:
                self._logits.append(float(self.detector(xt)[0, 0]))
                if len(self._logits) == self.window and sum(self._logits) / self.window > self.thr:
                    self.switched = True
            out = (self.ppo if self.switched else self.sac)(xt)[0].cpu()
        action = np.nan_to_num(_to_numpy(out), nan=0.0)

        self._speeds.append(abs(float(x[2])))
        if self._noise_left > 0:
            self._noise_left -= 1
            action = action + self._rng.normal(0.0, NOISE_SIGMA, 4)
        elif len(self._speeds) == STALL_WINDOW and float(np.mean(self._speeds)) < STALL_SPEED:
            self._noise_left = NOISE_STEPS
            self._speeds.clear()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
