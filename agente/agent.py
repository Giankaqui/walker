"""Adaptador de entrega: carga `policy.pt` (MLP + normalización de observaciones) y la ejecuta.

`policy.pt` contiene una MLP 24-256-256-4 exportada desde una política PPO de stable-baselines3 (promedio de
pesos de varios checkpoints, "model soup"), con la normalización de observaciones incluida. Solo depende de torch.

Anti-atasco: en tramos difíciles la política a veces se queda quieta frente a un hueco o escalón (velocidad
horizontal ~0, ambos pies apoyados) hasta agotar el tiempo. Si la velocidad horizontal observada se mantiene
~0 durante 2 s, se suma un ruido gaussiano pequeño a las acciones durante 0.6 s para sacarla de ese equilibrio.
El generador de ruido se reinicia con semilla fija en reset(), así que cada carrera es determinista.
"""

from collections import deque

import numpy as np
import torch

from walker_lab.agent import BaseAgent

from .architecture import load_policy

STALL_WINDOW = 100      # pasos (2 s a 50 FPS) con velocidad ~0 para considerar que está atascado
STALL_SPEED = 0.02      # umbral sobre |obs[2]| (velocidad horizontal en la escala del entorno)
NOISE_STEPS = 30        # pasos con ruido tras detectar el atasco
NOISE_SIGMA = 0.3       # desviación del ruido sumado a las acciones
NOISE_SEED = 0


def _to_tensor(x, device):
    try:
        return torch.as_tensor(x, device=device)
    except (RuntimeError, TypeError):  # torch antiguo sin puente con NumPy 2.x
        return torch.tensor(x.tolist(), dtype=torch.float32, device=device)


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        weights = self.model_dir / "policy.pt"
        if not weights.is_file():
            raise FileNotFoundError(f"Falta {weights}: la carpeta agente/ debe incluir policy.pt")
        self.policy = load_policy(weights, self.device)
        self.reset()

    def reset(self):
        self._speeds = deque(maxlen=STALL_WINDOW)
        self._noise_left = 0
        self._rng = np.random.default_rng(NOISE_SEED)

    def act(self, observation: np.ndarray) -> np.ndarray:
        x = np.array(observation, dtype=np.float32).reshape(-1)  # copia escribible, forma (24,)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        with torch.no_grad():
            out = self.policy(_to_tensor(x, self.device).unsqueeze(0))[0].cpu()
        try:
            action = out.numpy()
        except (RuntimeError, TypeError):
            action = np.array(out.tolist(), dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0)

        self._speeds.append(abs(float(x[2])))
        if self._noise_left > 0:
            self._noise_left -= 1
            action = action + self._rng.normal(0.0, NOISE_SIGMA, 4)
        elif len(self._speeds) == STALL_WINDOW and float(np.mean(self._speeds)) < STALL_SPEED:
            self._noise_left = NOISE_STEPS
            self._speeds.clear()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
