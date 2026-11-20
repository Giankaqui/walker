"""Agente híbrido de entrega: SAC + detector de pista difícil + PPO.

- `policy_sac.pt`: SAC (`sac_max`), el más rápido y casi perfecto en pistas normales, pero cae en obstáculos hardcore.
- `policy_ppo.pt`: PPO (soup de ramas `x_all3`), el más robusto en pistas hardcore / hardcore-plus.
- `detector.pt`: MLP pequeña entrenada con carreras de SAC en pistas de entrenamiento (semillas distintas de las de
  evaluación) que da, en cada paso, el log-odds de "esta pista es hardcore". Su entrada es la observación actual y,
  si `detector.pt` trae `lags`, también las de hace k pasos (al inicio de la carrera se repite la primera).

La carrera empieza con SAC. Si la media del log-odds en los últimos `W` pasos supera `thr` (algo que en las pistas
normales casi nunca pasa), la pista queda marcada como difícil. El cambio a PPO (en un solo sentido) ocurre cuando,
además, la distancia recorrida estimada por odometría (vel_x * dt, con obs[2] = 0.12 vel_x y dt = 1/50) llega a
`D_SW`: SAC es mejor que PPO en los tramos 1-3 incluso en pistas hardcore, y PPO en los tramos 4-5. Opciones en
detector.pt: `DET_MAX` (ignora detecciones más allá de esa distancia), `FLAT` (tras D_SW espera suelo llano según
el lidar, con un máximo de FALLBACK m), `DELAY` (pasos a esperar tras la detección), `N_SW` (si a esa distancia la pista no fue marcada como difícil, pasa a `policy_ppo_n.pt` —o a PPO si no existe— para los tramos finales de una pista normal), `STALL2` (si no avanza 0.3 m en STALL2 pasos, conduce la otra política
STALL2_LEN pasos). Sin esas claves se comporta como el cambio inmediato original.
Anti-atasco (igual que la entrega anterior): si la velocidad horizontal es ~0 durante 100 pasos, se suma ruido
N(0, 0.3) a las acciones durante 30 pasos; el generador se reinicia con semilla fija en reset() (determinista).
Todo se carga con torch.load(weights_only=True); solo depende de torch y numpy.
"""

import math
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
        st = blob["state"]
        n = len([k for k in st if k.endswith(".weight")])
        layers = []
        for i in range(n):
            w = st[f"{2 * i}.weight"]
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            if i < n - 1:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.net.load_state_dict(st)
        self.lags = [int(v) for v in blob.get("lags", [0])]
        self.register_buffer("mu", blob["mu"].float())
        self.register_buffer("sd", blob["sd"].float())

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


LIDAR_L = 160 / 30.0
_COS = np.cos(1.5 * np.arange(10) / 10)
_DEFAULTS = dict(N_SW=math.inf, DELAY=0, D_SW=-math.inf, DET_MAX=math.inf, FLAT=False, FLAT_K=5, FLAT_TOL=0.06, FALLBACK=12.0, STALL2=0, STALL2_LEN=150)


def _lidar_flat(x, tol):
    f = x[14:24]
    if f[0] >= 0.99:
        return False
    pred = f[0] * LIDAR_L / (_COS * LIDAR_L)
    m = pred < 0.97
    return bool(np.all(np.abs(f[m] - pred[m]) < tol) and np.all(f[~m] > 0.97))


class Agent(BaseAgent):
    def __init__(self, model_dir, device="cpu"):
        super().__init__(model_dir, device)
        for f in ("policy_sac.pt", "policy_ppo.pt", "detector.pt"):
            if not (self.model_dir / f).is_file():
                raise FileNotFoundError(f"Falta {self.model_dir / f}")
        self.sac = load_policy(self.model_dir / "policy_sac.pt", self.device)
        self.ppo = load_policy(self.model_dir / "policy_ppo.pt", self.device)
        pn = self.model_dir / "policy_ppo_n.pt"
        self.ppo_n = load_policy(pn, self.device) if pn.is_file() else self.ppo
        blob = torch.load(self.model_dir / "detector.pt", map_location=self.device, weights_only=True)
        self.window, self.thr = int(blob["W"]), float(blob["thr"])
        self.cfg = {k: type(v)(blob[k]) if k in blob else v for k, v in _DEFAULTS.items()}
        self.detector = _Detector(blob).to(self.device).eval()
        self._hist_len = max(self.detector.lags) + 1
        self.reset()

    def reset(self):
        self._logits = deque(maxlen=self.window)
        self._hist = deque(maxlen=self._hist_len)
        self.detected = False
        self._since_det = 0
        self._det_d = 0.0
        self.switched = False
        self.normal_switched = False
        self.distance = 0.0
        self._flat = 0
        self._best, self._since, self._other = 0.0, 0, 0
        self._speeds = deque(maxlen=STALL_WINDOW)
        self._noise_left = 0
        self._rng = np.random.default_rng(NOISE_SEED)

    def act(self, observation: np.ndarray) -> np.ndarray:
        c = self.cfg
        x = np.array(observation, dtype=np.float32).reshape(-1)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        v = float(x[2])
        if abs(v) < 10.0:  # odometría: solo velocidades plausibles (obs[2] = 0.12 vel_x, dt = 1/50)
            self.distance += v / 6.0
        xt = _to_tensor(x, self.device).unsqueeze(0)
        with torch.no_grad():
            if not self.detected and self.distance <= c["DET_MAX"]:
                self._hist.append(x)
                h = list(self._hist)
                feat = np.concatenate([h[max(len(h) - 1 - lag, 0)] for lag in self.detector.lags])
                self._logits.append(float(self.detector(_to_tensor(feat, self.device).unsqueeze(0))[0, 0]))
                if len(self._logits) == self.window and sum(self._logits) / self.window > self.thr:
                    self.detected, self._det_d = True, self.distance
            if self.detected and not self.switched:
                self._since_det += 1
            if self.detected and not self.switched and self.distance >= c["D_SW"] and self._since_det > c["DELAY"]:
                if c["FLAT"]:
                    self._flat = self._flat + 1 if _lidar_flat(x, c["FLAT_TOL"]) else 0
                    self.switched = self._flat >= c["FLAT_K"] or self.distance >= max(c["D_SW"], self._det_d) + c["FALLBACK"]
                else:
                    self.switched = True
            if not self.detected and not self.normal_switched and self.distance >= c["N_SW"]:
                self.normal_switched = True
            use_ppo = self.switched
            if c["STALL2"]:
                if self.distance > self._best + 0.3:
                    self._best, self._since = self.distance, 0
                else:
                    self._since += 1
                if self._since >= c["STALL2"] and self._other == 0:
                    self._other, self._since = c["STALL2_LEN"], 0
                if self._other > 0:
                    self._other -= 1
                    use_ppo = not use_ppo
            if use_ppo:
                pol = self.ppo
            elif self.normal_switched and not (c["STALL2"] and self._other > 0):
                pol = self.ppo_n
            else:
                pol = self.sac
            out = pol(xt)[0].cpu()
        action = np.nan_to_num(_to_numpy(out), nan=0.0)

        self._speeds.append(abs(float(x[2])))
        if self._noise_left > 0:
            self._noise_left -= 1
            action = action + self._rng.normal(0.0, NOISE_SIGMA, 4)
        elif len(self._speeds) == STALL_WINDOW and float(np.mean(self._speeds)) < STALL_SPEED:
            self._noise_left = NOISE_STEPS
            self._speeds.clear()
        return np.clip(action, -1.0, 1.0).astype(np.float32)
