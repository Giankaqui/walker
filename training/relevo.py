"""Entrenamiento "con relevo" para el agente híbrido (solo para entrenar; no cambia walker_lab ni la evaluación).

En la entrega, SAC conduce desde la salida y, cuando el detector reconoce una pista hardcore, PPO toma el control
hasta el final. Este entorno reproduce ese reparto: en cada reset genera una maratón hardcore / hardcore-plus,
deja que SAC + detector (los de agente/, misma regla W/thr que agent.py) conduzcan hasta el cambio y devuelve la
observación de ese instante. PPO aprende desde ahí, con la misma distribución de estados que verá en la carrera real
(incluido el traspaso a mitad de paso). Si SAC se cae antes del cambio, se genera otra pista.

Perfiles: relevo_hc (maratón hardcore) y relevo_hcp (hardcore-plus), con length/max_steps del YAML (240/6000).

Relevo TARDÍO (perfiles relevo_t_hc / relevo_t_hcp): para el agente con cambio tardío (agent.py con D_SW ~140 m),
SAC conduce hasta que la distancia estimada por odometría (igual que agent.py: obs[2]/6 por paso) llega a un valor
sorteado en [d_min, d_max] (por defecto 125-150 m) y PPO aprende los tramos 4-5 desde ese estado. No usa el
detector: en entrenamiento ya se sabe que la pista es hardcore.
"""
from collections import deque

import numpy as np
import torch

from walker_lab.environment import WalkerEnv

from .runtime import REPO_ROOT

RELEVO = {"relevo_hc": "hardcore", "relevo_hcp": "hardcore-plus",
          "relevo_t_hc": "hardcore", "relevo_t_hcp": "hardcore-plus", "relevo_t_n": "normal"}
_MODELS = {}


def _models(agent_dir):
    if "sac" not in _MODELS:
        torch.set_num_threads(1)
        from agente.agent import _Detector
        from agente.architecture import load_policy
        _MODELS["sac"] = load_policy(agent_dir / "policy_sac.pt", "cpu")
        blob = torch.load(agent_dir / "detector.pt", map_location="cpu", weights_only=True)
        _MODELS["det"] = _Detector(blob).eval()
        _MODELS["W"], _MODELS["thr"] = int(blob["W"]), float(blob["thr"])
    return _MODELS


class RelevoEnv(WalkerEnv):
    def __init__(self, kind, length=240, max_steps=6000, agent_dir=None, d_min=125.0, d_max=150.0):
        super().__init__(profile="maraton", length=length, max_steps=max_steps, difficulty=RELEVO[kind])
        self._agent_dir = REPO_ROOT / "agente" if agent_dir is None else agent_dir
        self.handover_distance = 0.0
        self._late = kind.startswith("relevo_t_")
        self._d = (float(d_min), float(d_max))
        self._r = np.random.default_rng()

    def _drive_sac_late(self, obs, info):
        sac = _models(self._agent_dir)["sac"]
        target, d = self._r.uniform(*self._d), 0.0
        while d < target:
            d += float(obs[2]) / 6.0
            with torch.no_grad():
                a = sac(torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0].numpy()
            obs, _, term, trunc, info = super().step(np.clip(a, -1, 1))
            if term or trunc:
                return None, info
        return obs, info

    def _drive_sac(self, obs, info):
        m = _models(self._agent_dir)
        win = deque(maxlen=m["W"])
        lags = getattr(m["det"], "lags", [0])
        hist = deque(maxlen=max(lags) + 1)
        while True:
            xo = np.asarray(obs, dtype=np.float32)
            x = torch.as_tensor(xo).unsqueeze(0)
            hist.append(xo)
            h = list(hist)
            feat = np.concatenate([h[max(len(h) - 1 - lag, 0)] for lag in lags])
            with torch.no_grad():
                win.append(float(m["det"](torch.as_tensor(feat).unsqueeze(0))[0, 0]))
                if len(win) == m["W"] and sum(win) / m["W"] > m["thr"]:
                    return obs, info
                a = m["sac"](x)[0].numpy()
            obs, _, term, trunc, info = super().step(np.clip(a, -1, 1))
            if term or trunc:
                return None, info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._r = np.random.default_rng(seed)
        for attempt in range(10):
            obs, info = super().reset(seed=seed if attempt == 0 else None, options=options)
            got, info = (self._drive_sac_late if self._late else self._drive_sac)(obs, info)
            if got is not None:
                self.handover_distance = float(info.get("distance", 0.0))
                return got, info
        return super().reset(options=options)  # muy raro: SAC cayó 10 veces seguidas antes del cambio
