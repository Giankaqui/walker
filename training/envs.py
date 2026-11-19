"""Entornos de entrenamiento: mezcla de perfiles por episodio + VecEnv (+ normalización opcional)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from walker_lab.environment import WalkerEnv

from .tracks_extra import KINDS, ExtraTrackEnv

ENV_KEYS = ("profile", "difficulty", "length", "max_steps")


class MixedWalkerEnv(gym.Env):
    """Cada episodio sortea uno de varios WalkerEnv (perfil/dificultad/longitud) según `weight`.

    Así, aunque solo haya 2 CPUs (Colab), cada worker ve toda la mezcla del curriculum.
    """

    metadata = {"render_modes": []}

    def __init__(self, specs: list[dict], seed: int = 0):
        super().__init__()
        # perfiles propios de training/tracks_extra.py (obst_var, bloques_var): pista nueva en cada reset
        self.envs = [ExtraTrackEnv(s["profile"], s.get("length", 60), s.get("max_steps", 1800)) if s["profile"] in KINDS
                     else WalkerEnv(**{k: s[k] for k in ENV_KEYS if k in s}) for s in specs]
        weights = np.array([float(s.get("weight", 1.0)) for s in specs])
        self.p = weights / weights.sum()
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._rng = np.random.default_rng(seed)
        self._cur = self.envs[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:  # reproducible: siembra la mezcla y cada sub-entorno una sola vez
            self._rng = np.random.default_rng(seed)
            for i, env in enumerate(self.envs):
                env.reset(seed=int(seed) * 1000 + i)
        self._cur = self.envs[int(self._rng.choice(len(self.envs), p=self.p))]
        return self._cur.reset()

    def step(self, action):
        return self._cur.step(action)

    def close(self):
        for env in self.envs:
            env.close()


def _factory(specs: list[dict], seed: int):
    def _init():
        return Monitor(MixedWalkerEnv(specs, seed), info_keywords=("distance", "success"))
    return _init


def build_vec_env(specs: list[dict], n_envs: int, seed: int, normalize: bool = False, obs_rms=None):
    fns = [_factory(specs, seed + i) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv.seed(seed)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if obs_rms is not None:
            venv.obs_rms = obs_rms
    return venv
