"""Entornos de entrenamiento: mezcla de perfiles por episodio + VecEnv (+ normalización opcional)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from walker_lab.environment import WalkerEnv

from .relevo import RELEVO, RelevoEnv
from .tracks_extra import KINDS, ExtraTrackEnv

ENV_KEYS = ("profile", "difficulty", "length", "max_steps")


class _RewardShaping(gym.Wrapper):
    """Reescala la recompensa SOLO para el entrenamiento (nunca se usa en evaluar.py).

    Referencia: FORK (Yu Wang et al., arXiv:2010.01652) encontró que en BipedalWalkerHardcore la
    penalización de caída (-100, muy por encima de las recompensas por paso) vuelve a SAC/TD3
    demasiado conservador; reduce la caída y escala el resto para que el riesgo de intentar un
    obstáculo difícil valga la pena frente a jugar seguro. No cambia obs ni action, solo el escalar
    de recompensa que ve el algoritmo durante el entrenamiento.
    """

    def __init__(self, env: gym.Env, fall_penalty: float, scale: float):
        super().__init__(env)
        self.fall_penalty, self.scale = float(fall_penalty), float(scale)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = -self.fall_penalty if info.get("reason") == "fall" else reward * self.scale
        return obs, reward, terminated, truncated, info


def _build_track_env(spec: dict, reward_shaping: dict | None) -> gym.Env:
    if spec["profile"] in RELEVO:  # SAC + detector de agente/ conducen hasta el cambio; PPO aprende desde ahí
        env = RelevoEnv(spec["profile"], spec.get("length", 240), spec.get("max_steps", 6000),
                        d_min=spec.get("d_min", 125.0), d_max=spec.get("d_max", 150.0))
    elif spec["profile"] in KINDS:  # perfiles propios de training/tracks_extra.py (obst_var, bloques_var): pista nueva en cada reset
        env = ExtraTrackEnv(spec["profile"], spec.get("length", 60), spec.get("max_steps", 1800))
    else:
        env = WalkerEnv(**{k: spec[k] for k in ENV_KEYS if k in spec})
    if reward_shaping:
        env = _RewardShaping(env, reward_shaping["fall_penalty"], reward_shaping["scale"])
    return env


class MixedWalkerEnv(gym.Env):
    """Cada episodio sortea uno de varios WalkerEnv (perfil/dificultad/longitud) según `weight`.

    Así, aunque solo haya 2 CPUs (Colab), cada worker ve toda la mezcla del curriculum.
    """

    metadata = {"render_modes": []}

    def __init__(self, specs: list[dict], seed: int = 0, reward_shaping: dict | None = None):
        super().__init__()
        self.envs = [_build_track_env(s, reward_shaping) for s in specs]
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


def _factory(specs: list[dict], seed: int, reward_shaping: dict | None):
    def _init():
        return Monitor(MixedWalkerEnv(specs, seed, reward_shaping), info_keywords=("distance", "success"))
    return _init


def build_vec_env(specs: list[dict], n_envs: int, seed: int, normalize: bool = False, obs_rms=None,
                   reward_shaping: dict | None = None):
    fns = [_factory(specs, seed + i, reward_shaping) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
    venv.seed(seed)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
        if obs_rms is not None:
            venv.obs_rms = obs_rms
    return venv
