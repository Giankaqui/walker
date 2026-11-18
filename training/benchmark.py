"""Calibración de velocidad en la máquina REAL (Colab/Kaggle): mide pasos/s y en qué se va el tiempo
para varias configuraciones de SAC. ~3-5 min. No mide calidad de aprendizaje, solo velocidad.

    python -m training.benchmark                      # usa configs/sac_max.yaml
    python -m training.benchmark --steps 6000 --only 1,3
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch

from .callbacks import ProgressCallback, Timers
from .config import load_config
from .envs import build_vec_env
from .runtime import REPO_ROOT, GpuSampler, default_n_envs, environment_info
from .train import build_model

# (nombre, utd, batch, net_arch, train_freq): UTD = actualizaciones de gradiente por transición recogida
VARIANTS = [
    ("1 actual: UTD 1, b512, 256x256", 1.0, 512, [256, 256], 1),
    ("2 UTD 1, b512, ráfagas de 8", 1.0, 512, [256, 256], 8),
    ("3 UTD 0.5, b1024", 0.5, 1024, [256, 256], 8),
    ("4 UTD 0.25, b2048", 0.25, 2048, [256, 256], 8),
    ("5 UTD 0.25, b2048, 400x300", 0.25, 2048, [400, 300], 8),
    ("6 UTD 0.125, b4096", 0.125, 4096, [256, 256], 8),
]


def run_variant(cfg: dict, name: str, utd: float, batch: int, net: list[int], train_freq: int, steps: int, device: str):
    cfg = copy.deepcopy(cfg)
    n_envs = default_n_envs() if cfg.get("n_envs", "auto") == "auto" else int(cfg["n_envs"])
    cfg["utd"] = utd
    kw = cfg["algo_kwargs"]
    kw.update(batch_size=batch, train_freq=train_freq, learning_starts=2000, buffer_size=200_000)
    kw["policy_kwargs"] = {"net_arch": net}
    specs = cfg["stages"][min(2, len(cfg["stages"]) - 1)]["envs"]
    venv = build_vec_env(specs, n_envs, cfg["seed"], normalize=False)
    try:
        model = build_model(cfg, venv, device, n_envs)
        model.learn(2500)  # calentamiento: llena el buffer (sin actualizaciones)
        timers, upd0 = Timers(), getattr(model, "_n_updates", 0)
        sampler = GpuSampler(interval=0.5)
        sampler.mean_and_reset()
        cb = ProgressCallback(10**12, 10**12, None, lambda: "bench", timers)
        t0 = time.perf_counter()
        model.learn(steps, callback=cb, reset_num_timesteps=False)
        wall = time.perf_counter() - t0
        gpu = sampler.mean_and_reset()  # media durante el entrenamiento, no después
        n_upd = getattr(model, "_n_updates", 0) - upd0
        return {"name": name, "sps": steps / wall, "ms_upd": 1000 * timers.t["update"] / max(n_upd, 1),
                "sim": 100 * timers.t["collect"] / wall, "upd": 100 * timers.t["update"] / wall,
                "gpu": gpu, "n_envs": n_envs}
    finally:
        venv.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrar velocidad de SAC en esta máquina.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "sac_max.yaml")
    parser.add_argument("--steps", type=int, default=4000, help="Pasos medidos por variante.")
    parser.add_argument("--only", default="", help="Números de variante separados por coma (por defecto, todas).")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Entorno: {environment_info()}\nDispositivo: {device}\n", flush=True)
    chosen = {int(x) for x in args.only.split(",") if x} or set(range(1, len(VARIANTS) + 1))
    results = []
    for i, variant in enumerate(VARIANTS, 1):
        if i in chosen:
            r = run_variant(cfg, *variant, steps=args.steps, device=device)
            results.append(r)
            print(f"  {r['name']:34s} {r['sps']:6.0f} sps | {r['ms_upd']:5.1f} ms/upd | sim {r['sim']:3.0f}% "
                  f"upd {r['upd']:3.0f}% | GPU {'-' if r['gpu'] is None else format(r['gpu'], '.0f')}%", flush=True)
    base = results[0]["sps"]
    print(f"\n{'variante':34s} {'sps':>6s} {'x vs 1ª':>8s} {'h por 10M pasos':>16s}")
    for r in results:
        print(f"{r['name']:34s} {r['sps']:6.0f} {r['sps'] / base:7.1f}x {10e6 / r['sps'] / 3600:16.1f}")
    print(f"\nn_envs={results[0]['n_envs']}. Más sps NO implica mejor resultado: con UTD bajo se aprende menos por muestra.")


if __name__ == "__main__":
    main()
