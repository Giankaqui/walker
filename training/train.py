"""Entrenamiento orquestado por configuración YAML. Reanuda solo si existe un checkpoint.

    python -m training.train --config configs/sac_curriculum.yaml
    python -m training.train --config configs/smoke.yaml --set name=prueba --fresh
"""

from __future__ import annotations

import argparse
import json
import pickle
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from torch import nn

from .callbacks import CheckpointCallback, DistanceEvalCallback, ProgressCallback, Timers
from .checkpoints import CheckpointManager
from .config import load_config
from .envs import build_vec_env
from .runtime import REPO_ROOT, default_n_envs, environment_info, git_info, persistence_warning, runs_root

ALGOS = {"SAC": SAC, "TD3": TD3, "PPO": PPO}
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}


def stage_ends(stages: list[dict]) -> list[int]:
    ends, total = [], 0
    for s in stages:
        total += int(s["timesteps"])
        ends.append(total)
    return ends


def _init_dir(cfg: dict) -> Path:
    src = Path(cfg["init_from"])
    return src if src.is_absolute() else runs_root() / src


def build_model(cfg: dict, venv, device: str, n_envs: int):
    cls = ALGOS[cfg["algo"]]
    kwargs = dict(cfg.get("algo_kwargs") or {})
    policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
    if cfg.get("utd") is not None and cfg["algo"] in ("SAC", "TD3"):  # actualizaciones por transición, sin depender de n_envs
        kwargs["gradient_steps"] = max(1, round(float(cfg["utd"]) * int(kwargs.get("train_freq", 1)) * n_envs))
    if isinstance(policy_kwargs.get("activation_fn"), str):
        policy_kwargs["activation_fn"] = ACTIVATIONS[policy_kwargs["activation_fn"]]
    if cfg.get("init_from"):  # arranque en caliente desde el mejor modelo de otro experimento
        src = _init_dir(cfg)
        print(f"Inicializando desde {src / 'best' / 'model.zip'}", flush=True)
        # SB3 conserva los hiperparámetros guardados en el zip: se sobrescriben los que tiene sentido cambiar al afinar.
        custom = {k: kwargs[k] for k in ("learning_rate", "ent_coef", "target_kl", "clip_range", "n_epochs") if k in kwargs}
        custom["seed"] = cfg["seed"]  # si no, _setup_model resiembra el entorno con la semilla guardada en el zip
        print(f"  hiperparámetros sobrescritos: {custom}", flush=True)
        return cls.load(src / "best" / "model.zip", env=venv, device=device, custom_objects=custom)
    return cls("MlpPolicy", venv, seed=cfg["seed"], device=device, verbose=0, policy_kwargs=policy_kwargs, **kwargs)


def archive_run(run_dir: Path) -> None:
    """--fresh: aparta (renombra, sin copiar ni borrar) el run anterior para no perder horas de entrenamiento."""
    if any((run_dir / sub).exists() for sub in ("ckpt", "best")):
        dest = run_dir.with_name(f"{run_dir.name}.old-{time.strftime('%Y%m%d-%H%M%S')}")
        run_dir.rename(dest)
        print(f"Run anterior archivado en {dest} (bórrelo a mano cuando no lo necesite).", flush=True)


def restore(manager: CheckpointManager, algo: str) -> dict | None:
    """Primer checkpoint legible, del más reciente al más antiguo. Si hay checkpoints pero ninguno carga, falla alto."""
    cands = manager.candidates()
    for cand in cands:
        try:
            ALGOS[algo].load(cand["model"], device="cpu")  # sonda de integridad
            if cand["obs_rms_path"] is not None:
                cand["obs_rms"] = pickle.loads(cand["obs_rms_path"].read_bytes())
            else:
                cand["obs_rms"] = None
            return cand
        except Exception as exc:
            print(f"⚠ Checkpoint {cand['step']:,} ilegible ({exc!r}); pruebo el anterior.", flush=True)
    if cands:
        raise RuntimeError("Hay checkpoints pero ninguno se puede cargar. Revise runs/…/ckpt, o use --fresh "
                           "(archiva el run viejo y empieza de cero).")
    return None


def run(cfg: dict, run_dir: Path, fresh: bool) -> None:
    if fresh:
        archive_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(cfg["seed"])
    # GPU solo ayuda a los algoritmos off-policy (SAC/TD3); PPO con MLP pequeño va más rápido en CPU.
    use_gpu = torch.cuda.is_available() and cfg["algo"] in ("SAC", "TD3")
    device = ("cuda" if use_gpu else "cpu") if cfg.get("device", "auto") == "auto" else cfg["device"]
    n_envs = default_n_envs() if cfg.get("n_envs", "auto") == "auto" else int(cfg["n_envs"])
    stages, ends = cfg["stages"], stage_ends(cfg["stages"])
    normalize = bool(cfg.get("normalize_obs", False))
    ck_cfg, ev_cfg = cfg.get("checkpoint", {}), cfg["eval"]
    manager = CheckpointManager(run_dir, ck_cfg.get("keep", 2), ck_cfg.get("save_replay_buffer", True))

    state = restore(manager, cfg["algo"])
    start = state["step"] if state else 0
    if state and start >= ends[-1]:
        print(f"El run ya terminó ({start:,} pasos). Use --fresh o un nombre nuevo.", flush=True)
        return
    stage_idx = next(i for i, e in enumerate(ends) if e > start)
    meta = {"name": cfg["name"], "algo": cfg["algo"], "git": git_info(), "env": environment_info()}
    (run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    warning = persistence_warning(run_dir)
    if warning:
        print(warning, flush=True)
    print(f"RUN {cfg['name']} | {cfg['algo']} | device={device} | n_envs={n_envs} | "
          f"{'RESUME desde ' + format(start, ',') if state else 'desde cero'} | dir={run_dir}", flush=True)

    ctx = SimpleNamespace(vn=None)
    obs_rms = state["obs_rms"] if state else None
    if normalize and not state and cfg.get("init_from"):  # heredar normalización del run de origen
        obs_rms = pickle.loads((_init_dir(cfg) / "best" / "obs_rms.pkl").read_bytes())
    venv = ctx.vn = build_vec_env(stages[stage_idx]["envs"], n_envs, cfg["seed"], normalize, obs_rms,
                                   reward_shaping=cfg.get("reward_shaping"))
    if not normalize:
        ctx.vn = None
    if state:
        model = ALGOS[cfg["algo"]].load(state["model"], env=venv, device=device)
        if state["replay"] is not None and hasattr(model, "load_replay_buffer"):
            try:
                model.load_replay_buffer(state["replay"])
            except Exception as exc:  # buffer corrupto: se continúa con uno vacío en vez de abortar
                print(f"⚠ Replay buffer ilegible ({exc!r}); continúo con el buffer vacío.", flush=True)
    else:
        model = build_model(cfg, venv, device, n_envs)
    formats = ["csv"] + (["tensorboard"] if _has_tensorboard() else [])
    model.set_logger(configure(str(run_dir / "logs"), formats))

    current = {"i": stage_idx}
    label = lambda: f"{current['i'] + 1}/{len(stages)} {stages[current['i']].get('name', '')}"
    timers = Timers()
    progress = ProgressCallback(ev_cfg.get("progress_every", 10_000), ends[-1], cfg.get("time_limit_hours"), label,
                                timers, run_dir / "metrics.jsonl")
    evaluator = DistanceEvalCallback(run_dir, ctx, ev_cfg["suites"], ev_cfg["select_on"], ev_cfg["every_steps"], meta, timers)
    callbacks = CallbackList([progress, evaluator,
                              CheckpointCallback(manager, ctx, ck_cfg.get("every_steps", 100_000), timers)])

    try:
        for i in range(stage_idx, len(stages)):
            current["i"] = i
            if i > stage_idx:  # nuevo stage: nuevo VecEnv, mismas estadísticas de normalización
                rms = ctx.vn.obs_rms if ctx.vn is not None else None
                venv.close()
                venv = build_vec_env(stages[i]["envs"], n_envs, cfg["seed"] + 1000 * i, normalize, rms,
                                      reward_shaping=cfg.get("reward_shaping"))
                ctx.vn = venv if normalize else None
                model.set_env(venv)
            print(f"=== Stage {i + 1}/{len(stages)} '{stages[i].get('name', '')}' hasta {ends[i]:,} pasos ===", flush=True)
            model.learn(ends[i] - model.num_timesteps, callback=callbacks, reset_num_timesteps=False,
                        log_interval=ev_cfg.get("log_interval", 4))
            manager.try_save(model, ctx.vn)
            if progress.timed_out:
                break
    except KeyboardInterrupt:
        print(">>> Interrumpido: guardando checkpoint.", flush=True)
        manager.try_save(model, ctx.vn)
    finally:
        if evaluator.evaluated_at != model.num_timesteps:  # evaluación final, salvo que ya se hizo en este paso
            evaluator.safe_evaluate()
        venv.close()
    best = json.loads(evaluator.best_file.read_text()) if evaluator.best_file.is_file() else {}
    print(f"FIN. Mejor: {best}", flush=True)
    print(f"RUN_DIR={run_dir}", flush=True)


def _has_tensorboard() -> bool:
    try:
        import tensorboard  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenar al bípedo según un YAML.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--fresh", action="store_true", help="Ignorar checkpoints existentes y empezar de cero.")
    args = parser.parse_args()
    # Un SIGTERM (cierre de sesión, `kill`) se trata como Ctrl+C: guarda checkpoint antes de salir.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    config_path = args.config if args.config.is_absolute() else (Path.cwd() / args.config)
    if not config_path.exists():
        config_path = REPO_ROOT / args.config
    cfg = load_config(config_path, args.overrides)
    run(cfg, runs_root() / cfg["name"], args.fresh)


if __name__ == "__main__":
    main()
