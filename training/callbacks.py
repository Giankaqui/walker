"""Callbacks: progreso, evaluación periódica sobre pistas reales (métrica = distancia), checkpoints."""

from __future__ import annotations

import glob
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from walker_lab.environment import WalkerEnv

from .export import build_policy, save_agent
from .runtime import REPO_ROOT


def run_scenario(policy, scenario_path: str) -> dict:
    """Un episodio determinista; misma lógica que walker_lab.evaluation.Evaluator."""
    env = WalkerEnv(scenario=scenario_path)
    try:
        obs, info = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        return {"scenario": Path(scenario_path).stem, "distance": info["distance"], "length": info["length"],
                "success": bool(info["success"]), "steps": info["steps"], "time": float(info["simulated_seconds"]),
                "reason": info["reason"]}
    finally:
        env.close()


def resolve_suites(suites: dict) -> dict[str, list[str]]:
    out = {}
    for name, patterns in suites.items():
        if patterns is None:  # `suites: {hardcore: null}` en un YAML hijo desactiva la suite heredada
            continue
        files = sorted({f for p in patterns for f in glob.glob(str(REPO_ROOT / p))})
        if not files:
            raise FileNotFoundError(f"La suite de evaluación '{name}' no encontró pistas: {patterns}")
        out[name] = files
    return out


class Timers:
    """Acumula en qué se va el tiempo: collect (simulador + inferencia), update (gradientes), eval, ckpt."""

    def __init__(self):
        self.t = {"collect": 0.0, "update": 0.0, "eval": 0.0, "ckpt": 0.0}

    def add(self, name: str, seconds: float) -> None:
        self.t[name] += seconds

    def nested(self) -> float:  # tiempo de callbacks que ocurre DENTRO de la recolección
        return self.t["eval"] + self.t["ckpt"]


class ProgressCallback(BaseCallback):
    """Línea de progreso con desglose de tiempo + metrics.jsonl; corta al agotar `time_limit_hours`.

    El desglose usa los hooks de rollout de SB3 (sin parchear nada): el tiempo entre el fin de un rollout y el
    inicio del siguiente es la fase de actualización; el resto de la recolección es simulador + inferencia.
    """

    def __init__(self, every_steps: int, total_steps: int, time_limit_hours: float | None, stage_label,
                 timers: Timers, metrics_file: Path | None = None):
        super().__init__()
        self.every, self.total = every_steps, total_steps
        self.deadline = time.time() + time_limit_hours * 3600 if time_limit_hours else None
        self.stage_label, self.timers, self.metrics_file = stage_label, timers, metrics_file
        self.timed_out = False
        self._t0 = self._last_t = time.time()
        self._last_step = None
        self._rollout_start = self._rollout_end = None
        self._nested_at_start = 0.0
        self._last_snap, self._last_updates = dict(timers.t), 0
        from .runtime import GpuSampler
        self.gpu = GpuSampler()

    def _on_training_start(self):
        if self._last_step is None:
            self._last_step = self.num_timesteps
            self._last_updates = getattr(self.model, "_n_updates", 0)

    def _on_rollout_start(self):
        now = time.perf_counter()
        if self._rollout_end is not None:
            self.timers.add("update", now - self._rollout_end)
        self._rollout_start, self._nested_at_start = now, self.timers.nested()

    def _on_rollout_end(self):
        now = time.perf_counter()
        if self._rollout_start is not None:
            self.timers.add("collect", now - self._rollout_start - (self.timers.nested() - self._nested_at_start))
        self._rollout_end = now

    def _on_step(self) -> bool:
        if self.deadline and time.time() > self.deadline:
            self.timed_out = True
            print(">>> Límite de tiempo alcanzado: guardando y saliendo.", flush=True)
            return False
        if self.num_timesteps - self._last_step >= self.every:
            self._report()
        return True

    def _report(self) -> None:
        now = time.time()
        wall = max(now - self._last_t, 1e-9)
        steps = self.num_timesteps - self._last_step
        sps = steps / wall
        delta = {k: self.timers.t[k] - self._last_snap[k] for k in self.timers.t}
        updates = getattr(self.model, "_n_updates", 0) - self._last_updates
        buf = [e for e in self.model.ep_info_buffer if "distance" in e]
        dist = float(np.mean([e["distance"] for e in buf])) if buf else float("nan")
        succ = float(np.mean([e["success"] for e in buf])) if buf else float("nan")
        ep_len = float(np.mean([e["l"] for e in buf])) if buf else float("nan")
        self.logger.record("train/ep_distance_mean", dist)
        self.logger.record("train/ep_success_rate", succ)
        pct = {k: 100 * v / wall for k, v in delta.items()}
        ms_upd = 1000 * delta["update"] / updates if updates else float("nan")
        eta_h = (self.total - self.num_timesteps) / max(sps, 1e-9) / 3600
        keep = ("ent_coef", "critic_loss", "actor_loss", "loss", "value_loss", "policy_gradient_loss", "std")
        losses = {k.split("/")[-1]: round(float(v), 5) for k, v in self.logger.name_to_value.items()
                  if k.startswith("train/") and isinstance(v, (int, float)) and k.split("/")[-1] in keep}
        gpu_pct = self.gpu.mean_and_reset()
        gpu_txt = f" | GPU {gpu_pct:3.0f}%" if gpu_pct is not None else ""
        print(f"[{self.num_timesteps:>9,}/{self.total:,} | {sps:5.0f} sps | {self.stage_label()}] "
              f"dist={dist:6.1f} meta={succ:4.0%} | sim {pct['collect']:3.0f}% upd {pct['update']:3.0f}% "
              f"eval {pct['eval']:2.0f}% | {ms_upd:5.1f} ms/upd{gpu_txt} | ETA {eta_h:4.1f} h", flush=True)
        if self.metrics_file is not None:
            from .runtime import system_stats
            row = {"ts": round(now), "step": int(self.num_timesteps), "stage": self.stage_label(), "sps": round(sps, 1),
                   "wall_s": round(wall, 1), "pct": {k: round(v, 1) for k, v in pct.items()},
                   "ms_per_update": None if updates == 0 else round(ms_upd, 2), "updates": updates,
                   "utd": round(updates / steps, 3) if steps else None, "train_dist": dist, "train_success": succ,
                   "ep_len": ep_len, "eta_h_total": round(eta_h, 2), "losses": losses, **system_stats()}
            row["gpu_util_pct"] = gpu_pct  # promedio de la ventana (sustituye a la lectura puntual)
            try:
                with self.metrics_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, allow_nan=True) + "\n")
            except OSError:
                pass  # las métricas nunca deben tumbar el entrenamiento
        self._last_step, self._last_t = self.num_timesteps, now
        self._last_snap, self._last_updates = dict(self.timers.t), getattr(self.model, "_n_updates", 0)


def rank_key(rows: list[dict]) -> tuple:
    """Criterio del ranking del curso: más llegadas a meta; a igualdad, menor tiempo medio de las
    carreras completadas; a igualdad, mayor distancia media. Tupla comparable (mayor = mejor)."""
    done = [r for r in rows if r["success"]]
    mean_time = float(np.mean([r["time"] for r in done])) if done else 0.0
    return (len(done), -mean_time, float(np.mean([r["distance"] for r in rows])))


class DistanceEvalCallback(BaseCallback):
    """Evalúa la política determinista en suites de pistas fijas; guarda last/best (modelo y agente).

    El 'mejor' se decide con rank_key sobre las pistas de `select_on` (una suite o una lista de suites).
    Un fallo al evaluar nunca interrumpe el entrenamiento.
    """

    def __init__(self, run_dir: Path, ctx, suites: dict, select_on, every_steps: int, meta: dict,
                 timers: Timers | None = None):
        super().__init__()
        self.run_dir, self.ctx, self.every, self.meta = Path(run_dir), ctx, every_steps, meta
        self.timers = timers or Timers()
        self.suites = resolve_suites(suites)
        self.select_on = [select_on] if isinstance(select_on, str) else list(select_on)
        unknown = [s for s in self.select_on if s not in self.suites]
        if unknown:
            raise ValueError(f"eval.select_on contiene {unknown}, que no están en eval.suites {list(self.suites)}")
        self.best_file = self.run_dir / "best" / "best.json"
        self.best = self._load_best(self.best_file)
        self._last = None
        self.evaluated_at = None

    @staticmethod
    def _load_best(path: Path) -> tuple:
        if not path.is_file():
            return (-1, 0.0, 0.0)
        data = json.loads(path.read_text())
        if "key" in data:
            return tuple(data["key"])
        return (0, 0.0, float(data.get("score", 0.0)))  # formato antiguo (solo distancia): permite reanudar runs previos

    def _on_training_start(self):
        if self._last is None:
            self._last = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            self.safe_evaluate()
        return True

    def safe_evaluate(self):
        t0 = time.perf_counter()
        try:
            return self.evaluate()
        except Exception:
            traceback.print_exc()
            print("    ⚠ La evaluación falló; el entrenamiento continúa.", flush=True)
            return None
        finally:
            self.timers.add("eval", time.perf_counter() - t0)

    def evaluate(self) -> tuple:
        self._last = self.evaluated_at = self.num_timesteps
        t0 = time.time()
        policy = build_policy(self.model, self.ctx.vn)  # la misma MLPPolicy que se entrega
        summary, rows = {}, {}
        for suite, files in self.suites.items():
            rows[suite] = [run_scenario(policy, f) for f in files]
            done = [r for r in rows[suite] if r["success"]]
            summary[suite] = {"n": len(rows[suite]), "arrivals": len(done),
                              "mean_time_arrived": float(np.mean([r["time"] for r in done])) if done else None,
                              "mean_distance": float(np.mean([r["distance"] for r in rows[suite]]))}
            for key, value in (("arrivals", len(done)), ("mean_distance", summary[suite]["mean_distance"])):
                self.logger.record(f"eval/{suite}_{key}", value)
        key = rank_key([r for s in self.select_on for r in rows[s]])
        step = int(self.num_timesteps)
        is_best = key > self.best
        info = {"step": step, "rank_key": list(key), "select_on": self.select_on, "summary": summary, **self.meta}
        save_agent(policy, self.run_dir / "last_agent", info)
        if is_best:
            self.best = key
            self.model.save(self.run_dir / "best" / "model.zip")
            if self.ctx.vn is not None:
                (self.run_dir / "best" / "obs_rms.pkl").write_bytes(pickle.dumps(self.ctx.vn.obs_rms))
            save_agent(policy, self.run_dir / "best_agent", info)
            self.best_file.write_text(json.dumps({"step": step, "key": list(key)}))
        with (self.run_dir / "eval_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "rank_key": list(key), "summary": summary, "rows": rows, "best": is_best}) + "\n")
        parts = " | ".join(
            f"{s}: meta {v['arrivals']}/{v['n']}"
            + (f" t={v['mean_time_arrived']:.0f}s" if v["mean_time_arrived"] is not None else "")
            + f" dist={v['mean_distance']:.1f}" for s, v in summary.items())
        print(f"    EVAL @ {step:,}  {parts}  {'<< NUEVO MEJOR' if is_best else ''} ({time.time() - t0:.0f}s)", flush=True)
        return key


class CheckpointCallback(BaseCallback):
    def __init__(self, manager, ctx, every_steps: int, timers: Timers | None = None):
        super().__init__()
        self.manager, self.ctx, self.every = manager, ctx, every_steps
        self.timers = timers or Timers()
        self._last = None

    def _on_training_start(self):
        if self._last is None:
            self._last = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last >= self.every:
            t0 = time.perf_counter()
            self.manager.try_save(self.model, self.ctx.vn)  # un fallo de disco/Drive no mata el entrenamiento
            self.timers.add("ckpt", time.perf_counter() - t0)
            self._last = self.num_timesteps
        return True
