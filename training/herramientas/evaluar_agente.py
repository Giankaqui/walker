"""Evalúa una carpeta agente/ COMPLETA (su agent.py real: híbrido, anti-atasco, etc.) igual que el Evaluator oficial
(env.reset(seed) -> agent.reset() -> agent.act(obs) por paso) en los sets de pistas nuevas de holdout.py.

    python -m training.herramientas.evaluar_agente <dir_con_agente> <nombre> --set test5 test6 ... [--procs 8]
    (p. ej. <dir_con_agente> = . para el agente/ del repo; resultados en runs/evaluar_agente/holdout_<set>/)
"""
import argparse
import json
import sys
from multiprocessing import get_context
from pathlib import Path

from training.herramientas import holdout
from training.runtime import runs_root

_A = {}


def _init(pkg):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, pkg)
    for m in [m for m in sys.modules if m == "agente" or m.startswith("agente.")]:
        del sys.modules[m]
    from agente.agent import Agent
    _A["a"] = Agent(model_dir=Path(pkg) / "agente", device="cpu")


def _episode(job):
    from walker_lab.environment import WalkerEnv
    suite, name, scene, seed = job
    env = WalkerEnv(scenario=scene)
    obs, info = env.reset(seed=seed)
    agent = _A["a"]
    agent.reset()
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(agent.act(obs))
        done = term or trunc
    env.close()
    return {"suite": suite, "track": name, "seed": seed, "success": bool(info["success"]), "distance": float(info["distance"]),
            "time": float(info["simulated_seconds"]), "reason": info["reason"], "switch": bool(getattr(agent, "switched", False))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg", type=Path)
    ap.add_argument("name")
    ap.add_argument("--set", nargs="+", required=True, choices=list(holdout.SETS))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    for st in a.set:
        holdout.SET = st
        jobs = [(su, n, sc, sd) for su, n, sc in holdout.tracks() for sd in [sc["simulation_seed"]] + holdout.EXTRA_SEEDS[:a.seeds]]
        with get_context("spawn").Pool(a.procs, initializer=_init, initargs=(str(a.pkg.resolve()),)) as pool:
            rows = pool.map(_episode, jobs, chunksize=1)
        out = runs_root() / "evaluar_agente" / f"holdout_{st}"
        out.mkdir(parents=True, exist_ok=True)
        s = holdout.summarize(rows)
        (out / f"{a.name}.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
        sw = {k: sum(r["switch"] for r in rows if r["suite"] == k) for k in s["suites"]}
        print(f"{st:7s} {a.name}: {s['total'][0]}/{s['n']} t={-s['total'][1]:.1f}s " +
              " ".join(f"{k}={v['llegadas']}/{v['n']}" for k, v in s["suites"].items()) + f" | cambios a PPO: {sw}", flush=True)


if __name__ == "__main__":
    main()
