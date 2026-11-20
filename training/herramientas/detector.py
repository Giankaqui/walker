"""Detector de "pista difícil" para el agente híbrido (SAC en pistas normales, PPO en hardcore).

    # 1) carreras de la política rápida (SAC) en maratones de ENTRENAMIENTO (semillas 400001+, no usadas para evaluar)
    python -m training.herramientas.detector datos runs/detector/sac.npz agente/policy_sac.pt --seed0 400001 --n 400
    # 2) MLP 24-64-64-1 por paso -> log-odds de "pista hardcore/hardcore-plus"
    python -m training.herramientas.detector entrenar runs/detector/sac.npz runs/detector/red.pt
    # 3) empaquetar con la regla de cambio: media del log-odds en W pasos > thr  =>  cambiar a PPO (una sola vez)
    python -m training.herramientas.detector empaquetar runs/detector/red.pt agente/detector.pt --W 50 --thr 4

`entrenar` imprime, para validación (25 % de las carreras), falsos positivos en pistas normales y distancia de
detección en hardcore para varias W/thr. La elección final de W/thr se hace con evaluar_agente.py en miles de carreras.
"""
import argparse
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch
from torch import nn

DIFFS = ("normal", "hardcore", "hardcore-plus")
_P = {}


def red():
    return nn.Sequential(nn.Linear(24, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))


def _init(policy):
    torch.set_num_threads(1)
    from agente.architecture import load_policy
    _P["p"] = load_policy(policy, "cpu")


def _run(job):
    from walker_lab.environment import WalkerEnv
    from walker_lab.scenarios import generate_scenario
    diff, seed = job
    env = WalkerEnv(scenario=generate_scenario("maraton", seed, 240, 6000, diff))
    obs, info = env.reset(seed=seed)
    O, D, done = [], [], False
    while not done:
        O.append(obs.astype(np.float32)); D.append(info.get("distance", 0.0) if info else 0.0)
        with torch.no_grad():
            a = _P["p"](torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0].numpy()
        obs, _, term, trunc, info = env.step(np.clip(a, -1, 1))
        done = term or trunc
    env.close()
    return diff, np.array(O), np.array(D, dtype=np.float32)


def datos(a):
    jobs = [(d, a.seed0 + k * 10 + i) for i in range(a.n) for k, d in enumerate(DIFFS)]
    with get_context("spawn").Pool(a.procs, initializer=_init, initargs=(str(a.policy),)) as pool:
        res = pool.map(_run, jobs, chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, obs=np.concatenate([r[1] for r in res]), dist=np.concatenate([r[2] for r in res]),
                        ep=np.concatenate([np.full(len(r[1]), i) for i, r in enumerate(res)]),
                        hard=np.concatenate([np.full(len(r[1]), r[0] != "normal") for r in res]))
    print(f"{len(res)} carreras -> {a.out}")


def entrenar(a):
    d = np.load(a.datos)
    obs, ep, hard, dist = d["obs"], d["ep"], d["hard"].astype(np.float32), d["dist"]
    rng = np.random.default_rng(0)
    va = set(rng.choice(np.unique(ep), len(np.unique(ep)) // 4, replace=False).tolist())
    mtr = np.array([e not in va for e in ep])
    mu, sd = obs[mtr].mean(0), obs[mtr].std(0) + 1e-6
    torch.manual_seed(0)
    net = red()
    X, Y = torch.tensor((obs - mu) / sd, dtype=torch.float32), torch.tensor(hard)
    opt, lossf, idx = torch.optim.Adam(net.parameters(), 1e-3), nn.BCEWithLogitsLoss(), np.where(mtr)[0]
    for _ in range(a.iters):
        b = torch.tensor(rng.choice(idx, 4096))
        loss = lossf(net(X[b]).squeeze(1), Y[b]); opt.zero_grad(); loss.backward(); opt.step()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": torch.tensor(mu), "sd": torch.tensor(sd)}, a.out)
    with torch.no_grad():
        lo = net(X).squeeze(1).numpy()
    for W in (25, 50, 100):
        for thr in (2, 3, 4, 5):
            fp = nn_ = 0; det = []
            for e in sorted(va):
                m = ep == e; cs = np.convolve(lo[m], np.ones(W) / W, mode="valid"); hit = np.where(cs > thr)[0]
                if hard[m][0]:
                    det.append(dist[m][hit[0] + W - 1] if len(hit) else np.inf)
                else:
                    nn_ += 1; fp += bool(len(hit))
            det = np.array(det); ok = det[np.isfinite(det)]
            print(f"W {W:3d} thr {thr}: falsos positivos {fp}/{nn_} | hardcore detectadas {len(ok)}/{len(det)}, "
                  f"distancia mediana {np.median(ok) if len(ok) else -1:.0f} m")


def empaquetar(a):
    g = torch.load(a.red, weights_only=True)
    torch.save({**g, "W": int(a.W), "thr": float(a.thr)}, a.out)
    print(f"{a.out}: W={a.W} thr={a.thr}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("datos"); p.add_argument("out", type=Path); p.add_argument("policy", type=Path)
    p.add_argument("--seed0", type=int, default=400001); p.add_argument("--n", type=int, default=400); p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("entrenar"); p.add_argument("datos", type=Path); p.add_argument("out", type=Path); p.add_argument("--iters", type=int, default=15000)
    p = sub.add_parser("empaquetar"); p.add_argument("red", type=Path); p.add_argument("out", type=Path)
    p.add_argument("--W", type=int, default=50); p.add_argument("--thr", type=float, default=4.0)
    a = ap.parse_args()
    {"datos": datos, "entrenar": entrenar, "empaquetar": empaquetar}[a.cmd](a)


if __name__ == "__main__":
    main()
