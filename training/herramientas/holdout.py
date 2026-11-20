"""Re-evalúa candidatos (checkpoints PPO, best_agent o policy.pt sueltos) en pistas maratón NUEVAS con varias semillas
de condición inicial y los ordena con el criterio del ranking del curso (llegadas, -tiempo medio, distancia media).
Usa el policy.pt tal cual (sin el anti-atasco de agente/agent.py), igual para todos los candidatos.
Es la herramienta con la que se eligió la entrega; el EVAL interno del entrenamiento (10 carreras) no distingue candidatos.

    python -m training.herramientas.holdout <run_dir> [--set final] [--seeds 5] [--procs 4] [--only step_41 ...]
    python -m training.herramientas.holdout runs/comparacion --set final --seeds 5 \
        --ensemble NUEVO=runs/soup_ramas/policy.pt ACTUAL=agente/policy.pt --only NUEVO ACTUAL

Sin --set: pistas públicas + 30 nuevas. Con --set: solo pistas nuevas generadas con walker_lab (maratón 240 m):
final/confirm/test3 = 60 pistas (20 normal, 20 hardcore, 20 hardcore-plus); test4..test11 = 100 pistas (20/40/40).
--seeds 5 => 6 carreras por pista (semilla del JSON + 5). Elegir con unos sets y confirmar en otros NO usados.
Cachea resultados por candidato en <run_dir>/holdout_<set>/<candidato>_v2_s<seeds>.json.
"""
import argparse
import glob
import json
import pickle
import sys
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # raíz del repo (agente/, walker_lab/, training/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HOLDOUT = ([("normal", s) for s in range(7001, 7011)] + [("hardcore", s) for s in range(9001, 9011)]
           + [("hardcore-plus", s) for s in range(8001, 8011)])
EXTRA_SEEDS = [11, 22, 33, 44, 55, 66, 77]


FINAL = ([("normal", s) for s in range(17001, 17021)] + [("hardcore", s) for s in range(19001, 19021)]
         + [("hardcore-plus", s) for s in range(18001, 18021)])
CONFIRM = ([("normal", s) for s in range(27001, 27021)] + [("hardcore", s) for s in range(29001, 29021)]
           + [("hardcore-plus", s) for s in range(28001, 28021)])
TEST3 = ([("normal", s) for s in range(37001, 37021)] + [("hardcore", s) for s in range(39001, 39021)]
         + [("hardcore-plus", s) for s in range(38001, 38021)])
TEST4 = ([("normal", s) for s in range(47001, 47021)] + [("hardcore", s) for s in range(49001, 49041)]
         + [("hardcore-plus", s) for s in range(48001, 48041)])
TEST5 = ([("normal", s) for s in range(57001, 57021)] + [("hardcore", s) for s in range(59001, 59041)]
         + [("hardcore-plus", s) for s in range(58001, 58041)])
TEST6 = ([("normal", s) for s in range(67001, 67021)] + [("hardcore", s) for s in range(69001, 69041)]
         + [("hardcore-plus", s) for s in range(68001, 68041)])
TEST7 = ([("normal", s) for s in range(77001, 77021)] + [("hardcore", s) for s in range(79001, 79041)]
         + [("hardcore-plus", s) for s in range(78001, 78041)])
TEST8 = ([("normal", s) for s in range(87001, 87021)] + [("hardcore", s) for s in range(89001, 89041)]
         + [("hardcore-plus", s) for s in range(88001, 88041)])
TEST9 = ([("normal", s) for s in range(97001, 97021)] + [("hardcore", s) for s in range(99001, 99041)]
         + [("hardcore-plus", s) for s in range(98001, 98041)])
TEST10 = ([("normal", s) for s in range(107001, 107021)] + [("hardcore", s) for s in range(109001, 109041)]
          + [("hardcore-plus", s) for s in range(108001, 108041)])
TEST11 = ([("normal", s) for s in range(117001, 117021)] + [("hardcore", s) for s in range(119001, 119041)]
          + [("hardcore-plus", s) for s in range(118001, 118041)])
TEST12 = ([("normal", s) for s in range(127001, 127021)] + [("hardcore", s) for s in range(129001, 129041)]
          + [("hardcore-plus", s) for s in range(128001, 128041)])
TEST13 = ([("normal", s) for s in range(137001, 137021)] + [("hardcore", s) for s in range(139001, 139041)]
          + [("hardcore-plus", s) for s in range(138001, 138041)])
TEST14 = ([("normal", s) for s in range(147001, 147021)] + [("hardcore", s) for s in range(149001, 149041)]
          + [("hardcore-plus", s) for s in range(148001, 148041)])
TEST15 = ([("normal", s) for s in range(157001, 157021)] + [("hardcore", s) for s in range(159001, 159041)]
          + [("hardcore-plus", s) for s in range(158001, 158041)])
TEST16 = ([("normal", s) for s in range(167001, 167021)] + [("hardcore", s) for s in range(169001, 169041)]
          + [("hardcore-plus", s) for s in range(168001, 168041)])
TEST17 = ([("normal", s) for s in range(177001, 177021)] + [("hardcore", s) for s in range(179001, 179041)]
          + [("hardcore-plus", s) for s in range(178001, 178041)])
TEST18 = ([("normal", s) for s in range(187001, 187021)] + [("hardcore", s) for s in range(189001, 189041)]
          + [("hardcore-plus", s) for s in range(188001, 188041)])
TEST19 = ([("normal", s) for s in range(197001, 197021)] + [("hardcore", s) for s in range(199001, 199041)]
          + [("hardcore-plus", s) for s in range(198001, 198041)])
TEST20 = ([("normal", s) for s in range(207001, 207021)] + [("hardcore", s) for s in range(209001, 209041)]
          + [("hardcore-plus", s) for s in range(208001, 208041)])
TEST21 = ([("normal", s) for s in range(217001, 217021)] + [("hardcore", s) for s in range(219001, 219041)]
          + [("hardcore-plus", s) for s in range(218001, 218041)])
TEST22 = ([("normal", s) for s in range(227001, 227021)] + [("hardcore", s) for s in range(229001, 229041)]
          + [("hardcore-plus", s) for s in range(228001, 228041)])
TEST23 = ([("normal", s) for s in range(237001, 237021)] + [("hardcore", s) for s in range(239001, 239041)]
          + [("hardcore-plus", s) for s in range(238001, 238041)])
TEST24 = ([("normal", s) for s in range(247001, 247021)] + [("hardcore", s) for s in range(249001, 249041)]
          + [("hardcore-plus", s) for s in range(248001, 248041)])
TEST25 = ([("normal", s) for s in range(257001, 257021)] + [("hardcore", s) for s in range(259001, 259041)]
          + [("hardcore-plus", s) for s in range(258001, 258041)])
SETS = {"final": FINAL, "confirm": CONFIRM, "test3": TEST3, "test4": TEST4, "test5": TEST5, "test6": TEST6,
        "test7": TEST7, "test8": TEST8, "test9": TEST9, "test10": TEST10, "test11": TEST11, "test12": TEST12, "test13": TEST13, "test14": TEST14, "test15": TEST15, "test16": TEST16, "test17": TEST17, "test18": TEST18, "test19": TEST19, "test20": TEST20, "test21": TEST21, "test22": TEST22, "test23": TEST23, "test24": TEST24, "test25": TEST25}
SET = None


def tracks():
    from walker_lab.scenarios import generate_scenario, load_scenario
    out = []
    if SET:  # solo pistas nuevas
        for diff, seed in SETS[SET]:
            suite = {"normal": "F_normal", "hardcore": "F_hcn", "hardcore-plus": "F_hc"}[diff]
            out.append((suite, f"{diff}_{seed}", generate_scenario("maraton", seed, 240, 6000, diff)))
        return out
    for f in sorted(glob.glob(str(ROOT / "escenarios/validacion/*.json"))):
        out.append(("pub_val", Path(f).stem, load_scenario(f)))
    for f in sorted(glob.glob(str(ROOT / "escenarios/hardcore/*.json"))):
        out.append(("pub_hc", Path(f).stem, load_scenario(f)))
    for diff, seed in HOLDOUT:
        suite = {"normal": "ho_normal", "hardcore": "ho_hcn", "hardcore-plus": "ho_hc"}[diff]
        out.append((suite, f"{diff}_{seed}", generate_scenario("maraton", seed, 240, 6000, diff)))
    return out


class _VN:  # imita VecNormalize (clip_obs=10, epsilon=1e-8) para build_policy
    def __init__(self, obs_rms):
        self.obs_rms, self.clip_obs, self.epsilon = obs_rms, 10.0, 1e-8

    def normalize_obs(self, obs):
        import numpy as np
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs).astype(np.float32)


def export_candidate(zip_path: Path, out_pt: Path):
    """Checkpoint SB3 (+ su _obs_rms.pkl) -> policy.pt de la entrega (MLPPolicy con normalización incluida)."""
    from stable_baselines3 import PPO
    from training.export import build_policy
    model = PPO.load(zip_path, device="cpu")
    rms_path = zip_path.with_name(zip_path.stem + "_obs_rms.pkl")
    vn = _VN(pickle.loads(rms_path.read_bytes())) if rms_path.is_file() else None
    policy = build_policy(model, vn, verify=True)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    policy.to_file(out_pt)


_W = {}


def _init(root, policy_path):
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    torch.set_num_threads(1)
    from agente.architecture import load_policy
    paths = policy_path.split("|")
    if len(paths) == 1:
        _W["policy"] = load_policy(paths[0], "cpu")
    else:  # ensemble: promedio de acciones
        pols = [load_policy(pp, "cpu") for pp in paths]
        _W["policy"] = lambda x: torch.stack([pp(x) for pp in pols]).mean(0)


def _episode(job):
    import numpy as np
    import torch
    from walker_lab.environment import WalkerEnv
    suite, name, scene, seed = job
    env = WalkerEnv(scenario=scene)
    try:
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            with torch.no_grad():
                a = _W["policy"](torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0].numpy()
            obs, _, term, trunc, info = env.step(np.clip(a, -1, 1))
            done = term or trunc
        return {"suite": suite, "track": name, "seed": seed, "success": bool(info["success"]),
                "distance": float(info["distance"]), "time": float(info["simulated_seconds"]), "reason": info["reason"]}
    finally:
        env.close()


def rank_key(rows):
    ok = [r for r in rows if r["success"]]
    mean_t = sum(r["time"] for r in ok) / len(ok) if ok else 0.0
    return [len(ok), -round(mean_t, 2), sum(r["distance"] for r in rows) / max(1, len(rows))]


def evaluate(policy_path: Path, n_seeds: int, procs: int):
    jobs = []
    for suite, name, scene in tracks():
        for seed in [scene["simulation_seed"]] + EXTRA_SEEDS[:n_seeds]:
            jobs.append((suite, name, scene, seed))
    with get_context("spawn").Pool(procs, initializer=_init, initargs=(str(ROOT), str(policy_path))) as pool:
        return pool.map(_episode, jobs, chunksize=1)


def summarize(rows):
    suites = {}
    for r in rows:
        suites.setdefault(r["suite"], []).append(r)
    s = {k: {"n": len(v), "llegadas": sum(x["success"] for x in v), "key": rank_key(v),
             "dist": sum(x["distance"] for x in v) / len(v)} for k, v in suites.items()}
    return {"total": rank_key(rows), "n": len(rows), "suites": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="run con ckpt/ (y best_agent/), o cualquier carpeta para --ensemble")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=2, help="semillas extra por pista (además de la del JSON)")
    ap.add_argument("--set", choices=list(SETS), help="set de pistas nuevas (ver docstring); sin él, públicas + 30 nuevas")
    ap.add_argument("--ensemble", nargs="*", help="nombre=ruta.pt (o ruta1|ruta2|... = promedio de acciones)")
    ap.add_argument("--only", nargs="*", help="prefijos de candidatos a evaluar (p.ej. step_41 best_agent X)")
    args = ap.parse_args()
    global SET
    SET = args.set
    out_dir = args.run_dir / (f"holdout_{args.set}" if args.set else "holdout")
    cands = {}
    for z in sorted((args.run_dir / "ckpt").glob("step_*.zip")):
        cands[z.stem] = ("zip", z)
    if (args.run_dir / "best_agent" / "policy.pt").is_file():
        meta = json.loads((args.run_dir / "best_agent" / "meta.json").read_text())
        cands[f"best_agent@{meta.get('step')}"] = ("pt", args.run_dir / "best_agent" / "policy.pt")
    for spec in args.ensemble or []:
        name, paths = spec.split("=", 1)
        cands[name] = ("pt", Path(paths))
    if args.only:
        cands = {k: v for k, v in cands.items() if any(k.startswith(o) for o in args.only)}
    results = {}
    for name, (kind, path) in cands.items():
        cache = out_dir / f"{name}_v2_s{args.seeds}.json"
        if cache.is_file():
            results[name] = json.loads(cache.read_text())["summary"]
            continue
        pt = path
        if kind == "zip":
            pt = out_dir / "policies" / f"{name}.pt"
            if not pt.is_file():
                export_candidate(path, pt)
        rows = evaluate(pt, args.seeds, args.procs)
        summary = summarize(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"summary": summary, "policy": str(pt), "rows": rows}, indent=1))
        results[name] = summary
        print(f"{name}: total {summary['total']} | " + " | ".join(
            f"{k} {v['llegadas']}/{v['n']}" for k, v in summary["suites"].items()), flush=True)
    print("\n=== RANKING (llegadas, -tiempo medio, distancia media) ===")
    for name, s in sorted(results.items(), key=lambda kv: kv[1]["total"], reverse=True):
        print(f"{name:28s} {s['total'][0]:3d}/{s['n']}  t={-s['total'][1]:6.1f}s  d={s['total'][2]:6.1f}  " + " ".join(
            f"{k}={v['llegadas']}/{v['n']}({v['dist']:.0f}m)" for k, v in s["suites"].items()))


if __name__ == "__main__":
    main()
