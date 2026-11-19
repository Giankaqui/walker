"""Model soup del modelo SB3 completo (actor + crítico + normalización) para usarlo como `init_from`.

    python -m training.herramientas.soup_sb3 <out_dir> [--base ckpt.zip] ckpt1.zip ckpt2.zip ...
    (cada step_N.zip con su step_N_obs_rms.pkl al lado, como los deja training.train en <run>/ckpt/)

Crea <out_dir>/best/model.zip y <out_dir>/best/obs_rms.pkl. Con <out_dir> = <runs>/init_X, un YAML con
`init_from: init_X` afina desde el promedio. Pesos y normalización = promedio (en el orden dado); el resto del
modelo (contador de pasos `num_timesteps`, estado del optimizador, semilla) sale de `--base` (por defecto, el primer zip).
"""
import argparse
import copy
import pickle
from pathlib import Path

import torch
from stable_baselines3 import PPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("zips", type=Path, nargs="+")
    ap.add_argument("--base", type=Path, help="checkpoint del que se copian num_timesteps, optimizador y semilla")
    args = ap.parse_args()
    out, zips = args.out_dir / "best", args.zips
    models = [PPO.load(z, device="cpu") for z in zips]
    sds = [m.policy.state_dict() for m in models]
    avg = {k: (torch.stack([sd[k].float() for sd in sds]).mean(0).to(sds[0][k].dtype) if sds[0][k].is_floating_point() else sds[0][k])
           for k in sds[0]}
    base = PPO.load(args.base, device="cpu") if args.base else models[0]
    base.policy.load_state_dict(avg)
    rmss = [pickle.loads(z.with_name(z.stem + "_obs_rms.pkl").read_bytes()) for z in zips]
    rms = copy.deepcopy(rmss[0])
    rms.mean = sum(r.mean for r in rmss) / len(rmss)
    rms.var = sum(r.var for r in rmss) / len(rmss)
    out.mkdir(parents=True, exist_ok=True)
    base.save(out / "model.zip")
    (out / "obs_rms.pkl").write_bytes(pickle.dumps(rms))
    print(f"soup SB3 de {len(zips)} -> {out}")


if __name__ == "__main__":
    main()
