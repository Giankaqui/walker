"""Convierte la política de SB3 en la `MLPPolicy` de agente/ (con normalización incluida) y la guarda.

La MLPPolicy es la misma que se usa para evaluar durante el entrenamiento y para la entrega, así que
lo que se mide es exactamente lo que se entrega. Al convertir se verifica contra la política original.

    python -m training.export runs/mi_exp --to agente            # copia best_agent -> agente/
    python -m training.export runs/mi_exp --to agente --which last
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from agente.architecture import ACTIVATIONS, MLPPolicy

OBS_DIM = 24
_ACT_NAMES = {cls: name for name, cls in ACTIVATIONS.items()}


def _split_mlp(model):
    """Devuelve (módulos del MLP del actor en orden, salida 'tanh'|'clip')."""
    algo, pol = type(model).__name__, model.policy
    if algo == "SAC":
        return list(pol.actor.latent_pi) + [pol.actor.mu], "tanh"
    if algo == "TD3":
        return [m for m in pol.actor.mu if not isinstance(m, nn.Tanh)], "tanh"
    if algo == "PPO":
        return list(pol.mlp_extractor.policy_net) + [pol.action_net], "clip"
    raise NotImplementedError(f"Export no soportado para {algo}")


def build_policy(model, vn=None, verify: bool = True) -> MLPPolicy:
    """SB3 model (+ VecNormalize opcional) -> MLPPolicy en CPU."""
    modules, final = _split_mlp(model)
    linears = [m for m in modules if isinstance(m, nn.Linear)]
    acts = {_ACT_NAMES[type(m)] for m in modules if type(m) in _ACT_NAMES}
    if len(acts) > 1 or any(not isinstance(m, nn.Linear) and type(m) not in _ACT_NAMES for m in modules):
        raise NotImplementedError(f"Arquitectura no soportada por el exportador: {modules}")
    sizes = [linears[0].in_features] + [l.out_features for l in linears]
    policy = MLPPolicy(sizes, acts.pop() if acts else "relu", final, normalize=vn is not None,
                       clip_obs=vn.clip_obs if vn is not None else 10.0, eps=vn.epsilon if vn is not None else 1e-8)
    with torch.no_grad():
        for dst, src in zip([m for m in policy.net if isinstance(m, nn.Linear)], linears):
            dst.weight.copy_(src.weight.cpu())
            dst.bias.copy_(src.bias.cpu())
        if vn is not None:
            policy.obs_mean.copy_(torch.as_tensor(vn.obs_rms.mean, dtype=torch.float32))
            policy.obs_var.copy_(torch.as_tensor(vn.obs_rms.var, dtype=torch.float32))
    policy.eval()
    if verify:
        _verify(model, vn, policy)
    return policy


def _verify(model, vn, policy: MLPPolicy) -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(0.0, 1.0, (64, OBS_DIM)).astype(np.float32)
    seen = vn.normalize_obs(raw) if vn is not None else raw  # lo que ve SB3 durante el entrenamiento
    with torch.no_grad():
        ref = model.policy._predict(torch.as_tensor(seen, device=model.device), deterministic=True).clamp(-1, 1).cpu()
        got = policy(torch.as_tensor(raw))
    if not torch.allclose(ref, got, atol=1e-4):
        raise RuntimeError(f"La política exportada difiere de la original (máx. error {(ref - got).abs().max():.2e}).")


def save_agent(policy: MLPPolicy, out_dir: Path, meta: dict) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.to_file(out_dir / "policy.pt")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def promote(run_dir: Path, dest: Path, which: str = "best") -> None:
    src = Path(run_dir) / f"{which}_agent"
    if not (src / "policy.pt").is_file():
        raise FileNotFoundError(f"No existe {src / 'policy.pt'} (¿ya corrió alguna evaluación?)")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("policy.pt", "meta.json"):
        shutil.copy2(src / name, dest / name)
    print(f"Copiado {src} -> {dest}")


def main():
    parser = argparse.ArgumentParser(description="Copiar el agente exportado de un run a la carpeta de entrega.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--to", type=Path, default=Path("agente"))
    parser.add_argument("--which", choices=("best", "last"), default="best")
    args = parser.parse_args()
    promote(args.run_dir, args.to, args.which)


if __name__ == "__main__":
    main()
