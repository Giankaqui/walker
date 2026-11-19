"""Model soup de policy.pt: promedia pesos y normalización de varias MLPPolicy de la misma línea de entrenamiento.

    python -m training.herramientas.soup <salida.pt> <in1.pt> <in2.pt> ...

El resultado es un policy.pt normal (misma arquitectura), cargable por agente/architecture.py.
"""
import sys
from pathlib import Path

import torch

from agente.architecture import MLPPolicy


def soup(out: Path, ins: list) -> None:
    blobs = [torch.load(p, map_location="cpu", weights_only=True) for p in ins]
    cfg = blobs[0]["cfg"]
    assert all(b["cfg"] == cfg for b in blobs), "arquitecturas distintas"
    avg = {k: torch.stack([b["state"][k].float() for b in blobs]).mean(0) for k in blobs[0]["state"]}
    policy = MLPPolicy(**cfg)
    policy.load_state_dict(avg)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.to_file(out)
    print(f"soup de {len(ins)} -> {out}")


if __name__ == "__main__":
    soup(Path(sys.argv[1]), sys.argv[2:])
