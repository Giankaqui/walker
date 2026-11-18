"""Arquitectura de la política entregada: normalización de observaciones + MLP + salida en [-1, 1].

Solo usa API estable de PyTorch (nn.Linear, state_dict). `policy.pt` guarda tensores y datos
primitivos, y se carga con weights_only=True (sin ejecutar código arbitrario).
"""

import torch
from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "silu": nn.SiLU}


class MLPPolicy(nn.Module):
    def __init__(self, sizes, activation="relu", final="tanh", normalize=False, clip_obs=10.0, eps=1e-8):
        super().__init__()
        self.cfg = {"sizes": [int(s) for s in sizes], "activation": activation, "final": final,
                    "normalize": bool(normalize), "clip_obs": float(clip_obs), "eps": float(eps)}
        layers = []
        for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(n_in, n_out))
            if i < len(sizes) - 2:
                layers.append(ACTIVATIONS[activation]())
        self.net = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(sizes[0]))
        self.register_buffer("obs_var", torch.ones(sizes[0]))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        if c["normalize"]:
            obs = torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + c["eps"]), -c["clip_obs"], c["clip_obs"])
        out = self.net(obs)
        return torch.tanh(out) if c["final"] == "tanh" else torch.clamp(out, -1.0, 1.0)

    def to_file(self, path) -> None:
        state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        torch.save({"cfg": self.cfg, "state": state}, path)


def load_policy(path, device="cpu") -> MLPPolicy:
    blob = torch.load(path, map_location=device, weights_only=True)
    policy = MLPPolicy(**blob["cfg"])
    policy.load_state_dict(blob["state"])
    return policy.to(device).eval()
