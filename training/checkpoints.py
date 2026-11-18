"""Checkpoints reanudables: modelo + replay buffer + estadísticas de normalización."""

from __future__ import annotations

import json
import os
import pickle
import traceback
from pathlib import Path


class CheckpointManager:
    def __init__(self, run_dir: Path, keep: int = 2, save_replay_buffer: bool = True):
        self.dir = Path(run_dir) / "ckpt"
        self.keep = max(2, keep)  # mínimo 2: si el último está corrupto, queda el anterior
        self.save_replay_buffer = save_replay_buffer
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, vn=None) -> int:
        step = int(model.num_timesteps)
        model.save(self.dir / f"step_{step}.zip")
        if vn is not None:
            (self.dir / f"step_{step}_obs_rms.pkl").write_bytes(pickle.dumps(vn.obs_rms))
        if self.save_replay_buffer and getattr(model, "replay_buffer", None) is not None:
            model.save_replay_buffer(self.dir / f"step_{step}_replay.pkl")
        tmp = self.dir / "latest.json.tmp"  # latest.json se escribe al final y de forma atómica
        tmp.write_text(json.dumps({"step": step}))
        os.replace(tmp, self.dir / "latest.json")
        self._prune()
        return step

    def try_save(self, model, vn=None) -> bool:
        """Como save(), pero un fallo (disco lleno, Drive desconectado…) solo avisa y el entrenamiento sigue."""
        try:
            self.save(model, vn)
            return True
        except Exception:
            traceback.print_exc()
            print("    ⚠ No se pudo guardar el checkpoint; se reintentará en el siguiente intervalo.", flush=True)
            return False

    def candidates(self) -> list[dict]:
        """Checkpoints disponibles del más reciente al más antiguo (el que marca latest.json va primero)."""
        steps = sorted({int(p.stem.split("_")[1]) for p in self.dir.glob("step_*.zip")}, reverse=True)
        marker = self.dir / "latest.json"
        if marker.is_file():
            try:
                latest = json.loads(marker.read_text())["step"]
                steps = [latest] + [s for s in steps if s != latest]
            except Exception:
                pass
        out = []
        for step in steps:
            model = self.dir / f"step_{step}.zip"
            if not model.is_file():
                continue
            replay, rms = self.dir / f"step_{step}_replay.pkl", self.dir / f"step_{step}_obs_rms.pkl"
            out.append({"step": step, "model": model, "replay": replay if replay.is_file() else None,
                        "obs_rms_path": rms if rms.is_file() else None})
        return out

    def _prune(self) -> None:
        steps = sorted({int(p.stem.split("_")[1]) for p in self.dir.glob("step_*.zip")})
        for old in steps[: -self.keep]:
            for pattern in (f"step_{old}.zip", f"step_{old}_*"):  # sin prefijos ambiguos (1000 vs 10000)
                for p in self.dir.glob(pattern):
                    p.unlink(missing_ok=True)
