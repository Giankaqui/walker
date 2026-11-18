"""Checkpoints reanudables: modelo + replay buffer + estadísticas de normalización."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path


class CheckpointManager:
    def __init__(self, run_dir: Path, keep: int = 2, save_replay_buffer: bool = True):
        self.dir = Path(run_dir) / "ckpt"
        self.keep = keep
        self.save_replay_buffer = save_replay_buffer
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, vn=None) -> int:
        step = int(model.num_timesteps)
        model.save(self.dir / f"step_{step}.zip")
        if vn is not None:
            (self.dir / f"step_{step}_obs_rms.pkl").write_bytes(pickle.dumps(vn.obs_rms))
        if self.save_replay_buffer and hasattr(model, "replay_buffer") and model.replay_buffer is not None:
            model.save_replay_buffer(self.dir / f"step_{step}_replay.pkl")
        tmp = self.dir / "latest.json.tmp"  # latest.json se escribe al final y de forma atómica
        tmp.write_text(json.dumps({"step": step}))
        os.replace(tmp, self.dir / "latest.json")
        self._prune(step)
        return step

    def latest(self) -> dict | None:
        marker = self.dir / "latest.json"
        if not marker.is_file():
            return None
        step = json.loads(marker.read_text())["step"]
        model = self.dir / f"step_{step}.zip"
        if not model.is_file():
            return None
        replay = self.dir / f"step_{step}_replay.pkl"
        rms = self.dir / f"step_{step}_obs_rms.pkl"
        return {"step": step, "model": model, "replay": replay if replay.is_file() else None,
                "obs_rms": pickle.loads(rms.read_bytes()) if rms.is_file() else None}

    def _prune(self, current: int) -> None:
        steps = sorted({int(p.name.split("_")[1].split(".")[0]) for p in self.dir.glob("step_*.zip")})
        for old in steps[: -self.keep]:
            for pattern in (f"step_{old}.zip", f"step_{old}_*"):  # sin prefijos ambiguos (1000 vs 10000)
                for p in self.dir.glob(pattern):
                    p.unlink(missing_ok=True)
