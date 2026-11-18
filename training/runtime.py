"""Detección de plataforma (Colab / Kaggle / local) y rutas de salida."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def detect_platform() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").is_dir():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"


def runs_root() -> Path:
    """Dónde viven los experimentos. Persistente en Drive si está montado (Colab)."""
    override = os.environ.get("LAB4_RUNS_DIR")
    if override:
        return Path(override)
    platform = detect_platform()
    if platform == "colab":
        drive = Path("/content/drive/MyDrive")
        return drive / "lab4_runs" if drive.is_dir() else Path("/content/runs")
    if platform == "kaggle":
        return Path("/kaggle/working/runs")
    return REPO_ROOT / "runs"


def default_n_envs() -> int:
    return max(1, min(os.cpu_count() or 2, 8))


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = run("rev-parse", "--short", "HEAD")
    return {"commit": sha or None, "dirty": bool(run("status", "--porcelain")) if sha else None}


def environment_info() -> dict:
    info = {"platform": detect_platform(), "python": sys.version.split()[0], "cpus": os.cpu_count()}
    for mod in ("torch", "stable_baselines3", "gymnasium", "numpy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    try:
        import torch
        info["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        info["cuda"] = None
    return info


def persistence_warning(run_dir: Path) -> str | None:
    """Avisa cuando los resultados viven en disco efímero y se perderían si la sesión muere."""
    platform = detect_platform()
    if platform == "colab" and not str(run_dir).startswith("/content/drive"):
        return ("⚠ ATENCIÓN: Drive no está montado; los runs se guardan en disco temporal y se PIERDEN si la sesión "
                "se cierra. Monte Drive (celda 2) o fije LAB4_RUNS_DIR.")
    if platform == "kaggle":
        return ("ℹ Kaggle solo conserva /kaggle/working al terminar la ejecución con 'Save Version' (Run All). "
                "Si la sesión se corta antes, se pierde; use time_limit_hours para salir limpio antes del límite.")
    return None
