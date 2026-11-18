"""Detección de plataforma (Colab / Kaggle / local) y rutas de salida."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
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


def system_stats() -> dict:
    """Uso de GPU/CPU/RAM en este instante (cada dato es opcional: si no se puede leer, se omite)."""
    stats: dict = {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        util, used, total = (float(x) for x in out.strip().splitlines()[0].split(","))
        stats.update(gpu_util_pct=util, gpu_mem_used_mb=used, gpu_mem_total_mb=total)
    except Exception:
        pass
    try:
        import psutil
        stats.update(cpu_pct=psutil.cpu_percent(interval=None), ram_pct=psutil.virtual_memory().percent)
    except Exception:
        pass
    return stats


class GpuSampler:
    """Utilización media de la GPU en una ventana, muestreada en segundo plano.

    Una lectura puntual engaña: la GPU solo trabaja en la fase de actualización, no mientras se recolecta.
    Ojo: NVML mide "% del tiempo con algún kernel activo", no cuánto cómputo usa; con kernels diminutos puede dar
    alto sin estar saturada. La prueba decisiva es comparar ms/actualización entre dos GPUs.
    """

    def __init__(self, interval: float = 1.0):
        self._vals: list[float] = []
        self._lock = threading.Lock()
        self._read = self._make_reader()
        self.interval = interval if self._read is not None and not hasattr(self._read, "smi") else max(interval, 2.0)
        if self._read is not None:
            threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _make_reader():
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return lambda: float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            pass
        if shutil.which("nvidia-smi"):
            def smi() -> float:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                return float(out.strip().splitlines()[0])
            smi.smi = True
            return smi
        return None

    def _loop(self) -> None:
        import time
        while True:
            try:
                value = self._read()
                with self._lock:
                    self._vals.append(value)
            except Exception:
                pass
            time.sleep(self.interval)

    def mean_and_reset(self) -> float | None:
        with self._lock:
            vals, self._vals = self._vals, []
        return round(sum(vals) / len(vals), 1) if vals else None
