"""Soup de una ventana de checkpoints de uno o varios runs ("soup entre ramas").

    python -m training.herramientas.soup_window <salida.pt> <inicio_M> <fin_M> <run1> [<run2> ...]

Para cada run toma los checkpoints más cercanos a inicio_M, inicio_M+1, ..., fin_M millones de pasos
(checkpoints cada 1M), los exporta a <run>/policies/step_N.pt (MLPPolicy, igual que la entrega) y promedia todos.
Ejemplo (reconstruye agente/policy.pt bit a bit; el orden de los runs importa por redondeo):
    ... soup_window runs/soup_ramas/policy.pt 41.5 45.5 runs/ppo_it3_blq runs/ppo_it3_blq6 runs/ppo_it3_a
"""
import sys
from pathlib import Path

from training.herramientas.holdout import export_candidate
from training.herramientas.soup import soup


def window(run: Path, a: float, b: float) -> list:
    zips = list((run / "ckpt").glob("step_*.zip"))
    assert zips, f"sin checkpoints en {run / 'ckpt'}"
    pts, m = [], a
    while m <= b + 1e-6:
        z = min(zips, key=lambda p: abs(int(p.stem[5:]) - m * 1e6))  # con n_envs que no divide 1M, los nombres derivan un poco
        assert abs(int(z.stem[5:]) - m * 1e6) < 100_000, f"no hay checkpoint cerca de {m}M en {run}"
        pt = run / "policies" / f"{z.stem}.pt"
        if not pt.is_file():
            export_candidate(z, pt)
        pts.append(str(pt))
        m += 1.0
    return pts


def main():
    out, a, b, runs = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), [Path(r) for r in sys.argv[4:]]
    soup(out, [pt for run in runs for pt in window(run, a, b)])


if __name__ == "__main__":
    main()
