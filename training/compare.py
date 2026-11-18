"""Compara los mejores agentes de varios runs con el criterio del ranking y (opcional) promueve el ganador.

    python -m training.compare runs/*                      # tabla
    python -m training.compare /content/drive/MyDrive/lab4_runs/* --promote agente
Solo son comparables runs evaluados con las mismas suites (`eval.select_on`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import promote


def load(run: Path) -> dict | None:
    f = run / "best" / "best.json"
    if not f.is_file():
        return None
    data = json.loads(f.read_text())
    key = data.get("key") or [0, 0.0, data.get("score", 0.0)]  # formato antiguo: solo distancia
    return {"run": run, "step": data["step"], "key": tuple(key)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ranking de runs.")
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--promote", type=Path, help="Copia el mejor agente ganador a esta carpeta (p. ej. agente).")
    args = parser.parse_args()
    rows = sorted(filter(None, (load(r) for r in args.runs if r.is_dir())), key=lambda r: r["key"], reverse=True)
    if not rows:
        raise SystemExit("Ningún run tiene best/best.json todavía.")
    print(f"{'#':>2}  {'run':32s} {'paso':>12s}  {'llegadas':>8s}  {'t medio (s)':>11s}  {'dist media':>10s}")
    for i, r in enumerate(rows, 1):
        arrivals, neg_time, dist = r["key"]
        t = f"{-neg_time:11.1f}" if arrivals else f"{'-':>11s}"
        print(f"{i:>2}  {r['run'].name:32s} {r['step']:>12,}  {int(arrivals):>8d}  {t}  {dist:10.1f}")
    if args.promote:
        promote(rows[0]["run"], args.promote, "best")


if __name__ == "__main__":
    main()
