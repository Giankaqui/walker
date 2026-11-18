"""Resumen realista de un run a partir de sus métricas (pegable en un chat).

    python -m training.report /content/drive/MyDrive/lab4_runs/sac_max
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import yaml


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # línea a medio escribir por un corte
    return rows


def median(rows: list[dict], key: str):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return st.median(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumen de un run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=10, help="Últimas N mediciones para la velocidad estable.")
    args = parser.parse_args()
    run = args.run_dir
    metrics, evals = read_jsonl(run / "metrics.jsonl"), read_jsonl(run / "eval_history.jsonl")
    cfg_file = run / "config.resolved.yaml"
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) if cfg_file.is_file() else {}
    print(f"RUN {run.name} | algo={cfg.get('algo')} | utd={cfg.get('utd')} | {len(metrics)} mediciones, {len(evals)} evaluaciones")
    if metrics:
        last, win = metrics[-1], metrics[-args.window:]
        sps = median(win, "sps")
        print(f"\nPaso actual: {last['step']:,}  ({last['stage']})")
        print(f"Velocidad estable (mediana de {len(win)}): {sps:.0f} sps")
        pct = {k: st.median(r["pct"][k] for r in win) for k in ("collect", "update", "eval", "ckpt")}
        print("Reparto del tiempo: " + ", ".join(f"{k} {v:.0f}%" for k, v in pct.items())
              + f", otros {max(0, 100 - sum(pct.values())):.0f}%")
        for key, label in (("ms_per_update", "ms/actualización"), ("gpu_util_pct", "GPU util %"), ("cpu_pct", "CPU %"), ("ram_pct", "RAM %")):
            v = median(win, key)
            if v is not None:
                print(f"{label}: {v:.1f}")
        if cfg.get("stages") and sps:
            ends, total = [], 0
            for s in cfg["stages"]:
                total += int(s["timesteps"])
                ends.append((s.get("name", ""), total))
            print("\nProyección a la velocidad actual:")
            for name, end in ends:
                if end > last["step"]:
                    print(f"  fin de '{name}' ({end:,}): +{(end - last['step']) / sps / 3600:.1f} h")
    if evals:
        print("\nEvaluaciones (llegadas/n · tiempo medio de llegadas · distancia media):")
        for e in evals[-8:]:
            cells = []
            for suite, v in e["summary"].items():
                t = f" t={v['mean_time_arrived']:.0f}s" if v.get("mean_time_arrived") is not None else ""
                cells.append(f"{suite} {v['arrivals']}/{v['n']}{t} d={v['mean_distance']:.0f}")
            print(f"  {e['step']:>11,}  " + " | ".join(cells) + ("  *mejor" if e.get("best") else ""))


if __name__ == "__main__":
    main()
