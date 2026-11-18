"""Pipeline de entrenamiento del Lab 4 (RL). Se ejecuta con: python -m training.train --config configs/X.yaml"""

import sys

# Un print con tildes/símbolos nunca debe tumbar un entrenamiento (p. ej. consola Windows en cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - streams sin reconfigure (algunos notebooks)
        pass
