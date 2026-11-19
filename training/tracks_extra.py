"""Pistas de entrenamiento adicionales (solo para entrenar; no cambia walker_lab ni la evaluación).

`obst_var`: obstáculos de Hardcore con escala continua (0.5–1.15 del tamaño completo) y mayoría de bloques,
porque los bloques altos (≥1.2 escalones) causan la mitad de las caídas en las maratones hardcore.
"""
import numpy as np
from walker_lab.environment import WalkerEnv
from walker_lab.scenarios import BASE_Y, HARDCORE_STEP, validate_scenario

KINDS = {"obst_var": (("bloques", 0.6), ("huecos", 0.2), ("escalones", 0.2)),
         "bloques_var": (("bloques", 0.85), ("escalones", 0.15))}


def generate_extra(kind_mix, seed, length=60.0, max_steps=1800, smin=0.5, smax=1.15):
    rng = np.random.default_rng(seed)
    names, probs = zip(*KINDS[kind_mix])
    start_x = (14 / 30) * 20 / 2
    finish_x = start_x + length
    ground = [[0.0, BASE_Y], [10.0, BASE_Y]]
    obstacles, events = [], []

    def add(x, y):
        point = [round(float(x), 6), round(float(y), 6)]
        if point != ground[-1]:
            ground.append(point)

    cursor = 10.0
    while cursor < finish_x - 6:
        kind = str(rng.choice(names, p=probs))
        scale = float(rng.uniform(smin, smax))
        x0 = cursor
        if kind == "escalones":
            count = int(rng.integers(3, 5))
            width, height = 4 * HARDCORE_STEP, HARDCORE_STEP * scale
            for j in range(count):
                add(cursor, BASE_Y + (j + 1) * height); cursor += width; add(cursor, BASE_Y + (j + 1) * height)
            for j in reversed(range(count)):
                add(cursor, BASE_Y + j * height); cursor += width; add(cursor, BASE_Y + j * height)
        elif kind == "bloques":
            height = float(rng.choice([1, 2])) * HARDCORE_STEP * scale
            obstacles.append({"x": round(cursor, 6), "y": BASE_Y, "width": height, "height": height})
            cursor += height
            add(cursor, BASE_Y)
        else:
            width = float(rng.choice([2, 3])) * HARDCORE_STEP * scale
            add(cursor, BASE_Y - 2.0); cursor += width; add(cursor, BASE_Y - 2.0); add(cursor, BASE_Y)
        events.append({"kind": kind, "x_start": round(x0, 6), "x_end": round(cursor, 6), "stage": 5})
        cursor += float(rng.uniform(2.5, 4.0))
        add(cursor, BASE_Y)
    add(max(cursor, finish_x + 10), BASE_Y)
    scene = {"profile": "mixto", "seed": int(seed), "difficulty": "hardcore-plus", "simulation_seed": int(seed),
             "start_x": start_x, "length": float(length), "finish_x": finish_x, "max_steps": int(max_steps),
             "ground": ground, "obstacles": obstacles, "events": events}
    validate_scenario(scene)
    return scene


class ExtraTrackEnv(WalkerEnv):
    """Genera una pista nueva de `generate_extra` en cada reset."""

    def __init__(self, kind_mix, length=60.0, max_steps=1800):
        self._mix, self._len, self._steps = kind_mix, float(length), int(max_steps)
        self._grng = np.random.default_rng()
        super().__init__(profile="mixto", scenario=generate_extra(kind_mix, 0, length, max_steps))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._grng = np.random.default_rng(seed)
        self.fixed_scenario = generate_extra(self._mix, int(self._grng.integers(0, 2**31 - 1)), self._len, self._steps)
        return super().reset(options=options)
