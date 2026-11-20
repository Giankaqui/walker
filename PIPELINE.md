# Pipeline de entrenamiento · Lab 4

# Guía de corrida paso a paso (Colab Pro)

**Siempre `Run all`.** Solo cambia `MODE` en la celda 1.
1. Colab → *Archivo → Abrir cuaderno → GitHub* → `Vasco2510/walker` → `notebooks/train_cloud.ipynb`.
2. *Entorno de ejecución → Cambiar tipo de entorno*: acelerador **T4 GPU** + **RAM alta** activada → Guardar. (¿T4 o L4? ver más abajo.)
3. Celda 1: `MODE = "calibrar"` → *Ejecutar todo*. Acepta el acceso a Drive. En ~5 min imprime una tabla de velocidades: cópiala.
4. Con la tabla se fija `utd`/`batch` en `configs/sac_max.yaml` (commit + push; el notebook siempre descarga lo último).
5. Celda 1: `MODE = "entrenar"` → *Ejecutar todo*. Deja la pestaña abierta. Si se corta: *Ejecutar todo* otra vez, reanuda solo.
6. Tras ~1 h, mira la celda 5 (reporte): velocidad real, ETA y evaluaciones.
7. Al terminar, la celda 4 verifica con `evaluar.py` y baja `agente.zip`.

**Dónde ver la ejecución:** en la **salida de la celda 3** del propio notebook (se actualiza sola, una línea cada 20 000 pasos). Cómo leerla:
`[ 120,000/26,500,000 | 98 sps | 2/5 obstaculos] dist=14.2 meta=0% | sim 15% upd 85% | 10.2 ms/upd | ETA 74 h`
- `sps` = pasos por segundo (lo que manda). `ETA` = horas que faltan de TODOS los stages a esa velocidad (el techo de pasos es mayor que lo que alcanzarás; no te asustes).
- `dist` / `meta` = distancia media y % de llegadas en entrenamiento.
- `EVAL @ …  validacion: meta 2/5 t=95s …  << NUEVO MEJOR` = prueba real con las pistas oficiales, cada 250 000 pasos. **Esto es lo que importa**: llegadas a meta (n/N) y tiempo.
- Kaggle: igual, pero si usas *Save Version → Run All* la salida se ve en la pestaña **Logs** de esa versión (no en el editor); el agente queda en *Output*.

**RESULTADO DE LA CALIBRACIÓN (medido):** Colab CPU (2 vCPU) = 22–28 sps con SAC → **inviable**. Kaggle T4 (4 vCPU) = 98 → 334 → 534 sps según variante. T4 y L4 rinden igual (98 sps con UTD 1). **Usar GPU T4.**

**Dónde está el modelo entrenado:**
- Colab: Drive → `Mi unidad/lab4_runs/<nombre>/` (`<nombre>` = `sac_max` salvo `name=` en OVERRIDES). Allí: `best_agent/` (**el que se entrega**: `policy.pt` + `meta.json`), `last_agent/`, `agente.zip` (al terminar), `eval_history.jsonl`, `metrics.jsonl`.
- Kaggle: pestaña **Output** de la versión (`agente.zip`), y `runs/<nombre>/best_agent/`.
- `best_agent/` se actualiza cada vez que una evaluación mejora (la primera a los 500 000 pasos): no hace falta esperar al final.
- **Sacar el modelo sin entrenar** (p. ej. mientras otra sesión sigue corriendo): en una sesión nueva, celda 1 con `MODE = "exportar"` y el mismo `CONFIG`/`OVERRIDES name=`; Ejecutar todo → verifica con `evaluar.py` y deja `agente.zip` (en Colab lo descarga).
- A mano: copiar `best_agent/policy.pt` y `meta.json` a la carpeta `agente/` del repo y correr `python evaluar.py`.

**Qué hardware elegir (histórico, antes de medir):** PPO (`ppo_curriculum`) → **CPU + RAM alta** (la GPU no lo acelera). SAC (`sac_max`) → **T4 + RAM alta** por defecto, pero **el CPU también es candidato**: con batch chico y poco UTD no queda lejos de una GPU (en un portátil de 8 núcleos: ~16 ms/actualización con batch 512 frente a ~10 ms en una L4), y el costo en CPU crece con el batch mientras que en GPU casi no. Esas cifras NO son de Colab: la velocidad del CPU de Colab es un dato pendiente.
**Cómo decidir (≈5 min por máquina):** ejecuta `MODE="calibrar"` en **T4** y en **CPU + RAM alta** (el benchmark usa el dispositivo que haya y muestra `cpus` en la línea `Entorno:`). Compara `sps` y `ms/upd` de la misma variante y elige la configuración (variante + máquina) con más `sps`. A100/H100/G4/TPU: no son necesarios. El CPU gasta bastantes menos unidades de Colab.

**¿T4 o L4?** El cuello de botella de SAC es la sobrecarga de cada actualización (Python + lanzar muchos kernels pequeños desde la CPU), no el cómputo de la GPU; por eso una GPU más potente casi no acelera, y la L4 gasta ~1,4× más unidades por hora que la T4 (≈1,71 vs ≈1,19 u/h, medición de mar-2026; ≈62 vs ≈43 u para 36 h). Por defecto **T4**. Se cambia a L4 solo con datos: ejecuta `MODE="calibrar"` en ambas y compara la columna `ms/upd` y `sps` de la misma variante; **usa L4 solo si es ≥1,3× más rápida** y te alcanzan las unidades. (La columna `GPU %` es solo orientativa: mide tiempo con algún kernel activo, no saturación.)
**Segunda corrida en paralelo:** otra pestaña/sesión (Colab CPU con PPO, o Kaggle) con `OVERRIDES = ["name=sac_max_s1", "seed=1"]` o `CONFIG = "configs/ppo_curriculum.yaml"`.

---

# Agente híbrido (entrega conjunta SAC + PPO)

**Idea:** los dos modelos del equipo son buenos en cosas distintas. SAC (`sac_max`, GPU) llega en ~87 % de las maratones **normales** y es rápido (~45 s), pero casi no pasa los obstáculos **hardcore**; PPO (`x_all3`, Mac) es el mejor en hardcore/hardcore-plus pero solo llega en ~61 % de las normales. La dificultad de una pista es fija y se nota en el terreno (escalones de 1,87 m, bloques cuadrados altos), así que `agente/` corre **SAC y cambia a PPO cuando un detector reconoce una pista hardcore**.

| Archivo | Qué es |
|---|---|
| `agente/policy_sac.pt` | SAC `sac_max` (24-400-300-4) |
| `agente/policy_ppo.pt` | PPO `x_all3` (24-256-256-4 + normalización; ver «Ruta PPO local») |
| `agente/detector.pt` | MLP 24-64-64-1: log-odds por paso de «pista hardcore» + regla `W`/`thr` |
| `agente/agent.py` | SAC hasta que la media del log-odds en los últimos `W`=25 pasos supera `thr`=3; desde ahí PPO hasta el final (un solo cambio, inmediato). Anti-atasco igual que antes |

```bash
python -m training.herramientas.detector datos runs/detector/sac.npz agente/policy_sac.pt --seed0 400001 --n 400   # 1200 carreras de SAC (~1 min)
python -m training.herramientas.detector entrenar runs/detector/sac.npz runs/detector/red.pt                       # imprime falsos positivos / distancia de detección
python -m training.herramientas.detector empaquetar runs/detector/red.pt agente/detector.pt --W 25 --thr 3
python -m training.herramientas.evaluar_agente . HIBRIDO --set test5 test6 --procs 8                             # agente/ completo, como el Evaluator
```
(Con esas semillas el detector se reconstruye bit a bit. Las semillas 400001+ son de entrenamiento: no se cruzan con los sets de evaluación.)

**Resultados** (pistas maratón nuevas × 6 semillas; cada agente con su `agent.py` real):
| | Total | Normal | Hardcore | Hardcore-plus | Tiempo/llegada |
|---|---|---|---|---|---|
| Elección de `W`/`thr` (`final+confirm+test3+test4`, 1680) | **744** (PPO 658, SAC 509) | 411/480 | 190/600 | 143/600 | 49,6 s |
| **Confirmación** (`test5`–`test11`, 4200, no usados) | **1629** | 716/840 | 532/1680 | 381/1680 | 51 s |
| PPO solo (entrega anterior) | 1488 | 509/840 | **565**/1680 | **414**/1680 | 55 s |
| SAC solo (`sac_max`) | 1015 | **721**/840 | 232/1680 | 62/1680 | **45 s** |

- El híbrido gana a PPO en los 7 sets de confirmación y a SAC por mucho. En normales iguala a SAC (cambia a PPO por error en ~7 % de las pistas normales); en hardcore pierde ~6 % frente a PPO solo (algunas caídas justo después del cambio). Si la pista del profesor es normal, gana ~+24 puntos; si es hardcore, cede ~2.
- Probado y descartado: esperar a tener ambos pies apoyados para cambiar, o mezclar acciones durante 10–30 pasos (633–729 vs 744: el cambio inmediato es mejor); ventanas/umbrales más conservadores (W100: 691–703).
- Públicas con `evaluar.py`: validación 4/5 (~42 s; cae en `maraton_01` a 204 m, igual que SAC solo), hardcore 2/5 (`maraton_03`, `maraton_04`). Con torch 2.2.2 las redes dan las mismas acciones (±1e-7) pero cada carrera puede cambiar por caos (360 carreras: 164 vs 177, dentro del ruido).

---

# Ruta PPO local (produce `agente/policy_ppo.pt`)

**Resumen:** PPO estable en un CPU local potente + afinados cortos + *model soups* (promedio de pesos de checkpoints), elegidos con miles de carreras sobre pistas maratón nuevas. Medido en un Mac M4 Pro (12 núcleos): ~7k pasos/s por run con 2–4 runs a la vez (SAC en T4: 98–534 sps).

### Linaje de `agente/policy.pt`
| # | Run (config) | Desde | Qué se toma |
|---|---|---|---|
| 1 | `ppo_v2d` (hereda `ppo_v2b` → `ppo_largo` → `ppo_curriculum`) | cero | checkpoint 31,5M → `<runs>/init_d31/best/` |
| 2 | `ppo_ft_hard` (lr 5e-5, solo maratones hardcore/hc-plus + obstáculos grandes) | `init_d31` | soup SB3 de 32,5–36,5M → `<runs>/init_soup/` |
| 3 | `ppo_ft_soup2` (lr 3e-5, otra semilla) | `init_soup` | soup SB3 de 35,5–39,5M → `<runs>/init_soupit2/` |
| 4 | 3 ramas de ~10M pasos: `ppo_it3_a` (misma mezcla), `ppo_it3_blq` (énfasis en bloques altos con pistas propias `obst_var`/`bloques_var` de `training/tracks_extra.py`), `ppo_it3_blq6` (como blq con lr 6e-5) | `init_soupit2` | 5 checkpoints por rama (41,5–45,5M) |
| 5 | **soup de las 3 ramas** (15 `policy.pt`) | — | **`agente/policy.pt`** |

Por qué funciona (interpretación): los bloques altos (≥1,2 escalones hardcore) causan ~la mitad de las caídas en maratones hardcore. Las tres ramas mejoran en bloques frente al modelo anterior, pero las de bloques empeoran en huecos; ninguna rama sola lo supera con claridad y el promedio de pesos de las tres sí.

### Comandos (desde la raíz del repo; `<runs>` = `runs/`, o `$LAB4_RUNS_DIR` si está definido)
```bash
python -m training.train --config configs/ppo_v2d.yaml            # cortar a mano tras el checkpoint 31.5M (SIGTERM, ver abajo; ojo con los nombres, ver nota)
mkdir -p runs/init_d31/best
cp runs/ppo_v2d/ckpt/step_31524948.zip           runs/init_d31/best/model.zip
cp runs/ppo_v2d/ckpt/step_31524948_obs_rms.pkl   runs/init_d31/best/obs_rms.pkl
python -m training.train --config configs/ppo_ft_hard.yaml         # hasta ~37M
python -m training.herramientas.soup_sb3 runs/init_soup --base runs/ppo_ft_hard/ckpt/step_34524948.zip runs/ppo_ft_hard/ckpt/step_3{2,3,4,5,6}524948.zip
python -m training.train --config configs/ppo_ft_soup2.yaml        # hasta ~40M
python -m training.herramientas.soup_sb3 runs/init_soupit2 runs/ppo_ft_soup2/ckpt/step_3{5,6,7,8,9}524948.zip
for c in ppo_it3_a ppo_it3_blq ppo_it3_blq6; do python -m training.train --config configs/$c.yaml & done; wait   # cortar tras 45.5M
python -m training.herramientas.soup_window runs/soup_ramas/policy.pt 41.5 45.5 runs/ppo_it3_blq runs/ppo_it3_blq6 runs/ppo_it3_a
cp runs/soup_ramas/policy.pt agente/policy.pt                    # solo si gana en la re-evaluación (abajo)
```
- Verificado con los checkpoints originales: estos comandos reconstruyen **bit a bit** `init_soup` e `init_soupit2` (pesos, normalización, pasos, optimizador) y `agente/policy.pt`. El orden importa: el promedio en otro orden cambia ~1e-7 por redondeo, y en `soup_sb3` el primer zip (o `--base`) fija `num_timesteps`, optimizador y semilla del resultado (en `init_soup` fue el de 34,5M, así que `ppo_ft_soup2` siguió desde 34,5M).
- Reentrenar desde cero **no** da los mismos pesos (dinámica caótica, varios procesos; además `ppo_ft_hard` corrió antes del parche de `seed`). El soup de ramas **varía mucho entre tiradas**: repetir el paso 4 con otras semillas/ramas dio 535–629 llegadas (réplica exacta: 579) frente a 658 en las mismas 1680 carreras.
- **Nombres de checkpoint:** `ppo_v2d` original se pausó a 6,52M y se reanudó; como guarda cada 5M desde el último guardado, sus checkpoints quedaron en `…_524948` y todos los runs siguientes heredan ese desfase (con `init_from`, los pasos continúan desde los del modelo inicial). Un `ppo_v2d` nuevo guardará en 5M, 10M, …: usar p. ej. `--set checkpoint.every_steps=1000000`, tomar el checkpoint más cercano a 31,5M y ajustar los nombres de los comandos siguientes. Con `n_envs` que no divide 1M (p. ej. 3) los nombres derivan unos pasos por millón: `soup_window` toma el más cercano.
- Con `init_from`, los `timesteps` de los stages son **absolutos** (incluyen los pasos del modelo inicial): por eso los afinados usan un tope de 81524948. Si el tope es menor que los pasos del modelo inicial, el run termina sin entrenar.
- `init_from` aplica `learning_rate`, `ent_coef`, `target_kl`, `clip_range`, `n_epochs` y `seed` del YAML nuevo (parche en `train.py`; antes SB3 restauraba los guardados y dos semillas daban checkpoints idénticos). `n_steps`, `batch_size`, `gamma` siguen siendo los guardados.
- `learning_rate: lin_X` ⇒ lr = X·max(0,1; 1 − pasos/fin_del_stage), con pasos absolutos: baja dentro de cada stage y **vuelve a subir** al empezar el siguiente.

### Elegir qué se entrega (`training.herramientas.holdout`)
El `EVAL` interno (10 carreras, 1 semilla) **no distingue candidatos**. La elección se hace con pistas maratón 240 m **nuevas** × 6 semillas, criterio del curso:
```bash
python -m training.herramientas.holdout runs/ppo_it3_a --set final --seeds 5 --procs 8 --only step_41 step_42   # checkpoints de un run
python -m training.herramientas.holdout runs/comparacion --set final --seeds 5 --procs 8 \
    --ensemble NUEVO=runs/soup_ramas/policy.pt ACTUAL=agente/policy.pt --only NUEVO ACTUAL           # policy.pt sueltos
```
- Sets: `final`, `confirm`, `test3` (60 pistas: 20 normal / 20 hardcore / 20 hardcore-plus = 360 carreras) y `test4`…`test11` (100 pistas: 20/40/40 = 600 carreras). La entrega se eligió con `final+confirm+test3+test4` (658 vs 602 del modelo anterior) y se confirmó con `test5`…`test11` (ya usados: para seguir, añadir sets con semillas nuevas en `holdout.py`).
- Ruido: el mismo modelo varía ±10–15 % entre sets de 360 carreras; dos modelos casi iguales difieren ~±25 llegadas en 1680. Diferencias menores no significan nada.
- Reproduce al `Evaluator` oficial (mismas pistas y semillas ⇒ mismas carreras) con el `policy.pt` tal cual, **sin** el anti-atasco de `agent.py`. Cachea en `<run_dir>/holdout_<set>/`.

### Resultado medido (`test5`–`test11`: 700 pistas × 6 semillas = 4200 carreras, no usados para elegir entre estos dos modelos)
| | soup anterior (`ppo_ft_soup2` 35,5–39,5M) | **entrega actual** |
|---|---|---|
| Llegadas totales | 1296 | **1472** (+13,6 %, gana en los 7 sets) |
| Hardcore + hardcore-plus (3360) | 777 (23 %) | **963 (29 %)** |
| Normal (840) | 519 | 509 (empate) |
| Tiempo medio por llegada | ~57,7 s | **~55 s** |

Cifras de `policy.pt` sin anti-atasco en ambos modelos. `test5`–`test9` sí se usaron para descartar otras alternativas; en `test10`–`test11`, nunca tocados antes: 388 vs 352. (`agente/meta.json` quedó con la confirmación de 5 sets, la disponible al empaquetar.)

Con una pista y 6 semillas (el formato oficial), la entrega gana a la anterior en ~65–70 % de las pistas hardcore; en pistas normales es casi una moneda al aire.

### `agente/agent.py`: anti-atasco
La política determinista a veces se queda quieta frente a un hueco hasta agotar el tiempo (~5 % de las carreras con la entrega). Si `|obs[2]|` (velocidad horizontal) < 0,02 durante 100 pasos, se suma ruido N(0, 0,3) a las acciones durante 30 pasos (RNG con semilla fija reiniciada en `reset()`, así que es determinista). Medido con la entrega en los 11 sets (5880 carreras): +23 llegadas y −2, neto +21 (≈ +1 %). `evaluar.py` verificado con torch 2.15 y 2.2.2.
Ojo: la carrera por defecto de `python evaluar.py` (`validacion/maraton_01`) la entrega no la termina (se queda a ~191 m por tiempo con torch 2.15); son resultados de una sola carrera, la selección se hizo con miles.

### Lo que NO funcionó (medido en miles de carreras salvo los ensembles, 240–360)
Ensembles de acciones de políticas distintas; afinar desde un soup ya promediado (lr 3e-5 y 1e-5: siempre empeora); soups grandes de 7–14 ramas (empatan o pierden); interpolación con el modelo anterior (WiSE-FT); cambiar a una política especialista en bloques cerca de un bloque alto; ventanas de soup más tardías (+10…+14M empeoran).

---


Idea: **el código vive en el repo; el notebook de la nube solo lo lanza.** Iterar = editar `.py`/YAML → `git push` → *Run all*.

```
lab4/
├─ walker_lab/ escenarios/ evaluar.py …   ← material del profesor (NO se edita)
├─ training/                              ← nuestro pipeline
│   ├─ train.py        orquesta: YAML → curriculum → entrenar → eval → checkpoint → export
│   ├─ envs.py         mezcla de perfiles por episodio (+ VecNormalize opcional)
│   ├─ tracks_extra.py pistas de entrenamiento propias (obst_var, bloques_var: bloques altos de escala continua)
│   ├─ callbacks.py    eval periódica sobre pistas reales, best/last, progreso, límite de tiempo
│   ├─ checkpoints.py  modelo + replay buffer + normalización (reanudable)
│   ├─ export.py       política SB3 → MLPPolicy (verifica que da las mismas acciones) → policy.pt
│   ├─ config.py       YAML con `extends` y `--set clave=valor`
│   ├─ runtime.py      detecta Colab/Kaggle/local y dónde guardar los runs
│   └─ herramientas/   holdout.py · evaluar_agente.py (re-evaluar en pistas nuevas) · detector.py (híbrido) · soup*.py
├─ configs/            base.yaml (curriculum + eval) · sac_curriculum.yaml · sac_max.yaml · ppo_curriculum.yaml · smoke*.yaml
│                      ppo_largo → ppo_v2b → ppo_v2d → ppo_ft_d31 → ppo_ft_hard → ppo_ft_soup2 / ppo_it3_* (linaje de la entrega)
├─ notebooks/train_cloud.ipynb            ← 4 celdas, idéntico en Colab y Kaggle
├─ agente/                                ← ENTREGA (no cambia entre experimentos, salvo policy.pt)
│   ├─ agent.py  architecture.py          híbrido SAC→PPO (+ anti-atasco) + MLP en PyTorch puro (API estable, sin TorchScript)
│   └─ policy_sac.pt policy_ppo.pt detector.pt meta.json
└─ runs/<nombre>/      (ignorado por git)
    ├─ config.resolved.yaml  meta.json    reproducibilidad (config final + commit + versiones)
    ├─ logs/ (csv + tensorboard)  eval_history.jsonl
    ├─ ckpt/                              reanudación
    └─ best_agent/  last_agent/           policy.pt + meta.json
```

## Flujo diario

| Quiero… | Hago |
|---|---|
| Probar que todo funciona | `python -m training.train --config configs/smoke.yaml --fresh` |
| Un experimento nuevo | copiar un YAML en `configs/` (hereda con `extends`), cambiar lo que difiere |
| Cambiar un hiperparámetro sin editar archivos | `--set algo_kwargs.learning_rate=0.0001` (o `OVERRIDES` en el notebook) |
| Cambiar de algoritmo | `configs/ppo_curriculum.yaml` (SAC/TD3/PPO soportados) |
| Seguir donde quedó (sesión cortada) | volver a ejecutar lo mismo: detecta `ckpt/` y reanuda |
| Afinar desde el mejor de otro run | `init_from: sac_v1` (lee `<runs>/sac_v1/best/model.zip` + `obs_rms.pkl`; aplica lr, ent_coef, target_kl, clip_range, n_epochs y seed del YAML nuevo) |
| Entregar | elegir con `python -m training.herramientas.holdout` (ver «Ruta PPO local»), copiar el `policy.pt` ganador a `agente/` y `python evaluar.py`. `training.export runs/<nombre> --to agente` copia solo `best_agent` |

## Qué se mide (criterio del ranking del curso)
El profesor evalúa 6 carreras sobre **una pista nueva** («empieza normal y se complica mucho»; probablemente el perfil maratón de 240 m), cada una con una semilla de condición inicial distinta, y ordena por: **1) nº de llegadas a meta**, 2) **menor tiempo medio** de las completadas, 3) mayor distancia media. El pipeline elige el *best* con ese mismo orden (`training/callbacks.py::rank_key`) sobre las suites de `eval.select_on` (por defecto `validacion` + `hardcore`). Ese *best* interno (10 carreras) es solo orientativo: la entrega se elige re-evaluando candidatos en pistas nuevas (ver «Ruta PPO local»).
La política evaluada es **exactamente el `policy.pt` que se entrega** (red + normalización).

## Resistencia a fallos (todo probado con fallos reales, ver tabla)
| Fallo | Qué pasa | Probado |
|---|---|---|
| Sesión muere / `kill -9` a mitad de entrenamiento | Se vuelve a ejecutar lo mismo y retoma del último checkpoint (modelo + replay buffer + normalización) | ✅ taskkill /F a mitad de run |
| Checkpoint más reciente corrupto (VM muere escribiendo en Drive) | Cae al anterior (se conservan 2) y avisa | ✅ archivo truncado |
| Replay buffer ilegible | Sigue con buffer vacío en vez de abortar | código |
| Falla la evaluación o el guardado (Drive caído, disco lleno) | Avisa y el entrenamiento continúa | ✅ fallos inyectados |
| `SIGTERM` / Ctrl+C | Guarda checkpoint antes de salir. **A veces el proceso se cuelga después** al cerrar `SubprocVecEnv` (visto en local): esperar `>>> Interrumpido: guardando checkpoint.` en el log y, si no aparece `FIN.` en ~1 min, `kill -9 <pid>` (el checkpoint ya está escrito) | código / ⚠ cuelgue al cerrar |
| `--fresh` / `FRESH=True` olvidado | Archiva el run anterior (`nombre.old-fecha`), no lo borra | ✅ |
| Cortes por límite de sesión | `time_limit_hours` sale limpio antes del límite | código |
| Print con símbolos en consola cp1252 | No tumba el entrenamiento | ✅ (bug encontrado y corregido) |

**No cubre:** reiniciar la sesión automáticamente (hay que dar Run all de nuevo); Colab sin Drive montado (avisa, pero los datos son efímeros); Kaggle solo conserva `/kaggle/working` al terminar con *Save Version*.

## Métricas (mediciones realistas siempre)
- Cada `progress_every` pasos: `[paso | sps | stage] dist meta | sim% upd% eval% | ms/upd | ETA`. `sim` = simulador + inferencia, `upd` = gradientes.
- `runs/<nombre>/metrics.jsonl`: lo mismo en JSON + UTD real, pérdidas, `ent_coef`, uso de GPU/CPU/RAM.
- `python -m training.benchmark`: prueba 6 configuraciones de SAC en la máquina real y estima horas por 10M pasos (celda 2b del notebook).
- `python -m training.report <run>`: velocidad estable, reparto de tiempo, proyección a cada stage y tabla de evaluaciones. **Pegar su salida basta para estimar con datos reales.**
- `python -m training.compare runs/*`: ranking de runs con el criterio del curso (`--promote agente` copia el ganador).
- `utd:` en el YAML fija las actualizaciones por transición sin depender de `n_envs`.

## GPU
Cuello de botella = simulador Box2D (1 núcleo por entorno, ~5k pasos/s) y, en SAC, las actualizaciones de la red (MLP de 256x256, <1 GB de VRAM, 1 GPU cualquiera basta; una más grande no acelera). PPO va igual o mejor en CPU (el pipeline lo fuerza). Mida los `sps` reales en la línea de progreso del entrenamiento.

## Notas
- Verificado en local (SAC y PPO+normalización): el `evaluar.py` oficial reproduce la distancia medida durante el entrenamiento.
- Colab/Kaggle traen `torch`; `agente/requirements.txt` lista lo que añade la entrega (fijar la versión usada, ver `meta.json`).
- SAC con `gradient_steps: -1` hace 1 actualización por transición (lo más eficiente en muestras, pero lento en reloj). Con poco tiempo, `gradient_steps: 1` y más `n_envs` sube mucho los pasos/segundo.
- Hiperparámetros por algoritmo viven en cada config (los dicts se fusionan al heredar). `suites: {hardcore: null}` desactiva una suite heredada.
- Reanudar sin replay buffer (`checkpoint.save_replay_buffer: false`) es válido, pero el buffer empieza vacío.
- Las duraciones/pasos de `base.yaml` son un punto de partida razonable, no un resultado medido: ajustar según la curva de `eval_history.jsonl`.
