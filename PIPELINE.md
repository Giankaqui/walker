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


Idea: **el código vive en el repo; el notebook de la nube solo lo lanza.** Iterar = editar `.py`/YAML → `git push` → *Run all*.

```
lab4/
├─ walker_lab/ escenarios/ evaluar.py …   ← material del profesor (NO se edita)
├─ training/                              ← nuestro pipeline
│   ├─ train.py        orquesta: YAML → curriculum → entrenar → eval → checkpoint → export
│   ├─ envs.py         mezcla de perfiles por episodio (+ VecNormalize opcional)
│   ├─ callbacks.py    eval periódica sobre pistas reales, best/last, progreso, límite de tiempo
│   ├─ checkpoints.py  modelo + replay buffer + normalización (reanudable)
│   ├─ export.py       política SB3 → MLPPolicy (verifica que da las mismas acciones) → policy.pt
│   ├─ config.py       YAML con `extends` y `--set clave=valor`
│   └─ runtime.py      detecta Colab/Kaggle/local y dónde guardar los runs
├─ configs/            base.yaml (curriculum + eval) · sac_curriculum.yaml · ppo_curriculum.yaml · smoke.yaml
├─ notebooks/train_cloud.ipynb            ← 4 celdas, idéntico en Colab y Kaggle
├─ agente/                                ← ENTREGA (no cambia entre experimentos, salvo policy.pt)
│   ├─ agent.py  architecture.py          adaptador + MLP en PyTorch puro (API estable, sin TorchScript)
│   └─ policy.pt  meta.json               pesos + normalización (los genera training.export)
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
| Afinar desde el mejor de otro run | `init_from: sac_v1` |
| Entregar | `python -m training.export runs/<nombre> --to agente` y luego `python evaluar.py` |

## Qué se mide (criterio del ranking del curso)
El profesor evalúa 6 carreras y ordena por: **1) nº de llegadas a meta**, 2) **menor tiempo medio** de las completadas, 3) mayor distancia media. El pipeline elige el *best* con ese mismo orden (`training/callbacks.py::rank_key`) sobre las suites de `eval.select_on` (por defecto `validacion` + `hardcore`). Se desconoce de qué pistas saldrán las 6 carreras: si lo averiguan, ajusten `select_on`.
La política evaluada es **exactamente el `policy.pt` que se entrega** (red + normalización).

## Resistencia a fallos (todo probado con fallos reales, ver tabla)
| Fallo | Qué pasa | Probado |
|---|---|---|
| Sesión muere / `kill -9` a mitad de entrenamiento | Se vuelve a ejecutar lo mismo y retoma del último checkpoint (modelo + replay buffer + normalización) | ✅ taskkill /F a mitad de run |
| Checkpoint más reciente corrupto (VM muere escribiendo en Drive) | Cae al anterior (se conservan 2) y avisa | ✅ archivo truncado |
| Replay buffer ilegible | Sigue con buffer vacío en vez de abortar | código |
| Falla la evaluación o el guardado (Drive caído, disco lleno) | Avisa y el entrenamiento continúa | ✅ fallos inyectados |
| `SIGTERM` / Ctrl+C | Guarda checkpoint antes de salir | código |
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
