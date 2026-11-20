# AGENTS.md — Lab 4 (RL, BipedalWalker) · léelo primero

Documento para cualquier agente o IA que trabaje en este proyecto. Idioma del proyecto: **español**.
(El `README.md` de esta carpeta es del profesor y no se edita; este archivo es el de trabajo.)

## 1. Qué es esto
Laboratorio 4 del curso de Deep Learning (UTEC): entrenar un agente de **aprendizaje por refuerzo** para un bípedo (`walker_lab.WalkerEnv`, basado en BipedalWalker de Gymnasium) y entregar la carpeta `agente/`.

- **Entrada:** `float32(24)` (torso, articulaciones, contactos, 10 lecturas lidar). **Salida:** `float32(4)` en [-1, 1].
- **Entrega:** carpeta `agente/` con código de inferencia + pesos + normalización. Antes de enviarla se debe comprobar que `python evaluar.py` la carga y ejecuta una carrera.
- **Cómo se califica:** 6 carreras sobre **una pista nueva** («empieza normal y se complica mucho», según el profesor; probablemente el perfil maratón de 240 m), cada una con una semilla de condición inicial distinta. Orden: 1) **más llegadas a meta**, 2) a igualdad, **menor tiempo medio** de las completadas, 3) a igualdad, **mayor distancia media**. La pista exacta y su dificultad se desconocen (hay ejemplos en `escenarios/validacion/` y `escenarios/hardcore/`: no optimizar contra ellos).
- **Plazo indicado:** 20-nov-2026 en clase. Confirmar la hora límite real y dejar ≥1 h de margen para exportar y verificar.
- **Repo:** https://github.com/Vasco2510/walker (rama `main`).

## 2. Estructura
```
walker_lab/ escenarios/ evaluar.py visualizar.py reproducir.py generar_escenario.py comprobar_instalacion.py
requirements.txt README.md          ← MATERIAL DEL PROFESOR: SOLO LECTURA
agente/            ← ENTREGA: agent.py (híbrido SAC→PPO + anti-atasco) + architecture.py, policy_sac.pt, policy_ppo.pt, detector.pt, meta.json
training/          ← pipeline propio (ver abajo)
configs/           ← YAML de experimentos (base.yaml · sac_max = SAC en GPU · ppo_v2d → ppo_ft_hard → ppo_ft_soup2 → ppo_it3_* = linaje PPO de la entrega · ppo_curriculum = PPO original, inestable · sac_curriculum = base de smoke · smoke*)
notebooks/train_cloud.ipynb   ← lanzador fino para Colab/Kaggle
PIPELINE.md        ← guía de uso detallada (paso a paso, métricas, resistencia a fallos)
runs/<nombre>/     ← resultados (git los ignora; en Colab viven en Drive)
```
`training/`: `train.py` (orquesta; `init_from` aplica lr/ent_coef/target_kl/clip_range/n_epochs/seed del YAML; `learning_rate: lin_X`) · `envs.py` (mezcla de perfiles por episodio) · `callbacks.py` (progreso con desglose de tiempo, evaluación, `rank_key`) · `checkpoints.py` · `export.py` (SB3 → `MLPPolicy`, verificado) · `benchmark.py` · `report.py` · `compare.py` · `config.py` (YAML con `extends`, `--set a.b=c`) · `runtime.py` (plataforma, rutas, métricas de sistema) · `tracks_extra.py` (pistas propias de entrenamiento: bloques altos de escala continua) · `herramientas/` (`holdout.py` = re-evaluar candidatos en pistas nuevas con el criterio del curso; `soup*.py` = model soups; `detector.py` = detector del híbrido; `evaluar_agente.py` = evaluar una carpeta agente/ completa). Detalle en PIPELINE.md, «Ruta PPO local».

## 3. Filosofía de trabajo (política central)
**El código vive en el repo; el notebook solo lo lanza.** Iterar = editar `.py`/YAML → commit/push → `Run all` en la nube. Ninguna lógica nueva va dentro del notebook. Los hiperparámetros se cambian por YAML o por `--set` / `OVERRIDES`, no editando código.

## 4. Políticas (obligatorias)
0. **Mantener simple.** Antes de añadir un archivo, config, herramienta o variable, preguntarse si basta con un `--set`/`OVERRIDES`. Preferir borrar a acumular. Una sola forma de hacer cada cosa; documentación solo en `AGENTS.md` (política) y `PIPELINE.md` (uso).
1. **No modificar el material del profesor** (`walker_lab/`, `escenarios/`, `evaluar.py`, etc.). Todo lo nuestro va en `training/`, `configs/`, `agente/` (solo `policy.pt`, `meta.json`), `notebooks/`.
2. **Selección del mejor modelo = criterio del ranking del curso** (`training/callbacks.py::rank_key`), nunca solo distancia. Se evalúa con la política determinista **exacta que se entrega** (`MLPPolicy`), no con la de SB3.
3. **Mediciones antes que estimaciones.** No afirmar velocidades, tiempos o mejoras sin medirlas (`training.benchmark`, `training.report`, `metrics.jsonl`). Marcar explícitamente lo no medido. Dar proyecciones solo a partir de `sps` real.
4. **Nunca perder entrenamiento:** `--fresh` archiva el run (`nombre.old-fecha`), no borra. No borrar `runs/` ni checkpoints sin que el usuario lo pida. Un run existente no puede cambiar su UTD/batch al reanudar (SB3 conserva los hiperparámetros guardados): para cambiarlos, run con **nombre nuevo**.
5. **Resistencia a fallos es requisito:** un fallo de evaluación o de guardado no debe interrumpir el entrenamiento; checkpoints reanudables (conservar ≥2); SIGTERM guarda. Todo cambio en `training/` debe pasar las pruebas de la sección 6.
6. **Dependencias:** el simulador está fijado por versión en `requirements.txt` (no tocar). Lo nuestro va en `requirements-train.txt`. La inferencia entregada solo depende de `torch` (API estable; **sin TorchScript**, está deprecado). `policy.pt` se carga con `weights_only=True`.
7. **Git:** autor = el usuario (Vasco2510). **Los commits NO llevan firma ni `Co-Authored-By` de ninguna IA.** Mensajes en español, descriptivos. Push normal a `main` está autorizado para este flujo; **nunca `--force`** ni reescribir historial sin que el usuario lo pida explícitamente. Nunca subir `runs/`, `resultados/` ni `*.zip`.
8. **Privacidad:** no incluir datos personales del usuario en llamadas a servicios externos.
9. **Acciones con costo o externas** (alquilar GPUs, pagar, publicar) requieren confirmación del usuario.

## 5. Cómo se corre (resumen; detalle en PIPELINE.md)
- **Local (humo):** `python -m training.train --config configs/smoke.yaml --fresh` (minutos, CPU). En un portátil débil, solo humo.
- **Local potente (PPO, lo que produjo la entrega):** en un Mac M4 Pro, PPO corre a ~7k pasos/s por run (2–4 runs a la vez). Receta completa, comandos y selección en PIPELINE.md, «Ruta PPO local».
- **Colab Pro (principal):** abrir `notebooks/train_cloud.ipynb` desde GitHub, runtime **T4 + RAM alta** para SAC (L4 solo si `MODE="calibrar"` la muestra ≥1,3× más rápida; el cuello de botella no es el cómputo de GPU). Primera vez `MODE="calibrar"` → Run all; luego `MODE="entrenar"` → Run all. Si se corta, Run all otra vez (reanuda).
- **Paralelo:** otra sesión (Kaggle, o Colab CPU RAM alta con `configs/ppo_curriculum.yaml`) o segunda semilla (`OVERRIDES=["name=sac_max_s1","seed=1"]`). Al final: `python -m training.compare <runs>/* --promote agente`.
- **Entrega:** `python -m training.export runs/<nombre> --to agente` → `python evaluar.py` → zip de `agente/`. En la nube: `MODE="exportar"` en el notebook hace lo mismo sin entrenar (el mejor agente está en `<runs>/<nombre>/best_agent/`).

## 6. Verificación (hacer tras tocar `training/` o `agente/`)
Entorno de prueba: Python 3.11 + `pip install -r requirements-train.txt` + torch. Probar con configs `smoke*.yaml` en un directorio de runs aparte (`LAB4_RUNS_DIR`):
1. Entrenar → evaluar → exportar → `evaluar.py` oficial reproduce la distancia medida en entrenamiento (SAC y PPO+normalización).
2. Reanudar tras `taskkill /F` a mitad de run.
3. Reanudar con el checkpoint más reciente truncado (debe caer al anterior).
4. Inyectar fallo en evaluación y guardado (debe terminar igual).
5. `--fresh` archiva en vez de borrar.

## 7. Estado y hechos conocidos
- **Entrega actual (`agente/`, 19-nov noche): agente híbrido** = SAC `sac_max` + detector de pista hardcore + PPO `x_all3` (PIPELINE.md, «Agente híbrido»). En 7 sets de pistas maratón nuevas (4200 carreras, `agent.py` real): 1629 llegadas vs 1488 del PPO solo y 1015 del SAC solo; normal 716/840, hardcore+plus 913/3360; ~51 s por llegada. `evaluar.py` verificado con torch 2.15 y 2.2.2.
- **Medido (PPO local):** con los hiperparámetros de `ppo_curriculum.yaml` (run `ppo_largo`) es inestable (approx_kl ~0,3, clip_fraction ~0,7, 0/10 llegadas a 24M). Estable con n_steps 2048, batch 2048, target_kl 0.03, ent_coef 0.001, gamma 0.995 (`ppo_v2b`/`ppo_v2d`). Afinar: lr 3e-5–5e-5, target_kl 0.01–0.015.
- **Selección:** el `EVAL` interno (10 carreras) no distingue checkpoints; elegir con `training/herramientas/holdout.py` (miles de carreras, varios sets) y confirmar en sets no usados. Ruido ≈ ±25 llegadas en 1680 carreras.
- **SIGTERM** guarda, pero a veces el proceso se cuelga al cerrar `SubprocVecEnv`: `kill -9` tras ver el checkpoint en el log.
- **Medido (calibración):** Colab CPU (2 vCPU): SAC 22–28 sps (inviable). Kaggle T4 (4 vCPU, `n_envs=4`): UTD1/b512 = 98 sps; UTD.5/b1024 = 186; UTD.25/b2048 = 334; UTD.125/b4096 = 534; la red 400×300 no cuesta velocidad. Una L4 (log del usuario) dio los mismos 98 sps que la T4 con UTD 1: **T4 basta**. UTD×batch=512 en todas las variantes (mismo reuso de datos por transición); el efecto en la CALIDAD final no está medido.
- **Plan de corridas:** A = Colab Pro T4 con `configs/sac_max.yaml` (UTD .25, b2048; ≈22 h, 2 sesiones con reanudación en Drive). B = Kaggle T4 con `utd=0.125`, `batch_size=4096`, `seed=1`, `time_limit_hours=11` (cabe en UNA sesión; el pipeline no reanuda entre sesiones de Kaggle). Elegir el ganador con `training.compare`.
- **Medido en Colab L4 (log del usuario, Python 3.13, código antiguo):** SAC estándar (UTD 1, batch 512, 256×256) ≈ **98 sps**. A ese ritmo 26M pasos ≈ 73 h → los 26M de `sac_max` son un **techo**, no un plan; el límite real son las horas (≈36 h de entrenamiento previstas: mié–vie, sesiones de ~12 h).
- **Medido en local (CPU, equipo débil):** ~85 % del tiempo va a las actualizaciones de SAC, ~15 % al simulador; PPO ≈ 20× más pasos/s que SAC UTD 1 en CPU. **No medido:** efecto real de UTD/batch en GPU (para eso es `training.benchmark`).
- `utd`/`batch_size` de `configs/sac_max.yaml` son **provisionales** hasta calibrar.
- El pipeline ya corre en Colab (L4); el aviso `pkg_resources` de pygame es inofensivo.
- Preferencia del usuario: ambición alta, flujo simple (un `Run all`), instrucciones claras y paso a paso.
