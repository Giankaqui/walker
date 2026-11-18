# AGENTS.md — Lab 4 (RL, BipedalWalker) · léelo primero

Documento para cualquier agente o IA que trabaje en este proyecto. Idioma del proyecto: **español**.
(El `README.md` de esta carpeta es del profesor y no se edita; este archivo es el de trabajo.)

## 1. Qué es esto
Laboratorio 4 del curso de Deep Learning (UTEC): entrenar un agente de **aprendizaje por refuerzo** para un bípedo (`walker_lab.WalkerEnv`, basado en BipedalWalker de Gymnasium) y entregar la carpeta `agente/`.

- **Entrada:** `float32(24)` (torso, articulaciones, contactos, 10 lecturas lidar). **Salida:** `float32(4)` en [-1, 1].
- **Entrega:** carpeta `agente/` con código de inferencia + pesos + normalización. Antes de enviarla se debe comprobar que `python evaluar.py` la carga y ejecuta una carrera.
- **Cómo se califica** (6 carreras, mismas condiciones): 1) **más llegadas a meta**, 2) a igualdad, **menor tiempo medio** de las completadas, 3) a igualdad, **mayor distancia media**. Las pistas exactas de esas carreras **se desconocen** (hay ejemplos en `escenarios/validacion/` y `escenarios/hardcore/`).
- **Plazo indicado:** 20-nov-2026 en clase. Confirmar la hora límite real y dejar ≥1 h de margen para exportar y verificar.
- **Repo:** https://github.com/Vasco2510/walker (rama `main`).

## 2. Estructura
```
walker_lab/ escenarios/ evaluar.py visualizar.py reproducir.py generar_escenario.py comprobar_instalacion.py
requirements.txt README.md          ← MATERIAL DEL PROFESOR: SOLO LECTURA
agente/            ← ENTREGA: agent.py + architecture.py (fijos) y policy.pt + meta.json (los genera training.export)
training/          ← pipeline propio (ver abajo)
configs/           ← YAML de experimentos (base.yaml, sac_max, sac_curriculum, sac_cpu, ppo_curriculum, smoke*)
notebooks/train_cloud.ipynb   ← lanzador fino para Colab/Kaggle
PIPELINE.md        ← guía de uso detallada (paso a paso, métricas, resistencia a fallos)
runs/<nombre>/     ← resultados (git los ignora; en Colab viven en Drive)
```
`training/`: `train.py` (orquesta) · `envs.py` (mezcla de perfiles por episodio) · `callbacks.py` (progreso con desglose de tiempo, evaluación, `rank_key`) · `checkpoints.py` · `export.py` (SB3 → `MLPPolicy`, verificado) · `benchmark.py` · `report.py` · `compare.py` · `config.py` (YAML con `extends`, `--set a.b=c`) · `runtime.py` (plataforma, rutas, métricas de sistema).

## 3. Filosofía de trabajo (política central)
**El código vive en el repo; el notebook solo lo lanza.** Iterar = editar `.py`/YAML → commit/push → `Run all` en la nube. Ninguna lógica nueva va dentro del notebook. Los hiperparámetros se cambian por YAML o por `--set` / `OVERRIDES`, no editando código.

## 4. Políticas (obligatorias)
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
- **Local (solo humo):** `python -m training.train --config configs/smoke.yaml --fresh` (minutos, CPU). El equipo local es poco potente: **no entrenar en serio ahí**.
- **Colab Pro (principal):** abrir `notebooks/train_cloud.ipynb` desde GitHub, runtime **L4 + RAM alta** para SAC. Primera vez `MODE="calibrar"` → Run all; luego `MODE="entrenar"` → Run all. Si se corta, Run all otra vez (reanuda).
- **Paralelo:** otra sesión (Kaggle, o Colab CPU RAM alta con `configs/ppo_curriculum.yaml`) o segunda semilla (`OVERRIDES=["name=sac_max_s1","seed=1"]`). Al final: `python -m training.compare <runs>/* --promote agente`.
- **Entrega:** `python -m training.export runs/<nombre> --to agente` → `python evaluar.py` → zip de `agente/`.

## 6. Verificación (hacer tras tocar `training/` o `agente/`)
Entorno de prueba: Python 3.11 + `pip install -r requirements-train.txt` + torch. Probar con configs `smoke*.yaml` en un directorio de runs aparte (`LAB4_RUNS_DIR`):
1. Entrenar → evaluar → exportar → `evaluar.py` oficial reproduce la distancia medida en entrenamiento (SAC y PPO+normalización).
2. Reanudar tras `taskkill /F` a mitad de run.
3. Reanudar con el checkpoint más reciente truncado (debe caer al anterior).
4. Inyectar fallo en evaluación y guardado (debe terminar igual).
5. `--fresh` archiva en vez de borrar.

## 7. Estado y hechos conocidos
- **Medido en Colab L4 (log del usuario, Python 3.13, código antiguo):** SAC estándar (UTD 1, batch 512, 256×256) ≈ **98 sps**. A ese ritmo 26M pasos ≈ 73 h → los 26M de `sac_max` son un **techo**, no un plan; el límite real son las horas (≈36 h de entrenamiento previstas: mié–vie, sesiones de ~12 h).
- **Medido en local (CPU, equipo débil):** ~85 % del tiempo va a las actualizaciones de SAC, ~15 % al simulador; PPO ≈ 20× más pasos/s que SAC UTD 1 en CPU. **No medido:** efecto real de UTD/batch en GPU (para eso es `training.benchmark`).
- `utd`/`batch_size` de `configs/sac_max.yaml` son **provisionales** hasta calibrar.
- El pipeline ya corre en Colab (L4); el aviso `pkg_resources` de pygame es inofensivo.
- Preferencia del usuario: ambición alta, flujo simple (un `Run all`), instrucciones claras y paso a paso.
