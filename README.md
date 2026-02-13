# RL and Generative Techniques

Repositorio de prácticas/proyectos de Aprendizaje por Refuerzo con dos líneas principales:

1. **Neuroevolución de MLP** con DEAP en entornos Gymnasium (p. ej. LunarLander y FlappyBird).
2. **RL tabular** con **Q-Learning**, **SARSA** y **Monte Carlo** (incluyendo discretización para observaciones continuas).

## Requisitos

- Python **>= 3.13**
- [uv](https://docs.astral.sh/uv/)

Instalación:

```bash
uv sync
```

## Ejecución rápida

### 1) Neuroevolución (DEAP + MLP)

```bash
uv run main.py
```

- El script carga por defecto `configurations/config_fb.yaml`.
- Para cambiar experimento, modifica la variable `file` en `main.py` o añade un YAML nuevo en `configurations/`.

### 2) RL tabular (Q-Learning / SARSA / Monte Carlo)

```bash
uv run q_main.py
```

- Selecciona algoritmo y entorno en el bloque final de `q_main.py` (`ALGORITHM`, `ENV_NAME`, etc.).
- Genera métricas, gráfica de entrenamiento y tabla Q serializada.

## Estructura del proyecto

- `main.py`: punto de entrada de neuroevolución.
- `q_main.py`: punto de entrada de RL tabular.
- `agents_rl/`:
	- `base.py`: `BaseRLAgent` + `ObservationDiscretizer`.
	- `q_agent.py`: implementación Q-Learning (off-policy).
	- `sarsa_agent.py`: implementación SARSA (on-policy).
	- `mc_agent.py`: implementación Monte Carlo first-visit.
- `neuroevolution/`:
	- `experiment.py`: ejecución completa del experimento evolutivo.
	- `config_exp.py`: dataclasses/enums de configuración y mapeo a operadores DEAP.
	- `config_manager.py`: utilidades para guardar/cargar/comparar configuraciones.
	- `MLP.py`: red MLP, codificación/decodificación cromosoma y visualización de red.
- `configurations/`: configuraciones YAML listas para ejecutar.
- `results/`: salidas de experimentos (neuroevolución y RL tabular).
- `Gymnasium alumno.ipynb`, `Q-learning.ipynb`: notebooks de apoyo.
- `xxx.py`: script adicional con PPO (Stable-Baselines3) para LunarLander.

## Salidas generadas

### Neuroevolución (`main.py`)

Se crea un directorio dentro de:

```text
results/<environment_name>/<algorithm>_<timestamp>/
```

Archivos típicos:

- `config.json` y `config.yaml`
- `metrics_per_generation.csv`
- `hall_of_fame.json`
- `training_evolution.png`
- `best_network.png`
- `final_evaluation.json` (en evaluaciones finales por escenarios)

### RL tabular (`q_main.py`)

Se crea un directorio dentro de:

```text
results/<environment_name>/<algorithm>/exp_<timestamp>/
```

Archivos típicos:

- `training_results.png`
- `q_table.pkl`

## Configuración

Las configuraciones de neuroevolución se definen en `configurations/*.yaml` con:

- Entorno (`environment_name`, `env_kwargs`)
- Arquitectura MLP (`architecture`, `labels`)
- Algoritmo evolutivo (`simple`, `mu_plus_lambda`, `mu_comma_lambda`, etc.)
- Operadores de `crossover`, `mutation` y `selection`
- Umbral de éxito (`success_threshold`)

## Dependencias destacadas

- `gymnasium[box2d]`
- `deap`
- `flappy-bird-gymnasium`
- `numpy`, `pandas`, `matplotlib`
- `torch`, `torchvision`
- `stable-baselines3[extra]`

## Notas

- Los resultados actuales del repositorio ya están versionados dentro de `results/`.
- Para reproducibilidad, fija `random_seed` en tus YAML.
