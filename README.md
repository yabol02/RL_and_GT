## Project

Evolutionary training of an MLP agent for Gymnasium environments (including Flappy Bird) using DEAP. The pipeline creates an experiment, stores metrics, and writes artifacts under `experiments/`.

## How to run (uv)

1) Create a uv environment with Python >= 3.13:

```bash
uv init
```

2) Install dependencies from the project:

```bash
uv sync
```

3) Run the default experiment:

```bash
uv run ./main.py
```

You can change the file to choose the configuration of the experiment in `main.py` or add a new one under `configurations/`.

## What is in the repo

- `main.py`: entry point. Defines `MLPExperiment`, configures DEAP, runs the evolutionary loop, and writes results to `experiments/` (config, metrics, hall of fame, plots, final evaluation).
- `config_exp.py`: configuration models (dataclasses) and enums for algorithm, crossover, mutation, and selection. Includes JSON/YAML load/save.
- `config_manager.py`: utilities to register, save, load, and compare configurations.
- `MLP.py`: multilayer perceptron implementation, chromosome conversion, and network visualization.
- `configurations/`: ready-to-run YAML configurations.
- `experiments/`: run outputs (config, metrics per generation, hall of fame, final evaluation, plots).
- `Gymnasium alumno.ipynb`: tutorial notebook for exploring Gymnasium.
- `pyproject.toml`: project metadata and dependencies.
- `LICENSE`: project license.
