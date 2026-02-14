import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import base, creator, tools
from loky import get_reusable_executor

from neuroevolution.config_exp import (
    ALGORITHM_FUNCTIONS,
    CROSSOVER_FUNCTIONS,
    MUTATION_FUNCTIONS,
    SELECTION_FUNCTIONS,
    ExperimentConfig,
)
from neuroevolution.MLP import MLP

EXTERNAL_ENV_REGISTRY = {
    "FlappyBird": "flappy_bird_gymnasium",
}


class MLPExperiment:
    """Manages a complete evolutionary training experiment."""

    def __init__(self, config: ExperimentConfig):
        """
        :param config: Experiment configuration object containing all settings for the experiment
        """
        self.config = config
        self.exp_dir = None
        self.toolbox = None
        self.num_genes = None
        self.n_workers = config.algorithm.n_workers or (
            os.cpu_count() - 1 if os.cpu_count() > 1 else 1
        )

        if config.algorithm.random_seed is not None:
            random.seed(config.algorithm.random_seed)
            np.random.seed(config.algorithm.random_seed)

        self._setup_deap()

    def _setup_deap(self) -> None:
        """Configures DEAP toolbox and types based on the experiment configuration."""
        dummy_agent = MLP(
            self.config.architecture,
            labels=(
                ["x", "y", "vx", "vy", "θ", "ω", "leg_L", "leg_R"],
                ["Nothing", "Fire Left", "Fire Main", "Fire Right"],
            ),
        ).to_chromosome()
        self.num_genes = len(dummy_agent)

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create(
                "Individual", list, fitness=creator.FitnessMax, strategy=None
            )

        self.toolbox = base.Toolbox()
        self.toolbox.register(alias="gene", function=random.uniform, a=-1.0, b=1.0)
        self.toolbox.register(
            alias="strategy_gene", function=random.uniform, a=0.1, b=0.5
        )
        self.toolbox.register(
            alias="individual",
            function=self._create_individual,
            icls=creator.Individual,
            gene_func=self.toolbox.gene,
            strat_func=self.toolbox.strategy_gene,
            n=self.num_genes,
        )
        self.toolbox.register(
            alias="population",
            function=tools.initRepeat,
            container=list,
            func=self.toolbox.individual,
        )

        self.toolbox.register(
            alias="evaluate", function=_fitness_function, config=self.config
        )

        selection_func = SELECTION_FUNCTIONS[self.config.algorithm.selection.method]
        selection_params = self.config.algorithm.selection.get_params()
        self.toolbox.register(
            alias="select", function=selection_func, **selection_params
        )

        crossover_func = CROSSOVER_FUNCTIONS[self.config.algorithm.crossover.method]
        crossover_params = self.config.algorithm.crossover.get_params()
        self.toolbox.register(alias="mate", function=crossover_func, **crossover_params)

        mutation_func = MUTATION_FUNCTIONS[self.config.algorithm.mutation.method]
        mutation_params = self.config.algorithm.mutation.get_params()
        self.toolbox.register(alias="mutate", function=mutation_func, **mutation_params)

    def _create_individual(
        self, icls, gene_func, strat_func, n
    ) -> "creator.Individual":
        """
        Creates an individual with a specified strategy.

        :param icls: DEAP individual class
        :param gene_func: Function to generate gene values
        :param strat_func: Function to generate strategy values
        :param n: Number of genes/strategy elements
        :return: An individual with initialized genes and strategy
        """
        ind = icls(gene_func() for _ in range(n))
        ind.strategy = [strat_func() for _ in range(n)]
        return ind

    def _setup_experiment_dir(self) -> Path:
        """Creates and returns the experiment directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = (
            self.config.experiments_dir
            / f"{self.config.environment_name}"
            / f"{self.config.algorithm.algorithm.value}_{timestamp}"
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    def _save_experiment_config(self) -> None:
        """Saves the experiment configuration."""
        config_dict = self.config.to_dict()
        config_dict["timestamp"] = datetime.now().isoformat()
        config_dict["num_genes"] = self.num_genes
        config_dict["n_workers"] = self.n_workers

        self.config.save_json(self.exp_dir / "config.json")
        self.config.save_yaml(self.exp_dir / "config.yaml")

    def _save_hall_of_fame(self, hall_of_fame: tools.HallOfFame) -> None:
        """Saves the Hall of Fame."""

        hof_data = {
            "best_individuals": [
                {
                    "rank": i + 1,
                    "fitness": ind.fitness.values[0],
                    "genome": ind,
                }
                for i, ind in enumerate(hall_of_fame)
            ]
        }

        with open(self.exp_dir / "hall_of_fame.json", "w") as f:
            json.dump(hof_data, f, indent=2)

    def _plot_evolution(self, logs: tools.Logbook) -> None:
        """Generates and saves the evolution plot."""
        gen = logs.select("gen")
        fit_avg = logs.select("avg")
        fit_std = logs.select("std")
        fit_max = logs.select("max")
        fit_min = logs.select("min")
        fit_median = logs.select("median")

        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

        avg = np.array(fit_avg)
        std = np.array(fit_std)

        ax.fill_between(
            gen, avg - std, avg + std, alpha=0.2, color="#00e676", label="Std Dev"
        )
        ax.plot(
            gen,
            fit_max,
            color="#ffeb3b",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Max Reward",
        )
        ax.plot(
            gen,
            fit_min,
            color="#ff5252",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Min Reward",
        )
        ax.plot(gen, avg, color="#00e676", linewidth=2, label="Average Reward")
        ax.plot(
            gen,
            fit_median,
            color="#2196f3",
            linestyle="-.",
            linewidth=2,
            label="Median Reward",
        )

        ax.grid(
            True, which="major", linestyle=":", linewidth=0.5, color="gray", alpha=0.5
        )
        ax.set_ylim(min(fit_median) - 50, max(fit_max) + 50)
        ax.set_xlabel("Generations", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_ylabel("Reward", fontsize=12, fontweight="bold", labelpad=10)
        ax.set_title(
            f"Evolutionary Training {self.config.environment_name}", fontsize=14, pad=20
        )

        if self.config.success_threshold is not None:
            ax.axhline(
                y=self.config.success_threshold,
                color="white",
                linestyle="-",
                linewidth=0.8,
                alpha=0.3,
            )
            ax.text(
                0,
                self.config.success_threshold + 10,
                f"Solved Threshold ({self.config.success_threshold})",
                color="white",
                alpha=0.6,
                fontsize=9,
            )

        ax.legend(
            loc="lower right", frameon=True, facecolor="#222222", edgecolor="gray"
        )

        plt.tight_layout()
        fig.savefig(
            self.exp_dir / "training_evolution.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    def run(self) -> dict:
        """Runs the full experiment: initializes population, runs the evolutionary algorithm, evaluates the best agent, and saves all results and metrics."""
        self.exp_dir = self._setup_experiment_dir()
        print("=" * 80)
        print(f"EXPERIMENT: {self.exp_dir.name}")
        print("=" * 80)

        if self.config.algorithm.random_seed is not None:
            print(f"Seed: {self.config.algorithm.random_seed}")

        self._save_experiment_config()

        start = time.time()
        print(f"STARTING EVOLUTION - {self.config.environment_name}")
        print("=" * 80)
        print(f"Architecture: {self.config.architecture}")
        print(f"Population: {self.config.algorithm.population_size}")
        print(f"Generations: {self.config.algorithm.num_generations}")
        print(f"Algorithm: {self.config.algorithm.algorithm.value}")
        print(f"Genes per individual: {self.num_genes}")
        print(f"Workers: {self.n_workers}")
        print(f"\nOperators:")
        print(
            f"  Crossover: {self.config.algorithm.crossover.method.value} "
            f"(p={self.config.algorithm.crossover.probability})"
        )
        print(
            f"  Mutation: {self.config.algorithm.mutation.method.value} "
            f"(p={self.config.algorithm.mutation.probability})"
        )
        print(f"  Selection: {self.config.algorithm.selection.method.value}")
        print("=" * 80)

        population = self.toolbox.population(n=self.config.algorithm.population_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("median", np.median)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hall_of_fame = tools.HallOfFame(self.config.algorithm.elite_size)

        executor = get_reusable_executor(
            max_workers=self.n_workers,
            initializer=_worker_initializer,
            initargs=(self.config.environment_name,),
        )
        self.toolbox.register(alias="map", function=executor.map)

        try:
            algorithm_func = ALGORITHM_FUNCTIONS[self.config.algorithm.algorithm]
            algorithm_params = self.config.algorithm.get_algorithm_params()
            algorithm_params["stats"] = stats
            algorithm_params["halloffame"] = hall_of_fame

            population, logs = algorithm_func(
                population, self.toolbox, **algorithm_params
            )

            end = time.time()

            print("=" * 80)
            print(f"Evolution completed in {end - start:.2f} seconds")
            print(f"Best fitness: {hall_of_fame[0].fitness.values[0]:.2f}")
            print("=" * 80)

            metrics_data = []
            for record in logs:
                gen = record["gen"]
                if gen % 10 == 0 or gen == self.config.algorithm.num_generations - 1:
                    metrics_data.append(
                        {
                            "generation": gen,
                            "max_fitness": record["max"],
                            "avg_fitness": record["avg"],
                            "median_fitness": record["median"],
                            "min_fitness": record["min"],
                            "std_fitness": record["std"],
                        }
                    )

            df_metrics = pd.DataFrame(metrics_data)
            df_metrics.to_csv(self.exp_dir / "metrics_per_generation.csv", index=False)

            self._save_hall_of_fame(hall_of_fame)
            self._plot_evolution(logs)

            best_agent = _create_agent(
                hall_of_fame[0],
                config=self.config,
            )
            fig_nn, _ = best_agent.plot_network(figsize=(14, 10))
            fig_nn.savefig(
                self.exp_dir / "best_network.png", dpi=300, bbox_inches="tight"
            )
            plt.close(fig_nn)

            print("\n" + "=" * 80)
            print("EXPERIMENT SUMMARY")
            print("=" * 80)
            print(f"Directory: {self.exp_dir}")
            print(f"Best fitness (training): {hall_of_fame[0].fitness.values[0]:.2f}")

            return {
                "exp_dir": self.exp_dir,
                "best_fitness": hall_of_fame[0].fitness.values[0],
                "best_genome": hall_of_fame[0],
                "training_time": end - start,
            }

        finally:
            executor.shutdown(wait=True)


def _worker_initializer(env_name: str):
    """
    Worker initializer function to import necessary environment modules based on the environment name.
    This ensures that each worker process has the required environment registered and available for use.

    :param env_name: Name of the environment being used in the experiment
    """
    for prefix, module_name in EXTERNAL_ENV_REGISTRY.items():
        if prefix in env_name:
            try:
                __import__(module_name)
            except ImportError:
                print(
                    f"Error: The module {module_name} is required for the environment {env_name}"
                )


def _fitness_function(individual: list, config: ExperimentConfig) -> Tuple[float]:
    """
    Fitness function that evaluates an individual's performance in the environment.
    It is created "out of the class" to be compatible with DEAP+Loky's requirements.

    :param individual: Genome of the agent to evaluate
    :param config: Experiment configuration object
    :return: Tuple with the fitness (average reward over evaluation episodes)
    """
    agent = _create_agent(individual, config)
    fitness, _, _, _ = _evaluate_agent(agent, config)
    return (fitness,)


def _create_agent(genome: list, config: ExperimentConfig) -> MLP:
    """
    Creates an MLP agent from a genome.

    :param genome: List of gene values representing the agent's parameters
    :param config: Experiment configuration object
    :return: MLP agent instance
    """
    agent = MLP(config.architecture, labels=config.labels)
    agent.from_chromosome(genome)
    return agent


def _evaluate_agent(
    agent: MLP,
    config: ExperimentConfig,
    num_episodes: int = None,
    render: str = None,
) -> Tuple[float, int, float, float]:
    """
    Evaluates an agent in the environment.

    :param agent: MLP agent to evaluate
    :param config: Experiment configuration object
    :param num_episodes: Number of episodes to evaluate
    :param render: Optional render mode
    :return: (fitness, successes, mean_reward, std_reward)
    """
    if num_episodes is None:
        num_episodes = config.algorithm.num_eval_episodes

    rewards = []
    successes = 0

    use_success_bonus = config.success_threshold is not None

    for _ in range(num_episodes):
        env = gym.make(
            config.environment_name,
            render_mode=render,
            **config.env_kwargs,
        )

        obs, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(agent.forward(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        if use_success_bonus and total_reward >= config.success_threshold:
            successes += 1

        env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))

    if not use_success_bonus:
        fitness = mean_reward - (
            0.1 * std_reward
        )  # Alternativas: lower confidence bound, percentil inferior (25%), media armónica (solo con valores positivos), media de la raíz cúbica (desigualdad de Jensen)
    else:
        success_bonus = successes * config.success_threshold / num_episodes
        fitness = mean_reward + success_bonus - (0.1 * std_reward)

    return fitness, successes, mean_reward, std_reward
