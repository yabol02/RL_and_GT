import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from loky import get_reusable_executor

from MLP import MLP

ARCHITECTURE = [8, 6, 4]
NUM_POPULATION = 100
NUM_GENERATIONS = 300
N_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

dummy_agent = MLP(
    ARCHITECTURE,
    labels=(
        ["x", "y", "vx", "vy", "θ", "ω", "leg_L", "leg_R"],
        ["Nothing", "Fire Left", "Fire Main", "Fire Right"],
    ),
).to_chromosome()
NUM_GENES = len(dummy_agent)

MUTATION_MU = 0.0
MUTATION_SIGMA = 1.0
MUTATION_INDPB = 2 / NUM_GENES
CROSSOVER_PROB = 0.75
MUTATION_PROB = 0.25
TOURNAMENT_SIZE = 4
ELITE_SIZE = 10

NUM_EVAL_EPISODES = 5
SUCCESS_THRESHOLD = 200.0

EXPERIMENTS_DIR = Path("experiments")
RANDOM_SEED = None  # None for random, or a number for reproducibility


def setup_experiment() -> Path:
    """
    Creates a directory for the current experiment with a timestamp.

    :return: Path to the experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = EXPERIMENTS_DIR / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_experiment_config(exp_dir: Path) -> None:
    """
    Saves the experiment configuration to a JSON file.

    :param exp_dir: Experiment directory
    """
    config = {
        "timestamp": datetime.now().isoformat(),
        "architecture": ARCHITECTURE,
        "population_size": NUM_POPULATION,
        "num_generations": NUM_GENERATIONS,
        "num_genes": NUM_GENES,
        "mutation": {
            "mu": MUTATION_MU,
            "sigma": MUTATION_SIGMA,
            "indpb": MUTATION_INDPB,
            "prob": MUTATION_PROB,
        },
        "crossover": {
            "prob": CROSSOVER_PROB,
            "alpha": 0.5,
        },
        "selection": {
            "type": "tournament",
            "tournament_size": TOURNAMENT_SIZE,
        },
        "elite_size": ELITE_SIZE,
        "num_eval_episodes": NUM_EVAL_EPISODES,
        "success_threshold": SUCCESS_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "n_workers": N_WORKERS,
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def save_hall_of_fame(exp_dir: Path, hall_of_fame: tools.HallOfFame) -> None:
    """
    Saves the Hall of Fame individuals to a JSON file.

    :param exp_dir: Experiment directory
    :param hall_of_fame: Hall of Fame with the best individuals
    """
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

    with open(exp_dir / "hall_of_fame.json", "w") as f:
        json.dump(hof_data, f, indent=2)


def evaluate_final_scenarios(
    agent: MLP, exp_dir: Path, render_mode: None | str = None
) -> dict:
    """
    Evaluates the best agent in three scenarios: easy, medium, and difficult.

    :param agent: MLP agent to evaluate
    :param exp_dir: Experiment directory to save results
    :param render_mode: Render mode for the environment (e.g., "human", "rgb_array", or None)
    :return: Dictionary with evaluation results for each scenario
    """
    results = {}
    np.random.seed(0)

    # Easy scenario (normal conditions)
    print("\nEvaluating EASY envs")
    easy_rewards = []
    easy_successes = 0
    for seed in range(100):
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        obs, _ = env.reset(seed=1000 + seed)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(agent.forward(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        easy_rewards.append(total_reward)
        if total_reward >= SUCCESS_THRESHOLD:
            easy_successes += 1
        env.close()

    results["easy"] = {
        "mean_reward": np.mean(easy_rewards),
        "std_reward": np.std(easy_rewards),
        "success_rate": easy_successes / 100,
        "min_reward": np.min(easy_rewards),
        "max_reward": np.max(easy_rewards),
    }

    # MEDIUM scenario (modified gravity between -12 and -10)
    print("Evaluating MEDIUM envs (modified gravity)...")
    medium_rewards = []
    medium_successes = 0

    for seed in range(100):
        gravity_value = np.random.uniform(-12, -10)
        env = gym.make("LunarLander-v3", gravity=gravity_value)
        obs, _ = env.reset(seed=2000 + seed)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(agent.forward(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        medium_rewards.append(total_reward)
        if total_reward >= SUCCESS_THRESHOLD:
            medium_successes += 1
        env.close()

    results["medium"] = {
        "mean_reward": np.mean(medium_rewards),
        "std_reward": np.std(medium_rewards),
        "success_rate": medium_successes / 100,
        "min_reward": np.min(medium_rewards),
        "max_reward": np.max(medium_rewards),
    }

    # Difficult scenario (gravity and random wind between -10 and -15 + turbulence power 1.5)
    print("Evaluating DIFFICULT envs (wind + turbulence)...")
    difficult_rewards = []
    difficult_successes = 0

    for seed in range(100):
        gravity_value = np.random.uniform(-15, -12)
        wind_power = np.random.uniform(10, 12)
        env = gym.make(
            "LunarLander-v3",
            wind_power=wind_power,
            turbulence_power=1.5,
        )
        obs, _ = env.reset(seed=3000 + seed)
        total_reward = 0.0
        done = False

        while not done:
            action = int(np.argmax(agent.forward(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        difficult_rewards.append(total_reward)
        if total_reward >= SUCCESS_THRESHOLD:
            difficult_successes += 1
        env.close()

    results["difficult"] = {
        "mean_reward": np.mean(difficult_rewards),
        "std_reward": np.std(difficult_rewards),
        "success_rate": difficult_successes / 100,
        "min_reward": np.min(difficult_rewards),
        "max_reward": np.max(difficult_rewards),
    }

    with open(exp_dir / "final_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def create_individual(
    genome: list[float], labels: tuple[list[str], list[str]] | None = None
) -> MLP:
    """
    Creates an MLP agent from a given genome.

    :param genome: A list of floats representing the weights and biases of the MLP.
    :param labels: Optional tuple of (input_labels, output_labels) for the MLP.
    :return: An MLP agent created from the given genome.
    """
    agent = MLP(ARCHITECTURE, labels=labels)
    agent.from_chromosome(genome)
    return agent


def evaluate_individual(
    agent: MLP, num_episodes: int = NUM_EVAL_EPISODES, render: str | None = None
) -> tuple[float]:
    """
    Evaluates the fitness of an MLP agent by running multiple episodes in the LunarLander-v3 environment.

    :param agent: The MLP agent to be evaluated
    :param num_episodes: Number of episodes to run for evaluation (default is NUM_EVAL_EPISODES)
    :param render: Render mode for the environment (default is None, set to "human" to visualize, or "rgb_array" for off-screen rendering)
    :return: A tuple containing the fitness value, number of successes, mean reward, and standard deviation of rewards for the agent
    """
    rewards = []
    successes = 0

    for _ in range(num_episodes):
        reward = run_simulation(agent, render)
        if reward >= SUCCESS_THRESHOLD:
            successes += 1

        rewards.append(reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Bonus exponencial por éxito (no funciona muy bien, el agente aprende a volar siempre y así no se estrella salvo en casos sencillos que baja tranquilamente hasta el suelo)
    # success_bonus = ((successes / num_episodes) ** 2) * SUCCESS_THRESHOLD*1.5  # Bonus hasta 300 puntos
    # consistency_penalty = std_reward * 0.2  # Penalización por inconsistencia
    # fitness = mean_reward + success_bonus - consistency_penalty  # Fitness total

    # Bonus lineal por éxitos
    success_bonus = successes * 50
    fitness = mean_reward + success_bonus - (0.1 * std_reward)

    return fitness, successes, mean_reward, std_reward


def run_simulation(agent: MLP, render: str | None) -> None:
    """
    Runs a single episode of the LunarLander-v3 environment using the provided MLP agent.

    :param agent: The MLP agent to be evaluated in the environment
    :param render: Render mode for the environment (e.g., "human" for visualization, "rgb_array" for off-screen rendering, or None for no rendering)
    """
    env = gym.make("LunarLander-v3", render_mode=render)
    obs, _ = env.reset()
    total_reward = 0.0
    done = False

    while not done:
        action = int(np.argmax(agent.forward(obs)))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    return total_reward


def fitness_function(genome: list[float]) -> tuple[float]:
    """
    Evaluates the fitness of an individual based on its genome.

    :param genome: A list of floats representing the weights and biases of the MLP agent.
    :return: A tuple containing the fitness value of the individual.
    """
    agent = create_individual(genome)
    reward, _, _, _ = evaluate_individual(agent)
    return (reward,)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register(alias="gene", function=random.uniform, a=-1.0, b=1.0)
toolbox.register(
    alias="individual",
    function=tools.initRepeat,
    container=creator.Individual,
    func=toolbox.gene,
    n=NUM_GENES,
)
toolbox.register(
    alias="population",
    function=tools.initRepeat,
    container=list,
    func=toolbox.individual,
)
toolbox.register(alias="evaluate", function=fitness_function)
toolbox.register(
    alias="select", function=tools.selTournament, tournsize=TOURNAMENT_SIZE
)
toolbox.register(alias="mate", function=tools.cxBlend, alpha=0.5)
toolbox.register(
    alias="mutate",
    function=tools.mutGaussian,
    mu=MUTATION_MU,
    sigma=MUTATION_SIGMA,
    indpb=MUTATION_INDPB,
)


def evaluate_population_metrics(population) -> dict:
    """
    Evaluates and aggregates metrics for a given population of individuals (only top 20 for speed).

    :param population: List of individuals in the population
    :return: Dictionary containing aggregated metrics of the population
    """
    all_success_rates = []
    all_mean_rewards = []
    all_std_rewards = []

    for ind in population[: min(20, len(population))]:
        agent = create_individual(ind)
        _, successes, mean_rew, std_rew = evaluate_individual(agent)
        success_rate = successes / NUM_EVAL_EPISODES
        all_success_rates.append(success_rate)
        all_mean_rewards.append(mean_rew)
        all_std_rewards.append(std_rew)

    return {
        "mean_success_rate": np.mean(all_success_rates),
        "max_success_rate": np.max(all_success_rates),
        "mean_reward": np.mean(all_mean_rewards),
        "mean_std": np.mean(all_std_rewards),
    }


def plot_logs(logs: tools.logs, exp_dir: Path = None) -> tuple:
    """
    Plots the evolution of fitness metrics over generations using the logs data.

    :param logs: The logs object containing the recorded metrics for each generation during the evolutionary process.
    :param exp_dir: Optional directory to save the plot
    :return: Tuple with (fig, ax)
    """
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
    ax.grid(True, which="major", linestyle=":", linewidth=0.5, color="gray", alpha=0.5)

    ax.set_ylim(min(fit_median) - 50, max(fit_max) + 50)

    ax.set_xlabel("Generations", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Reward", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("Evolutionary Training: MLP on LunarLander-v3", fontsize=14, pad=20)

    ax.axhline(y=200, color="white", linestyle="-", linewidth=0.8, alpha=0.3)
    ax.text(0, 210, "Solved Threshold (200)", color="white", alpha=0.6, fontsize=9)
    ax.legend(loc="lower right", frameon=True, facecolor="#222222", edgecolor="gray")
    plt.tight_layout()

    if exp_dir:
        fig.savefig(exp_dir / "training_evolution.png", dpi=300, bbox_inches="tight")

    return fig, ax


def run_evolution() -> None:
    """
    Executes the complete evolutionary process with an experiment management system:
    - Saves configuration and results
    - Records metrics every 10 generations
    - Saves Hall of Fame
    - Evaluates in multiple scenarios
    - Generates visualizations
    """
    exp_dir = setup_experiment()
    print("=" * 80)
    print(f"EXPERIMENT: {exp_dir.name}")
    print("=" * 80)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        print(f"Seed set: {RANDOM_SEED}")

    save_experiment_config(exp_dir)

    start = time.time()
    print("STARTING EVOLUTION - LUNAR LANDER")
    print("=" * 80)
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Population: {NUM_POPULATION}")
    print(f"Generations: {NUM_GENERATIONS}")
    print(f"Genes per individual: {NUM_GENES}")
    print(f"Workers: {N_WORKERS}")
    print(
        f"Mutation mu: {MUTATION_MU}, sigma: {MUTATION_SIGMA}, indpb: {MUTATION_INDPB}"
    )
    print(f"Crossover prob: {CROSSOVER_PROB}, Mutation prob: {MUTATION_PROB}")
    print("=" * 80)

    population = toolbox.population(n=NUM_POPULATION)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register(name="avg", function=np.mean)
    stats.register(name="std", function=np.std)
    stats.register(name="median", function=np.median)
    stats.register(name="min", function=np.min)
    stats.register(name="max", function=np.max)

    hall_of_fame = tools.HallOfFame(ELITE_SIZE)

    executor = get_reusable_executor(max_workers=N_WORKERS)
    toolbox.register(alias="map", function=executor.map)

    try:
        population, logs = algorithms.eaMuCommaLambda(
            population,
            toolbox,
            mu=NUM_POPULATION,
            lambda_=NUM_POPULATION * 2,
            cxpb=CROSSOVER_PROB,
            mutpb=MUTATION_PROB,
            ngen=NUM_GENERATIONS,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=True,
        )

        end = time.time()
        print("=" * 80)
        print(f"Evolución completada en {end - start:.2f} segundos")
        print(f"Mejor fitness: {hall_of_fame[0].fitness.values[0]:.2f}")
        print("=" * 80)

        # Logs for each 10 generations
        metrics_data = []
        for record in logs:
            gen = record["gen"]
            if gen % 10 == 0 or gen == NUM_GENERATIONS - 1:
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
        df_metrics.to_csv(exp_dir / "metrics_per_generation.csv", index=False)

        save_hall_of_fame(exp_dir, hall_of_fame)

        fig, ax = plot_logs(logs, exp_dir)
        plt.close(fig)

        # Testing the best agent in multiple scenarios
        print("\nEvaluating the best agent in final scenarios...")
        best_agent = create_individual(
            hall_of_fame[0],
            labels=(
                ["x", "y", "vx", "vy", "θ", "ω", "leg_L", "leg_R"],
                ["Nothing", "Fire Left", "Fire Main", "Fire Right"],
            ),
        )
        eval_results = evaluate_final_scenarios(best_agent, exp_dir)

        fig_nn, ax_nn = best_agent.plot_network(figsize=(14, 10))
        fig_nn.savefig(exp_dir / "best_network.png", dpi=300, bbox_inches="tight")
        plt.close(fig_nn)

        # Final summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Directory: {exp_dir}")
        print(f"Best fitness (training): {hall_of_fame[0].fitness.values[0]:.2f}")
        print("\nEvaluation in scenarios:")
        print(
            f"  Easy:    {eval_results['easy']['mean_reward']:.2f} ± {eval_results['easy']['std_reward']:.2f} "
            f"(success: {eval_results['easy']['success_rate']*100:.0f}%)"
        )
        print(
            f"  Medium:    {eval_results['medium']['mean_reward']:.2f} ± {eval_results['medium']['std_reward']:.2f} "
            f"(success: {eval_results['medium']['success_rate']*100:.0f}%) "
        )
        print(
            f"  Difficult:  {eval_results['difficult']['mean_reward']:.2f} ± {eval_results['difficult']['std_reward']:.2f} "
            f"(success: {eval_results['difficult']['success_rate']*100:.0f}%) "
        )
        print(f"\nFiles saved in: {exp_dir}")

    finally:
        executor.shutdown(wait=True)


if __name__ == "__main__":
    run_evolution()
