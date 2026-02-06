import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from deap import algorithms, base, creator, tools
from loky import get_reusable_executor

from MLP import MLP

ARCHITECTURE = [8, 6, 4]
NUM_POPULATION = 100
NUM_GENERATIONS = 300
N_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1

dummy_agent = MLP(ARCHITECTURE).to_chromosome()
NUM_GENES = len(dummy_agent)

MUTATION_MU = 0.0
MUTATION_SIGMA = 1.0
MUTATION_INDPB = 2 / NUM_GENES
CROSSOVER_PROB = 0.75
MUTATION_PROB = 0.25
TOURNAMENT_SIZE = 4
ELITE_SIZE = 1

NUM_EVAL_EPISODES = 5
SUCCESS_THRESHOLD = 200.0



def create_individual(genome: list[float]) -> MLP:
    """
    Creates an MLP agent from a given genome.
    
    :param genome: A list of floats representing the weights and biases of the MLP.
    :return: An MLP agent created from the given genome.
    """
    agent = MLP(ARCHITECTURE)
    agent.from_chromosome(genome)
    return agent


def evaluate_individual(agent: MLP, num_episodes: int = NUM_EVAL_EPISODES, render: str | None = None) -> tuple[float]:
    rewards = []
    successes = 0

    for _ in range(num_episodes):
        reward = run_simulation(agent, render)
        if reward >= SUCCESS_THRESHOLD:
            successes += 1

        rewards.append(reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Bonus exponencial por éxito (no funciona muy bien, el agente aprende a volar siempre y no así no se estrella salvo en casos sencillos que baja tranquilamente hasta el suelo)
    # success_bonus = ((successes / num_episodes) ** 2) * SUCCESS_THRESHOLD*1.5  # Bonus hasta 300 puntos
    # consistency_penalty = std_reward * 0.2  # Penalización por inconsistencia
    # fitness = mean_reward + success_bonus - consistency_penalty  # Fitness total
    
    # Bonus lineal por éxitos
    success_bonus = successes * 50
    fitness = mean_reward + success_bonus - (0.1 * std_reward)

    return fitness, successes, mean_reward, std_reward

def run_simulation(agent: MLP, render: str | None) -> None:
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
toolbox.register(alias="select", function=tools.selTournament, tournsize=TOURNAMENT_SIZE)
toolbox.register(alias="mate", function=tools.cxBlend, alpha=0.5)
toolbox.register(
    alias="mutate", function=tools.mutGaussian, mu=MUTATION_MU, sigma=MUTATION_SIGMA, indpb=MUTATION_INDPB
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

    for ind in population[:min(20, len(population))]:
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

def plot_logs(logbook: tools.Logbook) -> None:
    gen = logbook.select("gen")
    fit_avg = logbook.select("avg")
    fit_std = logbook.select("std")
    fit_max = logbook.select("max")
    fit_min = logbook.select("min")
    fit_median = logbook.select("median")

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    avg = np.array(fit_avg)
    std = np.array(fit_std)

    ax.fill_between(gen, avg - std, avg + std, alpha=0.2, color='#00e676', label='Std Dev')
    ax.plot(gen, fit_max, color='#ffeb3b', linestyle='--', linewidth=1, alpha=0.7, label='Max Reward')
    ax.plot(gen, fit_min, color='#ff5252', linestyle='--', linewidth=1, alpha=0.7, label='Min Reward')
    ax.plot(gen, avg, color='#00e676', linewidth=2, label='Average Reward')
    ax.plot(gen, fit_median, color='#2196f3', linestyle='-.', linewidth=2, label='Median Reward')
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.5)

    ax.set_ylim(min(fit_median)-50, max(fit_max)+50)
    
    ax.set_xlabel('Generations', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Evolutionary Training: MLP on LunarLander-v3', fontsize=14, pad=20)

    ax.axhline(y=200, color='white', linestyle='-', linewidth=0.8, alpha=0.3)
    ax.text(0, 210, 'Solved Threshold (200)', color='white', alpha=0.6, fontsize=9)
    ax.legend(loc='upper left', frameon=True, facecolor='#222222', edgecolor='gray')
    plt.tight_layout()
    return fig, ax

def run_evolution() -> None:
    start = time.time()
    print("="*80)
    print("INICIANDO EVOLUCIÓN - LUNAR LANDER")
    print("="*80)
    print(f"Arquitectura: {ARCHITECTURE}")
    print(f"Población: {NUM_POPULATION}")
    print(f"Generaciones: {NUM_GENERATIONS}")
    print(f"Genes por individuo: {NUM_GENES}")
    print(f"Workers: {N_WORKERS}")
    print(f"Mutation mu: {MUTATION_MU}, sigma: {MUTATION_SIGMA}, indpb: {MUTATION_INDPB}")
    print(f"Crossover prob: {CROSSOVER_PROB}, Mutation prob: {MUTATION_PROB}")
    print("="*80)

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
    finally:
        end = time.time()
        print(f"Evolution took {end - start:.2f} seconds")
        executor.shutdown(wait=True)
        print(
            f"Best individual: {hall_of_fame[0]}, Fitness: {hall_of_fame[0].fitness.values[0]}"
        )

        agent = create_individual(hall_of_fame[0])
        fitness, successes, mean_reward, std_reward = evaluate_individual(agent, 10, render="human")
        print(f"Test run: Reward = {fitness:.2f}, Successes = {successes}/10, Mean Reward = {mean_reward:.2f}, Std Reward = {std_reward:.2f}")

        fig, ax = plot_logs(logs)
        fig.savefig("evolution_logs.png", dpi=300)
        plt.show()

        # print("Evolution logs:")
        # for gen, log in enumerate(logs):
        #     print(f"Generation {gen}: {log}")


if __name__ == "__main__":
    run_evolution()
