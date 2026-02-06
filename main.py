import os
import random
import time

import gymnasium as gym
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


def create_individual(genome: list[float]) -> MLP:
    agent = MLP(ARCHITECTURE)
    agent.from_chromosome(genome)
    return agent


def evaluate_individual(agent: MLP, render: str | None = None) -> tuple[float]:
    rewards = []
    successes = 0

    for _ in range(5):
        env = gym.make("LunarLander-v3", render_mode=render)
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = int(np.argmax(agent.forward(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        if total_reward >= 200:
            successes += 1

        rewards.append(total_reward)
        env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Exponencial bonus for each success ( > 200). This ensures that 5 episodes of 190 are worse than 3 episodes of 210
    success_bonus = successes * 50
    fitness = mean_reward + success_bonus - (0.1 * std_reward)

    return fitness


def fitness_function(genome: list[float]) -> tuple[float]:
    agent = create_individual(genome)
    reward = evaluate_individual(agent)
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
toolbox.register(alias="select", function=tools.selTournament, tournsize=4)
toolbox.register(alias="mate", function=tools.cxBlend, alpha=0.5)
toolbox.register(
    alias="mutate", function=tools.mutGaussian, mu=0, sigma=1, indpb=2 / NUM_GENES
)


def run_evolution() -> None:
    population = toolbox.population(n=NUM_POPULATION)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register(name="avg", function=np.mean)
    stats.register(name="std", function=np.std)
    stats.register(name="median", function=np.median)
    stats.register(name="min", function=np.min)
    stats.register(name="max", function=np.max)

    hall_of_fame = tools.HallOfFame(1)

    executor = get_reusable_executor(max_workers=N_WORKERS)
    toolbox.register(alias="map", function=executor.map)
    try:
        population, logs = algorithms.eaMuCommaLambda(
            population,
            toolbox,
            mu=NUM_POPULATION,
            lambda_=NUM_POPULATION * 2,
            cxpb=0.75,
            mutpb=0.25,
            ngen=NUM_GENERATIONS,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=True,
        )
    finally:
        executor.shutdown(wait=True)
        print(
            f"Best individual: {hall_of_fame[0]}, Fitness: {hall_of_fame[0].fitness.values[0]}"
        )

        agent = create_individual(hall_of_fame[0])
        reward = evaluate_individual(agent, render="human")
        print(f"Test run: Reward = {reward}")

        # print("Evolution logs:")
        # for gen, log in enumerate(logs):
        #     print(f"Generation {gen}: {log}")


if __name__ == "__main__":
    run_evolution()
