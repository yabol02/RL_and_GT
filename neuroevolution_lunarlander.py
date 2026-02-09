import os
import pickle
import random
import time

import gymnasium as gym
import numpy as np
from deap import algorithms, base, creator, tools
from loky import get_reusable_executor

from MLP import MLP

# 1. CONFIGURACIÓN GENERAL
# 1.1 Arquitectura de la red neuronal
ARCHITECTURE = [8, 16, 16, 4]

# 1.2 Parámetros del algoritmo evolutivo
POPULATION_SIZE = 100
GENERATIONS = 50
CXPB = 0.7
MUTPB = 0.3
INDPB = 0.1
MU = 20
LAMBDA = 80

# 1.3 Parámetros de evaluación
N_EPISODES = 3
MAX_STEPS = 1000

# 1.4 Paralelización
N_WORKERS = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
print(f"🚀 Usando {N_WORKERS} workers con LOKY para paralelización")


# 2. FUNCIONES DE ENTORNO
def create_model():
    """Crea un modelo MLP con la arquitectura definida"""
    return MLP(ARCHITECTURE)


def policy(model, observation):
    """Define la política: forward pass + argmax"""
    output = model.forward(observation)
    action = np.argmax(output)
    return action


def evaluate_individual(individual):
    """
    Evalúa un individuo (cromosoma) ejecutándolo en el entorno.
    """
    model = create_model()
    model.from_chromosome(individual)

    env = gym.make("LunarLander-v3")

    total_rewards = []

    for episode in range(N_EPISODES):
        observation, info = env.reset()
        episode_reward = 0

        for step in range(MAX_STEPS):
            action = policy(model, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    env.close()

    fitness = np.mean(total_rewards)
    return (fitness,)


# 3. CONFIGURACIÓN DE DEAP

# 3.1 Crear tipos de fitness e individuo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 3.2 Crear toolbox
toolbox = base.Toolbox()

# 3.3 Obtener el tamaño del cromosoma
dummy_model = create_model()
CHROMOSOME_SIZE = dummy_model.size

# 3.4 Registrar funciones
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register(
    "individual",
    tools.initRepeat,
    creator.Individual,
    toolbox.attr_float,
    n=CHROMOSOME_SIZE,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=3)


# 4. ALGORITMO EVOLUTIVO CON PARALELIZACIÓN
def run_evolution_loky():
    """Ejecuta el algoritmo evolutivo con paralelización usando LOKY"""

    print("=" * 70)
    print("INICIO DE LA NEUROEVOLUCIÓN (LOKY PARALELIZADO)")
    print("=" * 70)
    print(f"Arquitectura: {ARCHITECTURE}")
    print(f"Tamaño del cromosoma: {CHROMOSOME_SIZE}")
    print(f"Población: {POPULATION_SIZE}")
    print(f"Generaciones: {GENERATIONS}")
    print(f"Workers (LOKY): {N_WORKERS}")
    print("=" * 70)

    # Crear executor reutilizable de LOKY
    executor = get_reusable_executor(max_workers=N_WORKERS)

    # Registrar el map paralelo en DEAP
    toolbox.register("map", executor.map)

    # Crear población inicial
    population = toolbox.population(n=POPULATION_SIZE)

    # Estadísticas
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of fame (mejores individuos)
    hof = tools.HallOfFame(1)

    start_time = time.time()

    # Ejecutar algoritmo evolutivo (mu + lambda)
    population, logbook = algorithms.eaMuPlusLambda(
        population,
        toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=CXPB,
        mutpb=MUTPB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=True,
    )

    elapsed_time = time.time() - start_time

    # Limpiar el executor
    executor.shutdown(wait=True)

    print("\n" + "=" * 70)
    print("EVOLUCIÓN COMPLETADA")
    print("=" * 70)

    # Mejor individuo
    best_individual = hof[0]
    best_fitness = best_individual.fitness.values[0]

    print(f"Mejor fitness: {best_fitness:.2f}")
    print(f"Tiempo total: {elapsed_time/60:.2f} minutos")
    print(f"Tiempo por generación: {elapsed_time/GENERATIONS:.2f} segundos")
    print("=" * 70)

    return best_individual, logbook, hof


# 5. GUARDADO Y CARGA MODELO
def save_best_model(chromosome, filename="best_model_loky.pkl"):
    """Guarda el mejor cromosoma"""
    with open(filename, "wb") as f:
        pickle.dump({"chromosome": chromosome, "architecture": ARCHITECTURE}, f)
    print(f"\n✓ Modelo guardado en: {filename}")


def load_best_model(filename="best_model_loky.pkl"):
    """Carga el mejor cromosoma"""
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data["chromosome"], data["architecture"]


# 6. TESTING Y VISUALIZACIÓN
def test_model(chromosome, n_episodes=5):
    """Prueba el modelo entrenado con visualización"""

    print("\n" + "=" * 70)
    print("PROBANDO MODELO ENTRENADO")
    print("=" * 70)

    # Crear modelo y cargar pesos
    model = create_model()
    model.from_chromosome(chromosome)

    # Crear entorno con visualización
    env = gym.make("LunarLander-v3", render_mode="human")

    rewards = []

    for episode in range(n_episodes):
        observation, info = env.reset()
        episode_reward = 0
        step = 0

        print(f"\nEpisodio {episode + 1}/{n_episodes}")

        while True:
            action = policy(model, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if terminated or truncated:
                normalized_reward = (episode_reward + 200) / 500
                print(f"  Recompensa: {episode_reward:.2f}")
                print(f"  Normalizada: {normalized_reward:.3f}")
                print(f"  Pasos: {step}")
                rewards.append(episode_reward)
                break

    env.close()

    print("\n" + "=" * 70)
    print(f"Recompensa promedio: {np.mean(rewards):.2f}")
    print(f"Desviación estándar: {np.std(rewards):.2f}")
    print(f"Mejor recompensa: {np.max(rewards):.2f}")
    print(f"Peor recompensa: {np.min(rewards):.2f}")
    print("=" * 70)


if __name__ == "__main__":

    print("\n¿Qué deseas hacer?")
    print("1. Entrenar nuevo modelo (LOKY PARALELIZADO)")
    print("2. Cargar modelo existente y probar")

    choice = input("\nElige una opción (1/2): ").strip()

    if choice == "1":
        # Entrenar
        best_chromosome, logbook, hof = run_evolution_loky()

        # Guardar
        save_best_model(best_chromosome)

        # Preguntar si quiere probar
        test_choice = (
            input("\n¿Deseas probar el modelo entrenado? (s/n): ").strip().lower()
        )
        if test_choice == "s":
            test_model(best_chromosome, n_episodes=5)

    elif choice == "2":
        # Cargar y probar
        try:
            chromosome, architecture = load_best_model()
            print(f"✓ Modelo cargado (arquitectura: {architecture})")
            test_model(chromosome, n_episodes=5)
        except FileNotFoundError:
            print("❌ No se encontró el archivo. Primero debes entrenar un modelo.")

    else:
        print("Opción no válida")
