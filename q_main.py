import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agents_rl import (MonteCarloAgent, ObservationDiscretizer, QLearningAgent,
                       SARSAAgent)


def plot_training_results(episode_rewards: np.ndarray, save_path: str, window_size: int = 100) -> None:
    """
    Plot training results with moving average.

    :param episode_rewards: Array of rewards per episode
    :param window_size: Window size for moving average
    """
    plt.figure(figsize=(12, 5))

    # Plot raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label="Episode Reward")

    # Calculate and plot moving average
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(
            episode_rewards, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(
            range(window_size - 1, len(episode_rewards)),
            moving_avg,
            label=f"{window_size}-Episode Moving Average",
            linewidth=2,
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot reward distribution
    plt.subplot(1, 2, 2)
    plt.hist(episode_rewards, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_path}/training_results.png", dpi=150, bbox_inches="tight")
    plt.close()


@dataclass
class Config:
    algorithm: str = "q_learning"  # Options: "q_learning", "sarsa", "monte_carlo"
    environment_name: str = "Taxi-v3"
    environment_kwargs: dict = field(default_factory=dict)
    observation_space: gym.Space | None = (
        None  # Only needed for continuous observation spaces
    )
    n_training_episodes: int = 10000
    max_steps: int = 99
    learning_rate: float = 0.7
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.0005
    n_eval_episodes: int = 100
    n_bins: int = 10


def main(config: Config) -> None:
    """
    Main function to configure and run Q-Learning training.

    Configure your environment and parameters here.
    """
    print("=" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"Algorithm: {config.algorithm}")
    print(f"Environment: {config.environment_name}")
    print(f"Environment kwargs: {config.environment_kwargs}")
    print(f"Training episodes: {config.n_training_episodes}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Gamma: {config.gamma}")
    print(
        f"Epsilon: {config.epsilon_start} -> {config.epsilon_end} (decay: {config.epsilon_decay})"
    )
    print("=" * 70)

    # Create environment
    env = gym.make(config.environment_name, **config.environment_kwargs)

    # Print environment information
    print("ENVIRONMENT INFORMATION")
    print("-" * 70)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}")

    # Create discretizer if needed
    discretizer = None
    if config.observation_space is not None:
        print(f"\nCreating discretizer for continuous observation space...")
        print(f"Observation space to discretize: {config.observation_space}")
        print(f"Number of bins per dimension: {config.n_bins}")
        discretizer = ObservationDiscretizer(
            config.observation_space, n_bins=config.n_bins
        )
        print(f"Total discrete states: {discretizer.n_states}")
    else:
        if isinstance(env.observation_space, gym.spaces.Discrete):
            print(f"Number of states: {env.observation_space.n}")
        else:
            print(f"Observation space shape: {env.observation_space.shape}")

    print("=" * 70 + "\n")

    # Create agent
    if config.algorithm == "q_learning":
        agent = QLearningAgent(
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            discretizer=discretizer,
        )
    elif config.algorithm == "sarsa":
        agent = SARSAAgent(
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            discretizer=discretizer,
        )
    elif config.algorithm == "monte_carlo":
        agent = MonteCarloAgent(
            env=env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
            discretizer=discretizer,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {config.algorithm}")

    # Train agent
    print("Starting training...\n")
    start = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"results/{config.environment_name}/{config.algorithm}/exp_{start}"
    os.makedirs(save_path, exist_ok=True)   
    episode_rewards = agent.train(
        n_episodes=config.n_training_episodes, max_steps=config.max_steps, verbose=True
    )
    env.close()

    print("\nTraining completed!")

    # Print Q-table statistics
    agent.print_statistics()

    # Evaluation
    print("Starting evaluation...\n")
    mean_reward, std_reward, eval_rewards = agent.evaluate(
        n_episodes=config.n_eval_episodes,
        max_steps=config.max_steps,
        render=False,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Min reward: {np.min(eval_rewards):.2f}")
    print(f"Max reward: {np.max(eval_rewards):.2f}")
    print(f"Success rate: {np.sum(eval_rewards > 0) / len(eval_rewards) * 100:.1f}%")
    print("=" * 70 + "\n")

    # Visualization
    print("Generating training visualization...")
    plot_training_results(episode_rewards, save_path, window_size=100)

    # Save Q-table
    agent.save(save_path+"/q_table.pkl")

    # Demonstrate learned policy
    print("\nDemonstrating learned policy (first 5 episodes):")
    print("-" * 70)

    env = gym.make(
        config.environment_name, **config.environment_kwargs, render_mode="human"
    )
    for episode in range(5):
        observation, info = env.reset()
        state = agent._get_state(observation)

        episode_reward = 0
        trajectory = [state]
        actions = []

        for step in range(config.max_steps):
            action = agent.select_action_greedy(state)
            actions.append(action)

            next_observation, reward, terminated, truncated, info = env.step(action)
            next_state = agent._get_state(next_observation)

            episode_reward += reward
            trajectory.append(next_state)

            if terminated or truncated:
                break

            state = next_state

        print(
            f"Episode {episode + 1}: Reward = {episode_reward:.1f}, Steps = {len(actions)}"
        )
        print(
            f"  Trajectory: {' -> '.join(map(str, trajectory[:10]))}{', ..., ' + str(trajectory[-min(len(trajectory)-10, 5):])[1:] if len(trajectory) > 10 else ']'}"
        )
        print(
            f"  Actions: {str(actions[:10])[:-1]}{', ..., ' + str(actions[-min(len(actions)-10, 5):])[1:] if len(actions) > 10 else ']'}"
        )

    print("-" * 70)
    env.close()


if __name__ == "__main__":
    ALGORITHM = "monte_carlo"  # Options: "q_learning", "sarsa", "monte_carlo"
    ENV_NAME = "Taxi-v3"  # Environment ID
    ENV_KWARGS = {
        "is_rainy": True,
        "fickle_passenger": True,
    }  # Environment-specific kwargs
    OBS_SPACE = None  # Observation space discretization (only needed for continuous spaces). For discrete environments like FrozenLake, set to None

    # For continuous observation spaces (e.g., LunarLander), uncomment:
    # ENV_NAME = "LunarLander-v3"
    # ENV_KWARGS = {}
    # OBS_SPACE = gym.spaces.Box(
    #     low=np.asanyarray([-2.5, -2.5, -10.0, -10.0, -6.2831855, -10.0, -0.0, -0.0]),
    #     high=np.asanyarray([2.5, 2.5, 10.0, 10.0, 6.2831855, 10.0, 1.0, 1.0]),
    #     shape=(8,),
    #     dtype=np.float32,
    # )  # As seen in the documentation: https://gymnasium.farama.org/environments/box2d/lunar_lander/

    config = Config(
        algorithm=ALGORITHM,
        environment_name=ENV_NAME,
        environment_kwargs=ENV_KWARGS,
        observation_space=OBS_SPACE,
        n_training_episodes=10000,  # Number of training episodes
        max_steps=99,  # Max steps per episode
        learning_rate=0.7,  # Learning rate (alpha)
        gamma=0.95,  # Discount factor
        epsilon_start=1.0,  # Probability of choosing a random action at the start of training
        epsilon_end=0.05,  # Minimum probability of choosing a random action at the end of training
        epsilon_decay=0.0005,  # Decay rate for epsilon (how quickly it decreases)
        n_eval_episodes=100,  # Number of evaluation episodes
        n_bins=10,  # Number of bins for discretization (only for continuous observation spaces)
    )

    main(config)
