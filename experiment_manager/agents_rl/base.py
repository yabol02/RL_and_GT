import inspect
import json
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import yaml
from tqdm import tqdm


class ObservationDiscretizer:
    """
    Discretizes continuous observation spaces into discrete bins.

    This is useful for environments with continuous observations (like LunarLander)
    where we need a discrete state space for Q-table representation.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Box,
        n_bins: Union[int, np.ndarray] = 10,
        normalize: bool = False,
    ):
        """
        Initialize the discretizer.

        :param obs_space: The continuous observation space to discretize
        :param n_bins: Number of bins per dimension (int for uniform, array for per-dimension)
        :param normalize: If True, normalize observations to [0, 1] before discretization
        """
        self.obs_space = obs_space
        self.n_dims = obs_space.shape[0]
        self.normalize = normalize

        # Handle uniform or per-dimension bins
        if isinstance(n_bins, int):
            self.n_bins = np.full(self.n_dims, n_bins, dtype=int)
        else:
            self.n_bins = np.array(n_bins, dtype=int)

        # Store bounds for normalization
        self.low = np.array(
            [
                obs_space.low[i] if obs_space.low[i] != -np.inf else -1e6
                for i in range(self.n_dims)
            ]
        )
        self.high = np.array(
            [
                obs_space.high[i] if obs_space.high[i] != np.inf else 1e6
                for i in range(self.n_dims)
            ]
        )

        # Create bins for each dimension
        self.bins = []
        for i in range(self.n_dims):
            if self.normalize:
                # Bins in normalized space [0, 1]
                self.bins.append(np.linspace(0, 1, self.n_bins[i] - 1))
            else:
                # Bins in original space
                self.bins.append(
                    np.linspace(self.low[i], self.high[i], self.n_bins[i] - 1)
                )

        # Calculate total number of discrete states
        self.n_states = int(np.prod(self.n_bins))

    def discretize(self, observation: np.ndarray) -> int:
        """
        Convert a continuous observation to a discrete state index.

        :param observation: Continuous observation from the environment
        :return: Discrete state index
        """
        # Clip observation to bounds
        clipped_obs = np.clip(observation, self.low, self.high)

        # Normalize if requested
        if self.normalize:
            # Normalize to [0, 1] range
            obs_range = self.high - self.low
            # Avoid division by zero
            obs_range = np.where(obs_range == 0, 1.0, obs_range)
            normalized_obs = (clipped_obs - self.low) / obs_range
            values_to_discretize = normalized_obs
        else:
            values_to_discretize = clipped_obs

        # Discretize each dimension
        discrete_obs = []
        for i, (val, bins) in enumerate(zip(values_to_discretize, self.bins)):
            discrete_obs.append(np.digitize(val, bins))

        # Convert multi-dimensional discrete observation to single index
        state_index = 0
        multiplier = 1
        for i in range(len(discrete_obs) - 1, -1, -1):
            state_index += discrete_obs[i] * multiplier
            multiplier *= self.n_bins[i]

        return int(state_index)


class BaseRLAgent(ABC):
    """
    Base class for tabular RL agents.

    Provides common functionality for Q-Learning, SARSA, and Monte Carlo methods.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.7,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.0005,
        discretizer: Optional[ObservationDiscretizer] = None,
    ):
        """
        Initialize the base RL agent.

        :param env: Gymnasium environment
        :param learning_rate: Learning rate (alpha) for value updates
        :param gamma: Discount factor for future rewards
        :param epsilon_start: Initial exploration probability
        :param epsilon_end: Minimum exploration probability
        :param epsilon_decay: Exponential decay rate for exploration probability
        :param discretizer: Optional discretizer for continuous observation spaces
        """
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.discretizer = discretizer

        # Determine state and action spaces
        self.action_space_size = env.action_space.n

        if discretizer is not None:
            self.state_space_size = discretizer.n_states
        else:
            self.state_space_size = env.observation_space.n

        # Initialize Q-table
        self.q_table = self._initialize_q_table()

    def _initialize_q_table(self) -> np.ndarray:
        """
        Initialize the Q-table with zeros.

        :return: Initialized Q-table of shape (state_space, action_space)
        """
        return np.zeros((self.state_space_size, self.action_space_size))

    def _get_state(self, observation: Union[int, np.ndarray]) -> int:
        """
        Convert observation to discrete state index.

        :param observation: Raw observation from environment
        :return: Discrete state index
        """
        if self.discretizer is not None:
            return self.discretizer.discretize(observation)
        return int(observation)

    def _calculate_epsilon(self, episode: int) -> float:
        """
        Calculate epsilon value for current episode using exponential decay.

        :param episode: Current episode number
        :return: Epsilon value for exploration
        """
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self.epsilon_decay * episode
        )
        return epsilon

    def select_action_greedy(self, state: int) -> int:
        """
        Select action using greedy policy (exploitation only).

        :param state: Current discrete state
        :return: Action with highest Q-value
        """
        return int(np.argmax(self.q_table[state]))

    def select_action_epsilon_greedy(self, state: int, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.

        :param state: Current discrete state
        :param epsilon: Exploration probability
        :return: Selected action
        """
        if np.random.random() > epsilon:
            return self.select_action_greedy(state)  # Exploitation: choose best action
        else:
            return self.env.action_space.sample()  # Exploration: choose random action

    @abstractmethod
    def train_episode(self, epsilon: float, max_steps: int) -> float:
        """
        Train the agent for one episode.

        :param epsilon: Current exploration probability
        :param max_steps: Maximum steps per episode
        :return: Total reward for the episode
        """
        pass

    def train(
        self, n_episodes: int, max_steps: int = 99, verbose: bool = True
    ) -> np.ndarray:
        """
        Train the agent.

        :param n_episodes: Number of training episodes
        :param max_steps: Maximum steps per episode
        :param verbose: Whether to show progress bar
        :return: Array of episode rewards
        """
        episode_rewards = []

        iterator = (
            tqdm(range(n_episodes), desc=f"Training {self.__class__.__name__}")
            if verbose
            else range(n_episodes)
        )

        for episode in iterator:
            # Calculate current epsilon
            epsilon = self._calculate_epsilon(episode)

            # Train for one episode
            episode_reward = self.train_episode(epsilon, max_steps)
            episode_rewards.append(episode_reward)

            # Update progress bar with recent performance
            if verbose and episode > 0 and episode % 100 == 0:
                recent_avg = np.mean(episode_rewards[-100:])
                iterator.set_postfix(
                    {"avg_reward_100": f"{recent_avg:.2f}", "epsilon": f"{epsilon:.3f}"}
                )

        return np.array(episode_rewards)

    def evaluate(
        self,
        n_episodes: int = 100,
        max_steps: int = 99,
        render: bool = False,
        verbose: bool = True,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Evaluate the trained agent.

        :param n_episodes: Number of evaluation episodes
        :param max_steps: Maximum steps per episode
        :param render: Whether to render the environment
        :param verbose: Whether to show progress bar
        :return: Tuple of (mean_reward, std_reward, episode_rewards)
        """
        episode_rewards = []

        iterator = (
            tqdm(range(n_episodes), desc="Evaluating") if verbose else range(n_episodes)
        )

        for episode in iterator:
            observation, info = self.env.reset()
            state = self._get_state(observation)

            episode_reward = 0.0

            for step in range(max_steps):
                # Always use greedy policy during evaluation
                action = self.select_action_greedy(state)

                if render:
                    self.env.render()

                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )
                next_state = self._get_state(next_observation)

                episode_reward += reward

                if terminated or truncated:
                    break

                state = next_state

            episode_rewards.append(episode_reward)

        episode_rewards = np.array(episode_rewards)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward, episode_rewards

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the Q-table to a file.

        :param filepath: Path to save the Q-table
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.q_table, f)

        print(f"Q-table saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseRLAgent":
        """
        Load a trained agent from a file or experiment directory.

        If `filepath` is a directory, it must contain a `q_table.pkl` file and at
        least one config file (`config.yaml`, or `config.json`).
        If `filepath` is a file, it is treated as the Q-table and the config is
        searched in the same directory.

        :param filepath: Path to an experiment directory or to `q_table.pkl`
        :return: Reconstructed agent instance with loaded Q-table
        """
        filepath = Path(filepath)

        if filepath.is_dir():
            experiment_dir = filepath
            q_table_path = experiment_dir / "q_table.pkl"
        else:
            q_table_path = filepath
            experiment_dir = q_table_path.parent

        if not q_table_path.exists():
            raise FileNotFoundError(f"Q-table file not found: {q_table_path}")

        config_path = None
        for candidate_name in ("config.yaml", "config.json"):
            candidate = experiment_dir / candidate_name
            if candidate.exists():
                config_path = candidate
                break

        if config_path is None:
            raise FileNotFoundError(
                "Configuration file not found. Expected one of: config.yaml, config.json"
            )

        if config_path.suffix.lower() in {".yaml", ".yml"}:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

        if not isinstance(config_data, dict):
            raise ValueError(f"Invalid config format in {config_path}")

        env_name = config_data.get("environment_name")
        if env_name is None:
            raise ValueError(
                f"Missing 'environment_name' in configuration file: {config_path}"
            )

        env_kwargs = config_data.get(
            "env_kwargs", config_data.get("environment_kwargs", {})
        )
        if env_kwargs is None:
            env_kwargs = {}

        algorithm_cfg = config_data.get("algorithm", config_data)
        if not isinstance(algorithm_cfg, dict):
            raise ValueError(
                "Invalid 'algorithm' section in configuration. Expected a dictionary"
            )

        learning_rate = algorithm_cfg.get("learning_rate", 0.7)
        gamma = algorithm_cfg.get("gamma", 0.95)
        epsilon_start = algorithm_cfg.get("epsilon_start", 1.0)
        epsilon_end = algorithm_cfg.get("epsilon_end", 0.05)
        epsilon_decay = algorithm_cfg.get("epsilon_decay", 0.0005)
        n_bins = algorithm_cfg.get("n_bins", 10)
        use_first_visit = algorithm_cfg.get("use_first_visit", True)

        env = gym.make(env_name, **env_kwargs)

        discretizer = None
        if isinstance(env.observation_space, gym.spaces.Box):
            discretizer = ObservationDiscretizer(env.observation_space, n_bins=n_bins)

        init_kwargs = {
            "env": env,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "discretizer": discretizer,
        }

        signature = inspect.signature(cls.__init__)
        if "use_first_visit" in signature.parameters:
            init_kwargs["use_first_visit"] = use_first_visit

        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)

        instance = cls(**init_kwargs)

        expected_shape = (instance.state_space_size, instance.action_space_size)
        if getattr(q_table, "shape", None) != expected_shape:
            raise ValueError(
                f"Loaded Q-table shape {getattr(q_table, 'shape', None)} does not match expected {expected_shape}"
            )

        instance.q_table = q_table

        print(f"Q-table loaded from {q_table_path}")
        print(f"Configuration loaded from {config_path}")

        return instance

    def get_policy(self) -> np.ndarray:
        """
        Extract the greedy policy from the Q-table.

        :return: Array of shape (state_space,) with best action for each state
        """
        return np.argmax(self.q_table, axis=1)

    def print_statistics(self) -> None:
        """Print statistics about the Q-table."""
        print("\n" + "=" * 50)
        print(f"{self.__class__.__name__} Q-TABLE STATISTICS")
        print("=" * 50)
        print(f"State space size: {self.state_space_size}")
        print(f"Action space size: {self.action_space_size}")
        print(f"Q-table shape: {self.q_table.shape}")
        print(f"Q-table mean: {np.mean(self.q_table):.4f}")
        print(f"Q-table std: {np.std(self.q_table):.4f}")
        print(f"Q-table min: {np.min(self.q_table):.4f}")
        print(f"Q-table max: {np.max(self.q_table):.4f}")
        print(
            f"Non-zero entries: {np.count_nonzero(self.q_table)} / {self.q_table.size}"
        )
        print("=" * 50 + "\n")
