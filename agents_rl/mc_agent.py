from typing import List, Tuple

from .base import BaseRLAgent


class MonteCarloAgent(BaseRLAgent):
    """
    Monte Carlo agent for discrete action spaces.

    Monte Carlo methods wait until the end of an episode to update values,
    using the actual observed returns rather than bootstrapped estimates.

    Key characteristics:
    - Updates at episode end (not step-by-step)
    - Uses actual returns (no bootstrapping)
    - Higher variance, lower bias than TD methods
    - Requires episodic tasks (must terminate)
    - Supports both first-visit and every-visit variants.
    """

    def __init__(self, *args, use_first_visit: bool = True, **kwargs):
        """
        Initialize Monte Carlo agent.

        :param use_first_visit: If True, use first-visit MC. If False, use every-visit MC.
        :param args: Arguments passed to BaseRLAgent
        :param kwargs: Keyword arguments passed to BaseRLAgent
        """
        super().__init__(*args, **kwargs)
        self.use_first_visit = use_first_visit

    def calculate_returns(
        self, trajectory: List[Tuple[int, int, float]]
    ) -> List[Tuple[int, int, float]]:
        """
        Calculate discounted returns for each state-action pair in trajectory.

        Uses backwards iteration: G_t = r_{t+1} + γ·G_{t+1}

        :param trajectory: List of (state, action, reward) tuples
        :return: List of (state, action, return) tuples
        """
        returns = []
        G = 0.0

        # Iterate backwards through trajectory
        for state, action, reward in reversed(trajectory):
            G = reward + self.gamma * G
            returns.append((state, action, G))

        returns.reverse()
        return returns

    def update_q_values_first_visit(
        self, trajectory: List[Tuple[int, int, float]]
    ) -> None:
        """
        Update Q-values using first-visit Monte Carlo.

        Only updates Q(s,a) based on the first time (s,a) appears in the episode.
        This is the most common variant in practice.

        :param trajectory: List of (state, action, reward) tuples from episode
        """
        returns = self.calculate_returns(trajectory)
        visited = set()

        # Update Q-values for first visits only
        for state, action, G in returns:
            sa_pair = (state, action)

            if sa_pair not in visited:
                visited.add(sa_pair)
                # Incremental Monte Carlo update: Q(s,a) <- Q(s,a) + α[G - Q(s,a)]
                current_q = self.q_table[state][action]
                self.q_table[state][action] = current_q + self.learning_rate * (
                    G - current_q
                )

    def update_q_values_every_visit(
        self, trajectory: List[Tuple[int, int, float]]
    ) -> None:
        """
        Update Q-values using every-visit Monte Carlo.

        Updates Q(s,a) every time (s,a) appears in the episode.
        Can converge faster but may have higher variance.

        :param trajectory: List of (state, action, reward) tuples from episode
        """
        returns = self.calculate_returns(trajectory)

        # Update Q-values for every occurrence
        for state, action, G in returns:
            # Incremental Monte Carlo update: Q(s,a) <- Q(s,a) + α[G - Q(s,a)]
            current_q = self.q_table[state][action]
            self.q_table[state][action] = current_q + self.learning_rate * (
                G - current_q
            )

    def train_episode(self, epsilon: float, max_steps: int) -> float:
        """
        Train the agent for one episode using Monte Carlo.

        Collects full trajectory, then updates Q-values at the end.

        :param epsilon: Current exploration probability
        :param max_steps: Maximum steps per episode
        :return: Total reward for the episode
        """
        observation, info = self.env.reset()
        state = self._get_state(observation)

        # Store trajectory: (state, action, reward) tuples
        trajectory = []
        episode_reward = 0.0

        for step in range(max_steps):
            action = self.select_action_epsilon_greedy(state, epsilon)

            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_state = self._get_state(next_observation)

            trajectory.append((state, action, reward))
            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state

        # Update Q-values using full episode trajectory
        if self.use_first_visit:
            self.update_q_values_first_visit(trajectory)
        else:
            self.update_q_values_every_visit(trajectory)

        return episode_reward
