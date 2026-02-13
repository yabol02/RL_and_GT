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
    - Can use first-visit or every-visit (this implements first-visit)
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize Monte Carlo agent.

        Adds visit counter for first-visit MC.
        """
        super().__init__(*args, **kwargs)
        # For first-visit MC: track which state-action pairs were visited
        self.returns = {
            (s, a): []
            for s in range(self.state_space_size)
            for a in range(self.action_space_size)
        }

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
            # Accumulate discounted return
            G = reward + self.gamma * G
            returns.append((state, action, G))

        # Reverse to get chronological order
        returns.reverse()
        return returns

    def update_q_values_first_visit(
        self, trajectory: List[Tuple[int, int, float]]
    ) -> None:
        """
        Update Q-values using first-visit Monte Carlo.

        Only updates Q(s,a) based on the first time (s,a) appears in the episode.

        :param trajectory: List of (state, action, reward) tuples from episode
        """
        # Calculate returns for all state-action pairs
        returns = self.calculate_returns(trajectory)

        # Track which state-action pairs we've seen
        visited = set()

        # Update Q-values for first visits only
        for state, action, G in returns:
            sa_pair = (state, action)

            if sa_pair not in visited:
                # First visit to this state-action pair
                visited.add(sa_pair)

                # Monte Carlo update
                current_q = self.q_table[state][action]
                new_q = current_q + self.learning_rate * (G - current_q)
                self.q_table[state][action] = new_q

    def train_episode(self, epsilon: float, max_steps: int) -> float:
        """
        Train the agent for one episode using Monte Carlo.

        Collects full trajectory, then updates Q-values at the end.

        :param epsilon: Current exploration probability
        :param max_steps: Maximum steps per episode
        :return: Total reward for the episode
        """
        # Reset environment
        observation, info = self.env.reset()
        state = self._get_state(observation)

        # Store trajectory: (state, action, reward) tuples
        trajectory = []
        episode_reward = 0.0

        for step in range(max_steps):
            # Select action using ε-greedy policy
            action = self.select_action_epsilon_greedy(state, epsilon)

            # Take action
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_state = self._get_state(next_observation)

            # Store transition
            trajectory.append((state, action, reward))
            episode_reward += reward

            # Check if episode ended
            if terminated or truncated:
                break

            state = next_state

        # Update Q-values using full episode trajectory
        self.update_q_values_first_visit(trajectory)

        return episode_reward
