from .base import BaseRLAgent


class QLearningAgent(BaseRLAgent):
    """
    Q-Learning agent for discrete action spaces.

    Q-Learning is a TD(0) off-policy algorithm that learns the optimal action-value
    function by taking the maximum Q-value over all possible next actions.

    Key characteristics:
    - Updates after each step (not at episode end)
    - Off-policy: learns optimal policy while following ε-greedy
    - Uses max Q(s',a') for updates (optimistic)
    """

    def update_q_value(
        self, state: int, action: int, reward: float, next_state: int, done: bool
    ) -> None:
        """
        Update Q-value using the Q-Learning update rule.

        Q(s,a) := Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

        This is off-policy because it uses max Q(s',a') regardless of which
        action would actually be taken by the current ε-greedy policy.

        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode terminated
        """
        current_q = self.q_table[state][action]

        if done:
            # No future rewards if episode terminated
            max_next_q = 0.0
        else:
            # Off-policy: use max Q-value (optimal action)
            max_next_q = max(self.q_table[next_state])

        # Q-Learning update rule
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        self.q_table[state][action] = new_q

    def train_episode(self, epsilon: float, max_steps: int) -> float:
        """
        Train the agent for one episode using Q-Learning.

        :param epsilon: Current exploration probability
        :param max_steps: Maximum steps per episode
        :return: Total reward for the episode
        """
        observation, info = self.env.reset()
        state = self._get_state(observation)

        episode_reward = 0.0

        for step in range(max_steps):
            action = self.select_action_epsilon_greedy(state, epsilon)

            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_state = self._get_state(next_observation)

            self.update_q_value(
                state, action, reward, next_state, terminated or truncated
            )
            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state

        return episode_reward
