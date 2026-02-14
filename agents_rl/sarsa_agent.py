from .base import BaseRLAgent


class SARSAAgent(BaseRLAgent):
    """
    SARSA agent for discrete action spaces.

    SARSA is a TD(0) on-policy algorithm that learns the value of the policy
    it is actually following (including exploration).

    Key characteristics:
    - Updates after each step (not at episode end)
    - On-policy: learns about the policy being followed
    - Uses Q(s',a') where a' is the action actually taken
    - More conservative than Q-Learning
    - Better for risky environments (considers exploration in learning)
    """

    def update_q_value(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool,
    ) -> None:
        """
        Update Q-value using the SARSA update rule.

        Q(s,a) := Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

        This is on-policy because it uses Q(s',a') where a' is the action
        that will actually be taken by the ε-greedy policy.

        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param next_action: Next action (actually selected by policy)
        :param done: Whether episode terminated
        """
        current_q = self.q_table[state][action]

        if done:
            # No future rewards if episode terminated
            next_q = 0.0
        else:
            # On-policy: use Q-value of action that will be taken
            next_q = self.q_table[next_state][next_action]

        # SARSA update rule
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        new_q = current_q + self.learning_rate * td_error

        self.q_table[state][action] = new_q

    def train_episode(self, epsilon: float, max_steps: int) -> float:
        """
        Train the agent for one episode using SARSA.

        :param epsilon: Current exploration probability
        :param max_steps: Maximum steps per episode
        :return: Total reward for the episode
        """
        observation, info = self.env.reset()
        state = self._get_state(observation)

        action = self.select_action_epsilon_greedy(state, epsilon)

        episode_reward = 0.0

        for step in range(max_steps):
            next_observation, reward, terminated, truncated, info = self.env.step(
                action
            )
            next_state = self._get_state(next_observation)

            next_action = self.select_action_epsilon_greedy(next_state, epsilon)

            self.update_q_value(
                state, action, reward, next_state, next_action, truncated
            )
            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        return episode_reward
