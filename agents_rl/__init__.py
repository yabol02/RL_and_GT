"""
RL Agents Package

This package contains implementations of tabular reinforcement learning algorithms:
- Monte Carlo (episode-based):
    Monte Carlo methods learn from complete episodes using actual returns.
    Updates Q-values using: Q(s,a) := Q(s,a) + α[G - Q(s,a)] where G is the actual return from that state-action pair.
- SARSA (TD on-policy):
    State-Action-Reward-State-Action (SARSA) is a TD(0) on-policy algorithm that learns the value of the policy it is actually following (including exploration).
    Updates Q-values using: Q(s,a) := Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
- Q-Learning (TD off-policy):
    Q-Learning is a TD(0) off-policy algorithm that learns the optimal action-value function by taking the maximum Q-value over all possible next actions.
    Updates Q-values using: Q(s,a) := Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]

All agents inherit from BaseRLAgent and share common functionality.
"""

from .base import BaseRLAgent, ObservationDiscretizer
from .mc_agent import MonteCarloAgent
from .q_agent import QLearningAgent
from .sarsa_agent import SARSAAgent

__all__ = [
    "BaseRLAgent",
    "ObservationDiscretizer",
    "QLearningAgent",
    "SARSAAgent",
    "MonteCarloAgent",
]
