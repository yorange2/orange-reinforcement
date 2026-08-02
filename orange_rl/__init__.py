"""A tiny, dependency-free reinforcement learning playground."""

from .gridworld import GridWorld
from .q_learning import QLearningAgent

__all__ = ["GridWorld", "QLearningAgent"]
__version__ = "0.1.0"
