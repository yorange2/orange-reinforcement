"""Tabular Q-learning with an epsilon-greedy policy.

The update rule is the classic one:

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
"""

from __future__ import annotations

import json
import random
from typing import Dict, Hashable, List, Sequence


class QLearningAgent:
    """A Q-table backed agent. States only need to be hashable."""

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        if n_actions < 1:
            raise ValueError("n_actions must be at least 1")

        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random = random.Random(seed)
        self.q: Dict[Hashable, List[float]] = {}

    def q_values(self, state: Hashable) -> List[float]:
        """Q-values for `state`, created lazily on first visit."""
        if state not in self.q:
            self.q[state] = [0.0] * self.n_actions
        return self.q[state]

    def act(self, state: Hashable, explore: bool = True) -> int:
        """Pick an action: epsilon-greedy while exploring, greedy otherwise."""
        if explore and self.random.random() < self.epsilon:
            return self.random.randrange(self.n_actions)
        return self.greedy_action(state)

    def greedy_action(self, state: Hashable) -> int:
        """Best known action, ties broken randomly so the agent doesn't drift."""
        values = self.q_values(state)
        best = max(values)
        return self.random.choice([i for i, v in enumerate(values) if v == best])

    def learn(self, state: Hashable, action: int, reward: float, next_state: Hashable, done: bool) -> float:
        """Apply one Q-learning update and return the TD error."""
        target = reward if done else reward + self.gamma * max(self.q_values(next_state))
        values = self.q_values(state)
        td_error = target - values[action]
        values[action] += self.alpha * td_error
        return td_error

    def decay_epsilon(self) -> float:
        """Anneal exploration towards `epsilon_min`. Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def policy(self, states: Sequence[Hashable]) -> Dict[Hashable, int]:
        """Greedy action for each state the agent has actually visited."""
        return {s: self.greedy_action(s) for s in states if s in self.q}

    # ------------------------------------------------------------ persistence

    def save(self, path: str) -> None:
        """Write the Q-table to JSON (state keys are stringified)."""
        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q": {json.dumps(state): values for state, values in self.q.items()},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load an agent saved by `save`. State keys become tuples again."""
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)

        agent = cls(
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
        )
        for key, values in payload["q"].items():
            decoded = json.loads(key)
            agent.q[tuple(decoded) if isinstance(decoded, list) else decoded] = values
        return agent
