"""A minimal grid-world environment with a Gym-like interface.

The agent starts at `S`, must reach the goal `G`, and should avoid the pits `X`.
Walls `#` block movement: bumping into one (or into the outer border) keeps the
agent where it is and still costs a step.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

State = Tuple[int, int]

# (row delta, column delta) for each action index.
ACTIONS: List[State] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES: List[str] = ["up", "down", "left", "right"]
ACTION_ARROWS: List[str] = ["^", "v", "<", ">"]

DEFAULT_LAYOUT: List[str] = [
    "S....",
    ".##.X",
    "....#",
    ".#X..",
    "..#.G",
]


class GridWorld:
    """Deterministic grid world.

    Rewards: `step_penalty` on every move, `goal_reward` for reaching `G`, and
    `pit_penalty` for falling into an `X`. Goals and pits end the episode.
    """

    def __init__(
        self,
        layout: List[str] | None = None,
        step_penalty: float = -1.0,
        goal_reward: float = 20.0,
        pit_penalty: float = -20.0,
        max_steps: int = 100,
    ) -> None:
        self.grid = list(layout if layout is not None else DEFAULT_LAYOUT)
        if not self.grid or not self.grid[0]:
            raise ValueError("layout must be a non-empty list of non-empty rows")
        if len({len(row) for row in self.grid}) != 1:
            raise ValueError("all layout rows must have the same width")

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.pit_penalty = pit_penalty
        self.max_steps = max_steps

        self.start = self._find_unique("S")
        self.goal = self._find_unique("G")
        self.walls = {pos for pos, cell in self._cells() if cell == "#"}
        self.pits = {pos for pos, cell in self._cells() if cell == "X"}

        self.state: State = self.start
        self.steps = 0

    # ------------------------------------------------------------------ setup

    def _cells(self):
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                yield (r, c), cell

    def _find_unique(self, marker: str) -> State:
        found = [pos for pos, cell in self._cells() if cell == marker]
        if len(found) != 1:
            raise ValueError(f"layout must contain exactly one '{marker}', found {len(found)}")
        return found[0]

    # -------------------------------------------------------------- interface

    @property
    def n_actions(self) -> int:
        return len(ACTIONS)

    def states(self) -> List[State]:
        """Every state the agent can legally occupy."""
        return [pos for pos, _ in self._cells() if pos not in self.walls]

    def reset(self) -> State:
        """Start a new episode and return the initial state."""
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, object]]:
        """Apply `action` and return `(next_state, reward, done, info)`."""
        if not 0 <= action < len(ACTIONS):
            raise ValueError(f"action must be in [0, {len(ACTIONS) - 1}], got {action!r}")

        dr, dc = ACTIONS[action]
        row, col = self.state
        candidate = (row + dr, col + dc)
        if self._is_walkable(candidate):
            self.state = candidate

        self.steps += 1
        reward = self.step_penalty
        done = False
        outcome = "moving"

        if self.state == self.goal:
            reward += self.goal_reward
            done = True
            outcome = "goal"
        elif self.state in self.pits:
            reward += self.pit_penalty
            done = True
            outcome = "pit"
        elif self.steps >= self.max_steps:
            done = True
            outcome = "timeout"

        return self.state, reward, done, {"outcome": outcome, "steps": self.steps}

    def _is_walkable(self, pos: State) -> bool:
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return pos not in self.walls

    # ---------------------------------------------------------------- display

    def render(self, agent: State | None = None) -> str:
        """Return the grid as text, with `A` marking the agent's position."""
        agent = self.state if agent is None else agent
        rows = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                cell = self.grid[r][c]
                cells.append("A" if (r, c) == agent else ("." if cell == "S" else cell))
            rows.append(" ".join(cells))
        return "\n".join(rows)
