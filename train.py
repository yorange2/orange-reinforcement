#!/usr/bin/env python3
"""Train a Q-learning agent on the grid world and show what it learned.

Usage:
    python train.py                       # train with the defaults
    python train.py --episodes 2000       # train longer
    python train.py --save qtable.json    # keep the trained Q-table
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

from orange_rl import GridWorld, QLearningAgent
from orange_rl.gridworld import ACTION_ARROWS, ACTION_NAMES


def train(
    env: GridWorld,
    agent: QLearningAgent,
    episodes: int,
    log_every: int = 100,
    verbose: bool = True,
) -> List[float]:
    """Run `episodes` training episodes and return the reward for each one."""
    rewards: List[float] = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total += reward

        agent.decay_epsilon()
        rewards.append(total)

        if verbose and log_every and episode % log_every == 0:
            window = rewards[-log_every:]
            avg = sum(window) / len(window)
            print(f"episode {episode:>5}  avg reward {avg:>7.2f}  epsilon {agent.epsilon:.3f}")

    return rewards


def evaluate(env: GridWorld, agent: QLearningAgent) -> Tuple[float, List[Tuple[int, int]], str]:
    """Run one greedy episode. Returns `(total reward, path, outcome)`."""
    state = env.reset()
    path = [state]
    total = 0.0
    done = False
    outcome = "moving"

    while not done:
        action = agent.act(state, explore=False)
        state, reward, done, info = env.step(action)
        path.append(state)
        total += reward
        outcome = str(info["outcome"])

    return total, path, outcome


def render_policy(env: GridWorld, policy: Dict[Tuple[int, int], int]) -> str:
    """Draw the greedy policy over the map: arrows on open cells."""
    rows = []
    for r in range(env.rows):
        cells = []
        for c in range(env.cols):
            pos = (r, c)
            if pos in env.walls:
                cells.append("#")
            elif pos == env.goal:
                cells.append("G")
            elif pos in env.pits:
                cells.append("X")
            elif pos in policy:
                cells.append(ACTION_ARROWS[policy[pos]])
            else:
                cells.append(".")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=500, help="training episodes (default: 500)")
    parser.add_argument("--alpha", type=float, default=0.1, help="learning rate (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor (default: 0.95)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="initial exploration rate (default: 1.0)")
    parser.add_argument("--epsilon-decay", type=float, default=0.99, help="per-episode decay (default: 0.99)")
    parser.add_argument("--max-steps", type=int, default=100, help="step limit per episode (default: 100)")
    parser.add_argument("--log-every", type=int, default=100, help="progress log interval (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument("--save", metavar="PATH", help="write the trained Q-table to PATH")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    env = GridWorld(max_steps=args.max_steps)
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    print("Grid world (S start, G goal, X pit, # wall)")
    print(env.render(agent=env.start))
    print(f"\nTraining for {args.episodes} episodes...\n")

    rewards = train(env, agent, args.episodes, log_every=args.log_every)

    total, path, outcome = evaluate(env, agent)
    first, last = rewards[: args.log_every], rewards[-args.log_every :]

    print("\nLearned policy (greedy action per visited cell)")
    print(render_policy(env, agent.policy(env.states())))

    print(f"\nFirst {len(first)} episodes: avg reward {sum(first) / len(first):.2f}")
    print(f"Last  {len(last)} episodes: avg reward {sum(last) / len(last):.2f}")
    print(f"\nGreedy run: {outcome} in {len(path) - 1} steps, total reward {total:.1f}")
    print("Path: " + " -> ".join(str(p) for p in path))

    if args.save:
        agent.save(args.save)
        print(f"\nSaved Q-table to {args.save}")

    return 0 if outcome == "goal" else 1


if __name__ == "__main__":
    raise SystemExit(main())
