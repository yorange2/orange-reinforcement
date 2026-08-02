# orange-reinforcement

A small, dependency-free reinforcement learning project: a tabular **Q-learning**
agent that learns to cross a grid world, reach the goal, and avoid the pits.

Pure Python standard library — no NumPy, no Gym, no install step. The whole thing
is about 300 lines, so it's readable end to end.

## Quick start

```bash
git clone https://github.com/yorange2/orange-reinforcement.git
cd orange-reinforcement
python3 train.py
```

Output:

```
Grid world (S start, G goal, X pit, # wall)
A . . . .
. # # . X
. . . . #
. # X . .
. . # . G

Training for 500 episodes...

episode   100  avg reward  -27.54  epsilon 0.366
episode   200  avg reward    4.95  epsilon 0.134
episode   300  avg reward   10.26  epsilon 0.049
episode   400  avg reward   11.43  epsilon 0.018
episode   500  avg reward   11.89  epsilon 0.010

Learned policy (greedy action per visited cell)
> > > v <
v # # v X
> > > v #
> # X v v
< > # > G

First 100 episodes: avg reward -27.54
Last  100 episodes: avg reward 11.89

Greedy run: goal in 8 steps, total reward 12.0
Path: (0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> (2, 3) -> (3, 3) -> (4, 3) -> (4, 4)
```

The agent starts out wandering into pits (average reward `-27`), and after a few
hundred episodes walks the shortest safe route in 8 steps — the optimal path.

## The environment

| Symbol | Meaning | Reward |
| ------ | ------- | ------ |
| `S`    | start   | — |
| `G`    | goal, ends the episode | `+20` |
| `X`    | pit, ends the episode | `-20` |
| `#`    | wall, blocks movement | — |
| any move | | `-1` per step |

Four actions: up, down, left, right. Movement is deterministic; bumping into a
wall or the border wastes a step. Episodes also end after `--max-steps` moves.

The step penalty is what makes the agent prefer *short* routes — without it,
any path to the goal would look equally good.

## How it works

Q-learning keeps a table `Q[state][action]` estimating the total future reward of
taking an action in a state. After each move it nudges that estimate towards what
it just observed:

```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

- `alpha` (learning rate) — how much each new experience overwrites the old estimate.
- `gamma` (discount) — how much distant rewards count relative to immediate ones.
- `epsilon` (exploration) — probability of a random action instead of the best known
  one. It starts at `1.0` (pure exploration) and decays each episode towards `0.01`,
  so the agent explores early and exploits what it learned later.

## Options

```bash
python3 train.py --episodes 2000        # train longer
python3 train.py --alpha 0.5            # learn faster (and less stably)
python3 train.py --gamma 0.8            # care less about distant rewards
python3 train.py --epsilon-decay 0.999  # explore for longer
python3 train.py --seed 7               # different random run
python3 train.py --save qtable.json     # persist the trained Q-table
python3 train.py --help                 # everything else
```

## Use it as a library

```python
from orange_rl import GridWorld, QLearningAgent

env = GridWorld(["S..", ".#X", "..G"])          # your own map
agent = QLearningAgent(n_actions=env.n_actions, seed=0)

for _ in range(300):
    state, done = env.reset(), False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
    agent.decay_epsilon()

print(agent.policy(env.states()))                # greedy action per state
```

`GridWorld` follows the familiar `reset()` / `step(action) -> (state, reward, done, info)`
interface, and `QLearningAgent` works with any hashable state — so you can point
either half at something else.

## Layout

```
orange_rl/
  gridworld.py     # the environment: layout parsing, movement, rewards, rendering
  q_learning.py    # the agent: Q-table, epsilon-greedy policy, updates, save/load
train.py           # training loop, greedy evaluation, policy rendering, CLI
tests/             # unit tests for both halves, plus a learning-convergence test
```

## Tests

```bash
python3 -m unittest discover -s tests -t .
```

21 tests, no dependencies. The last one trains a real agent for 500 episodes and
asserts it actually reaches the goal and improves over time.

## Ideas to extend it

- **SARSA** — swap `max_a' Q(s', a')` for the Q-value of the action actually taken.
- **Stochastic moves** — make actions slip sideways with some probability; the
  agent should learn to give pits a wider berth.
- **Bigger or random maps** — `GridWorld` takes any layout, so generate one.
- **Function approximation** — replace the Q-table with a linear model or a small
  network once the state space gets too big to enumerate.
