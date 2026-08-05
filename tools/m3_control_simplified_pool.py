"""M3 对照实验：简版引擎在 G9 vanilla 卡池上训练 30k 局。

目的：区分"52% 是卡池天花板"还是"hearthstone_os 移植有 bug"。
简版引擎（hearthstone/）的 PPO 训练代码不动，只把卡池从全池换成
G9 子集（hearthstone_os.decks 的 15 种 vanilla × 2 镜像），其余配置
（lr/batch/λ/层数）与 hearthstone_os.train 一致。

用法：
    .venv/bin/python -m tools.m3_control_simplified_pool
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional, Tuple

import torch

from hearthstone import game as hs_game
from hearthstone.arena import final_reward
from hearthstone.cards import CARD_INDEX, POOL
from hearthstone.policy import PolicyAgent, Step, UnifiedNet, save_agent
from hearthstone.train import _update, build_opponent

from hearthstone_os import decks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", default="/tmp/m3_control_simplified.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # G9 vanilla 卡池（简版引擎侧）
    names = [decks.SUBSET_MAP[cid] for cid in decks.VANILLA_IDS]
    deck = [POOL[CARD_INDEX[n]] for n in names for _ in range(2)]
    decklists = [deck, deck]

    net = UnifiedNet(hidden=128, layers=2, norm="layer", oracle_dim=0)
    print(f"简版引擎 G9 卡池训练 {args.episodes} 局（{net.n_params:,} 参数）")
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    agent = PolicyAgent(net, training=True, seed=args.seed)
    batch_steps: List[Step] = []
    batch_episodes: List[Tuple[int, float]] = []
    recent: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        seat = episode % 2
        opponent = build_opponent("rule", rng)
        players = [opponent, opponent]
        players[seat] = agent

        agent.trajectory.clear()
        result = hs_game.play_game(players, rng=rng, decklists=decklists)
        reward = final_reward(result, seat)
        recent.append(int(result.winner == seat))

        steps = agent.trajectory.steps
        if steps:
            batch_steps.extend(steps)
            batch_episodes.append((len(steps), reward))

        if episode % 8 == 0 and batch_steps:
            _update(optimizer, net, batch_steps, batch_episodes,
                    argparse.Namespace(algo="ppo", ppo_epochs=4, clip_ratio=0.2,
                                       lr=1e-3, gamma=0.99, gae_lambda=0.5,
                                       entropy_coef=0.01, value_coef=0.5,
                                       clip=5.0, device="cpu"),
                    "cpu")
            batch_steps, batch_episodes = [], []

        if episode % 5000 == 0:
            window = recent[-2000:]
            elapsed = time.time() - started
            print(f"第 {episode:>6} 局  近 2000 局胜率 "
                  f"{sum(window) / len(window) * 100:5.1f}%  用时 {elapsed:5.1f}s")

    if args.save:
        save_agent(args.save, net, meta={"pool": "G9-vanilla", "engine": "simplified"})
        print(f"已保存 {args.save}")


if __name__ == "__main__":
    main()
