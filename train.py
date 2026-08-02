#!/usr/bin/env python3
"""训练一个跑得快智能体，目标是打赢人为构造的规则算法。

用法：
    python train.py                                  # 默认对 rule 机器人训练
    python train.py --episodes 20000 --opponent mix  # 训练更久、对手随机混合
    python train.py --save models/agent.pt           # 保存权重
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from paodekuai.arena import evaluate_all, final_reward, format_table
from paodekuai.bots import make_bot
from paodekuai.game import play_game
from paodekuai.policy import (MoveScorer, PolicyAgent, ValueNet,
                              discounted_returns, save_agent)

OPPONENT_MIX = ["random", "greedy", "rule"]


def build_opponents(kind: str, rng: random.Random, episode: int) -> List:
    """造两个对手。`mix` 表示每局随机抽，避免只会打一种对手。"""
    if kind == "mix":
        names = [rng.choice(OPPONENT_MIX) for _ in range(2)]
    else:
        names = [kind, kind]
    return [make_bot(name, seed=rng.randrange(1 << 30)) for name in names]


def train(args: argparse.Namespace) -> PolicyAgent:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    scorer = MoveScorer(hidden=args.hidden, layers=args.layers).to(device)
    print(f"打分网络: {args.layers} 层 x {args.hidden} 宽，共 {scorer.n_params:,} 个参数")
    value = ValueNet().to(device)
    optimizer = torch.optim.Adam(
        list(scorer.parameters()) + list(value.parameters()), lr=args.lr
    )
    agent = PolicyAgent(scorer, value, device=device, training=True, seed=args.seed)

    batch_logps: List[torch.Tensor] = []
    batch_values: List[torch.Tensor] = []
    batch_entropy: List[torch.Tensor] = []
    batch_returns: List[np.ndarray] = []

    recent_wins: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        seat = episode % 3
        players = build_opponents(args.opponent, rng, episode)
        players.insert(seat, agent)

        agent.trajectory.clear()
        result = play_game(players, rng=rng)
        reward = final_reward(result, seat)
        recent_wins.append(int(result.winner == seat))

        steps = agent.trajectory.steps
        if steps:
            batch_logps.append(torch.stack([s.log_prob for s in steps]))
            batch_values.append(torch.stack([s.value for s in steps]))
            batch_entropy.append(torch.stack([s.entropy for s in steps]))
            batch_returns.append(discounted_returns(reward, len(steps), args.gamma))

        if episode % args.batch == 0 and batch_logps:
            _update(optimizer, batch_logps, batch_values, batch_entropy, batch_returns, args)
            batch_logps, batch_values, batch_entropy, batch_returns = [], [], [], []

        if episode % args.log_every == 0:
            window = recent_wins[-args.log_every :]
            elapsed = time.time() - started
            print(
                f"第 {episode:>6} 局  近 {len(window)} 局胜率 {sum(window) / len(window) * 100:5.1f}%"
                f"  用时 {elapsed:5.1f}s  ({episode / elapsed:4.1f} 局/秒)"
            )

        if args.eval_every and episode % args.eval_every == 0:
            stats = evaluate_all(agent.eval_agent(), games=args.eval_games, seed=12345)
            print(f"  [评测 @ {episode}]")
            print(format_table(stats))
            agent.training = True

    return agent


def _update(optimizer, logps, values, entropies, returns, args) -> None:
    """带基线的 REINFORCE：优势做批内标准化，价值网络用 MSE 拟合回报。"""
    log_prob = torch.cat(logps)
    value = torch.cat(values)
    entropy = torch.cat(entropies)
    target = torch.from_numpy(np.concatenate(returns)).to(value.device)

    advantage = target - value.detach()
    if advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    policy_loss = -(log_prob * advantage).mean()
    value_loss = F.mse_loss(value, target)
    entropy_bonus = entropy.mean()

    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for group in optimizer.param_groups for p in group["params"]], args.clip
    )
    optimizer.step()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=8000, help="训练局数（默认 8000）")
    parser.add_argument("--opponent", default="rule",
                        choices=["random", "greedy", "rule", "mix"], help="训练对手（默认 rule）")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--hidden", type=int, default=128, help="打分网络隐藏层宽度")
    parser.add_argument("--layers", type=int, default=2, help="打分网络隐藏层数量")
    parser.add_argument("--batch", type=int, default=16, help="多少局更新一次")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵奖励系数（鼓励探索）")
    parser.add_argument("--value-coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument("--log-every", type=int, default=500, help="多少局打印一次训练胜率")
    parser.add_argument("--eval-every", type=int, default=2000, help="多少局评测一次（0 表示不评）")
    parser.add_argument("--eval-games", type=int, default=300, help="每次评测打多少局")
    parser.add_argument("--final-eval-games", type=int, default=1000, help="训练结束后的评测局数")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--save", metavar="PATH", help="把训练好的权重存到这里")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print(f"训练对手: {args.opponent}   局数: {args.episodes}   设备: {args.device}")
    print("三家游戏，随机基准胜率 33.3%\n")

    agent = train(args)

    print("\n最终评测（贪心策略，轮转座位）")
    stats = evaluate_all(agent.eval_agent(), games=args.final_eval_games, seed=999)
    print(format_table(stats))

    print("\n规则对手互相对打作为参照")
    from paodekuai.arena import bot_vs_bot

    for a, b in [("greedy", "rule"), ("rule", "greedy"), ("rule", "rule")]:
        print(f"  {a:<7} vs 2x{b:<7} {bot_vs_bot(a, b, games=args.final_eval_games, seed=999)}")

    if args.save:
        save_agent(args.save, agent.scorer, agent.value, meta=vars(args))
        print(f"\n权重已保存到 {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
