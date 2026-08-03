#!/usr/bin/env python3
"""训练一个炉石智能体，目标是打赢人为构造的规则算法。

用法：
    python -m hearthstone.train                                  # 默认 PPO 跑 2000 局
    python -m hearthstone.train --episodes 50000                 # 出正式结果时拉长
    python -m hearthstone.train --opponent mix --save hearthstone/models/agent.pt
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .arena import evaluate_all, final_reward, format_table
from .bots import make_bot
from .game import play_game
from .policy import (
    NORMS,
    PolicyAgent,
    Step,
    UnifiedNet,
    discounted_returns,
    evaluate_batch,
    make_batch,
    save_agent,
)

OPPONENT_MIX = ["random", "greedy", "rule"]


def build_opponent(kind: str, rng: random.Random):
    """造一个规则对手。`mix` 表示每局随机抽。"""
    name = rng.choice(OPPONENT_MIX) if kind == "mix" else kind
    return make_bot(name, seed=rng.randrange(1 << 30))


def train(args: argparse.Namespace) -> PolicyAgent:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    net = UnifiedNet(hidden=args.hidden, layers=args.layers, norm=args.norm,
                     residual=args.residual).to(device)
    if not args.quiet:
        res_str = " + 残差" if args.residual else ""
        print(f"UnifiedNet: {args.layers} 层 x {args.hidden} 宽，归一化 {args.norm}{res_str}，"
              f"共 {net.n_params:,} 个参数")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = PolicyAgent(net, device=device, training=True, seed=args.seed)

    batch_steps: List[Step] = []
    batch_returns: List[np.ndarray] = []

    recent_wins: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        # 先后手轮换——和 arena.evaluate 一样的口径
        seat = episode % 2
        opponent = build_opponent(args.opponent, rng)
        players = [opponent, opponent]
        players[seat] = agent

        agent.trajectory.clear()
        result = play_game(players, rng=rng)
        reward = final_reward(result, seat)
        recent_wins.append(int(result.winner == seat))

        steps = agent.trajectory.steps
        if steps:
            batch_steps.extend(steps)
            batch_returns.append(discounted_returns(reward, len(steps), args.gamma))

        if episode % args.batch == 0 and batch_steps:
            _update(optimizer, net, batch_steps, batch_returns, args, device)
            batch_steps, batch_returns = [], []

        if not args.quiet and episode % args.log_every == 0:
            window = recent_wins[-args.log_every:]
            elapsed = time.time() - started
            print(
                f"第 {episode:>6} 局  近 {len(window)} 局胜率 {sum(window) / len(window) * 100:5.1f}%"
                f"  用时 {elapsed:5.1f}s  ({episode / elapsed:5.1f} 局/秒)"
            )

        if not args.quiet and args.eval_every and episode % args.eval_every == 0:
            stats = evaluate_all(agent.eval_agent(), games=args.eval_games, seed=12345)
            print(f"  [评测 @ {episode}]")
            print(format_table(stats))
            agent.training = True

    return agent


def _update(optimizer, net, steps, returns, args, device) -> None:
    batch = make_batch(steps, device)
    target = torch.from_numpy(np.concatenate(returns)).to(device)

    with torch.no_grad():
        _, _, old_values = evaluate_batch(net, batch)
    advantage = target - old_values
    if advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    epochs = args.ppo_epochs if args.algo == "ppo" else 1
    for _ in range(epochs):
        log_prob, entropy, values = evaluate_batch(net, batch)

        if args.algo == "ppo":
            ratio = torch.exp(log_prob - batch.old_log_probs)
            clipped = torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
            policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
        else:
            policy_loss = -(log_prob * advantage).mean()

        value_loss = F.mse_loss(values, target)
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=2000,
                        help="训练局数（默认 2000，够快速验证想法；要正式结果再拉长）")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"],
                        help="训练算法（默认 ppo）")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 每批数据重复训练几轮")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO 的概率比裁剪幅度")
    parser.add_argument("--opponent", default="rule",
                        choices=["random", "greedy", "rule", "mix"], help="训练对手")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--hidden", type=int, default=128, help="打分网络隐藏层宽度")
    parser.add_argument("--layers", type=int, default=2, help="打分网络隐藏层数量")
    parser.add_argument("--norm", default="layer", choices=list(NORMS),
                        help="归一化方式（默认 layer）")
    parser.add_argument("--residual", action="store_true", help="隐藏层之间加残差连接")
    parser.add_argument("--batch", type=int, default=8, help="多少局更新一次")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵奖励系数")
    parser.add_argument("--value-coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument("--log-every", type=int, default=200, help="多少局打印一次训练胜率")
    parser.add_argument("--eval-every", type=int, default=0, help="多少局评测一次（0=不评）")
    parser.add_argument("--eval-games", type=int, default=200, help="每次评测打多少局")
    parser.add_argument("--final-eval-games", type=int, default=400, help="训练结束后的评测局数")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--quiet", action="store_true", help="不打印训练过程")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--save", metavar="PATH", help="把训练好的权重存到这里")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print(f"算法: {args.algo}   训练对手: {args.opponent}   局数: {args.episodes}   设备: {args.device}")
    print("双人游戏，先后手轮换，随机基准胜率 50%\n")

    agent = train(args)

    print("\n最终评测（贪心策略，先后手轮换）")
    stats = evaluate_all(agent.eval_agent(), games=args.final_eval_games, seed=999)
    print(format_table(stats))

    print("\n规则对手互相对打作为参照")
    from .arena import evaluate

    for a, b in [("greedy", "rule"), ("rule", "greedy"), ("rule", "rule")]:
        stat = evaluate(make_bot(a, seed=999), b, games=args.final_eval_games, seed=999)
        print(f"  {a:<7} vs {b:<7} {stat}")

    if args.save:
        save_agent(args.save, agent.net, meta=vars(args))
        print(f"\n权重已保存到 {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
