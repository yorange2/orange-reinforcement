#!/usr/bin/env python3
"""训练一个炉石智能体，目标是打赢人为构造的规则算法（路线图 M3）。

从 `hearthstone/train.py` 平移：PPO + GAE(λ=0.5)，AlphaZero 风格二合一网络。
差异：对局驱动换成 orange-stone（`hearthstone_os.arena.play_game`），终局
奖励直接读引擎的 `terminal_reward="health_scaled"`（M1-G7 口径），对手是
`hearthstone_os.bots` 的规则机器人。

用法：
    python -m hearthstone_os.train                                  # 默认 PPO 跑 2000 局
    python -m hearthstone_os.train --episodes 30000                 # 正式结果
    python -m hearthstone_os.train --save hearthstone_os/models/agent.pt
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .arena import evaluate_all, format_table, play_game
from .bots import BOTS
from .policy import (
    NORMS,
    PolicyAgent,
    Step,
    UnifiedNet,
    evaluate_batch,
    gae_advantages,
    make_batch,
    save_agent,
)

OPPONENT_MIX = ["random", "greedy", "rule"]


def build_opponent(kind: str, rng: random.Random):
    """造一个规则对手。`mix` 表示每局随机抽。"""
    name = rng.choice(OPPONENT_MIX) if kind == "mix" else kind
    return BOTS[name](seed=rng.randrange(1 << 30))


def train(args: argparse.Namespace) -> PolicyAgent:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    net = UnifiedNet(hidden=args.hidden, layers=args.layers, norm=args.norm,
                     residual=args.residual).to(device)
    if not args.quiet:
        res_str = " + 残差" if args.residual else ""
        print(f"UnifiedNet: {args.layers} 层 x {args.hidden} 宽，归一化 {args.norm}"
              f"{res_str}，共 {net.n_params:,} 个参数")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = PolicyAgent(net, device=device, training=True, seed=args.seed)

    batch_steps: List[Step] = []
    # 每局的 (步数, 终局奖励, 座位)——座位参与按座归一化，见 _update
    batch_episodes: List[Tuple[int, float, int]] = []

    recent_wins: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        # 先后手轮换——和 evaluate 一样的口径
        seat = 1 + episode % 2
        opponent = build_opponent(args.opponent, rng)
        players = [opponent, opponent]
        players[seat - 1] = agent

        agent.trajectory.clear()
        result = play_game(players, rng=rng, seed=rng.randrange(1 << 30))
        # 终局奖励直接读引擎（health_scaled 口径，M1-G7 验证过公式）
        reward = result.rewards[seat - 1]
        recent_wins.append(int(result.winner == seat))

        steps = agent.trajectory.steps
        if steps:
            batch_steps.extend(steps)
            batch_episodes.append((len(steps), reward, seat))

        if episode % args.batch == 0 and batch_steps:
            _update(optimizer, net, batch_steps, batch_episodes, args, device)
            batch_steps, batch_episodes = [], []

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


def _update(optimizer, net, steps, episodes, args, device) -> None:
    batch = make_batch(steps, device)

    with torch.no_grad():
        _, _, old_values = evaluate_batch(net, batch)

    # GAE 必须按局分别倒序递推，不能跨局——所以这里按 episodes 里的步数切开。
    # 优势只用旧策略的价值算一次，PPO 的多轮更新复用同一份，这是标准做法。
    values_np = old_values.cpu().numpy()
    advs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    step_seats: List[int] = []          # 每一步属于哪个座位（按步展开）
    offset = 0
    for n_steps, reward, seat in episodes:
        adv, tgt = gae_advantages(
            values_np[offset:offset + n_steps], reward, args.gamma, args.gae_lambda
        )
        advs.append(adv)
        targets.append(tgt)
        step_seats.extend([seat] * n_steps)
        offset += n_steps
    assert offset == len(values_np), f"步数对不上：{offset} != {len(values_np)}"

    advantage = torch.from_numpy(np.concatenate(advs)).to(device)
    target = torch.from_numpy(np.concatenate(targets)).to(device)
    if advantage.numel() > 1:
        # 按座位分别归一化（M3 实测修正）：orange-stone 官方开局里先手第 1 回合
        # 不抽牌、后手有硬币+首抽，P1/P2 的开局不对称比简化引擎大得多。全局归一
        # 会把 P1 局的负优势全压成负分、P2 局全压成正分，策略学到"先手局面做什么
        # 都错"然后自我锁死（实测 P1 胜率塌到 3%、P2 98%）。按座位各自归一后，
        # 每个座位内部的最优决策才能拿到正优势。
        for seat in (1, 2):
            mask = torch.tensor([s == seat for s in step_seats], device=device)
            if mask.sum() > 1:
                seat_adv = advantage[mask]
                seat_adv = (seat_adv - seat_adv.mean()) / (seat_adv.std() + 1e-8)
                advantage[mask] = seat_adv

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
    parser.add_argument("--gae-lambda", type=float, default=0.5,
                        help="GAE 的 λ，在偏差和方差之间插值。1.0 是蒙特卡洛回报，"
                             "0 是单步 TD 残差。默认 0.5——在这个环境里实测最优")
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
        stat = evaluate(BOTS[a](seed=999), b, games=args.final_eval_games, seed=999)
        print(f"  {a:<7} vs {b:<7} {stat}")

    if args.save:
        save_agent(args.save, agent.net, meta=vars(args))
        print(f"\n权重已保存到 {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
