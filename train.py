#!/usr/bin/env python3
"""训练一个跑得快智能体，目标是打赢人为构造的规则算法。

用法：
    python train.py                                  # 默认 PPO 跑 2000 局，几秒钟出结果
    python train.py --episodes 120000                # 要正式结果时再拉长
    python train.py --opponent mix --save models/agent.pt

默认局数刻意压得很小：探索阶段要的是快速看方向对不对。PPO 跑 2000 局对 2xrule 就有约 49%
（跑满 12 万局也才 51%），足够判断一个改动有没有效果。
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from paodekuai.arena import evaluate_all, final_reward, format_table
from paodekuai.bots import make_bot
from paodekuai.encoding import ENCODERS, make_encoder
from paodekuai.game import play_game
from paodekuai.policy import (NORMS, MoveScorer, PolicyAgent, Step, ValueNet,
                              discounted_returns, evaluate_batch, make_batch,
                              save_agent)

OPPONENT_MIX = ["random", "greedy", "rule"]


def build_opponents(kind: str, rng: random.Random) -> List:
    """造两个规则对手。`mix` 表示每局随机抽，避免只会打一种对手。"""
    if kind == "mix":
        names = [rng.choice(OPPONENT_MIX) for _ in range(2)]
    else:
        names = [kind, kind]
    return [make_bot(name, seed=rng.randrange(1 << 30)) for name in names]


class OpponentPool:
    """自我对弈的对手池：装着模型过去若干个版本的快照。

    只跟"当前的自己"打有两个毛病：对手一直在变，等于追移动靶；而且两个相同的策略
    容易互相绕圈（我克制你、你克制我、转回原点）。所以定期存快照丢进池子，每局从
    池子里随机抽，等于跟自己的一整段历史打。

    另外按 `bot_ratio` 掺一些规则机器人保底，免得自我对弈把自己关进小房间里，
    练出一套只对自己人管用的打法。
    """

    def __init__(
        self,
        capacity: int,
        bot_ratio: float,
        snapshot_every: int,
        rng: random.Random,
        device: torch.device,
    ) -> None:
        self.capacity = capacity
        self.bot_ratio = bot_ratio
        self.snapshot_every = snapshot_every
        self.rng = rng
        self.device = device
        self.snapshots: List[PolicyAgent] = []

    def maybe_snapshot(self, episode: int, scorer: MoveScorer) -> bool:
        """到点了就把当前模型冻一份丢进池子，满了淘汰最老的。"""
        if episode % self.snapshot_every:
            return False

        frozen = copy.deepcopy(scorer).to(self.device)
        frozen.eval()
        for param in frozen.parameters():
            param.requires_grad_(False)

        # 快照对手要采样而不是取最高分，否则每局打得一模一样，提供不了多样的对局
        self.snapshots.append(
            PolicyAgent(frozen, device=self.device, training=False, sample=True)
        )
        if len(self.snapshots) > self.capacity:
            self.snapshots.pop(0)
        return True

    def draw(self) -> List:
        """抽两个对手。池子还空着的时候先用规则机器人顶上。"""
        if not self.snapshots:
            return build_opponents("mix", self.rng)

        opponents = []
        for _ in range(2):
            if self.rng.random() < self.bot_ratio:
                opponents.append(make_bot(self.rng.choice(OPPONENT_MIX),
                                          seed=self.rng.randrange(1 << 30)))
            else:
                opponents.append(self.rng.choice(self.snapshots))
        return opponents


def train(args: argparse.Namespace) -> PolicyAgent:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    encoder = make_encoder(args.features)
    grid = encoder.grid_slice if args.conv else None
    scorer = MoveScorer(dim=encoder.dim, hidden=args.hidden, layers=args.layers,
                        norm=args.norm, grid=grid, residual=args.residual).to(device)
    scorer.encoder_name = encoder.name
    if not args.quiet:
        print(f"输入: {encoder.name} 编码 {encoder.dim} 维"
              f"{'，点数网格走卷积' if grid else ''}")
        print(f"打分网络: {args.layers} 层 x {args.hidden} 宽，归一化 {args.norm}"
              f"{'，残差' if args.residual else ''}，共 {scorer.n_params:,} 个参数")
    value = ValueNet(dim=encoder.state_dim, norm=args.norm).to(device)
    optimizer = torch.optim.Adam(
        list(scorer.parameters()) + list(value.parameters()), lr=args.lr
    )
    agent = PolicyAgent(scorer, value, device=device, training=True, seed=args.seed,
                        encoder=encoder)

    pool = None
    if args.opponent == "self":
        pool = OpponentPool(args.pool_size, args.bot_ratio, args.snapshot_every, rng, device)

    batch_steps: List[Step] = []
    batch_returns: List[np.ndarray] = []

    recent_wins: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        seat = episode % 3
        if pool is not None:
            pool.maybe_snapshot(episode, scorer)
            players = pool.draw()
        else:
            players = build_opponents(args.opponent, rng)
        players.insert(seat, agent)

        agent.trajectory.clear()
        result = play_game(players, rng=rng)
        reward = final_reward(result, seat)
        recent_wins.append(int(result.winner == seat))

        steps = agent.trajectory.steps
        if steps:
            batch_steps.extend(steps)
            batch_returns.append(discounted_returns(reward, len(steps), args.gamma))

        if episode % args.batch == 0 and batch_steps:
            _update(optimizer, scorer, value, batch_steps, batch_returns, args, device,
                    encoder.state_offset)
            batch_steps, batch_returns = [], []

        if not args.quiet and episode % args.log_every == 0:
            window = recent_wins[-args.log_every :]
            elapsed = time.time() - started
            print(
                f"第 {episode:>6} 局  近 {len(window)} 局胜率 {sum(window) / len(window) * 100:5.1f}%"
                f"  用时 {elapsed:5.1f}s  ({episode / elapsed:4.1f} 局/秒)"
            )

        if not args.quiet and args.eval_every and episode % args.eval_every == 0:
            stats = evaluate_all(agent.eval_agent(), games=args.eval_games, seed=12345)
            print(f"  [评测 @ {episode}]")
            print(format_table(stats))
            agent.training = True

    return agent


def _update(optimizer, scorer, value_net, steps, returns, args, device, state_offset) -> None:
    """用一批轨迹更新一次网络。两种算法只差在策略损失和更新轮数上。

    REINFORCE：优势 x log π，一批数据只用一轮。
    PPO：看新旧策略在这个动作上的概率比 r，把 r 裁剪到 [1-ε, 1+ε] 再取较小的那项。
         裁剪相当于给更新幅度上了保险，所以同一批数据可以安全地反复用好几轮。
    """
    batch = make_batch(steps, device, state_offset)
    target = torch.from_numpy(np.concatenate(returns)).to(device)

    # 优势只算一次，用的是更新前的价值估计（PPO 的标准做法）
    with torch.no_grad():
        _, _, old_values = evaluate_batch(scorer, value_net, batch)
    advantage = target - old_values
    if advantage.numel() > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    epochs = args.ppo_epochs if args.algo == "ppo" else 1
    for _ in range(epochs):
        log_prob, entropy, values = evaluate_batch(scorer, value_net, batch)

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
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]], args.clip
        )
        optimizer.step()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=2000,
                        help="训练局数（默认 2000，够快速验证一个想法；出正式结果再拉长）")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"],
                        help="训练算法（默认 ppo）")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO 每批数据重复训练几轮")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO 的概率比裁剪幅度 ε")
    parser.add_argument("--opponent", default="rule",
                        choices=["random", "greedy", "rule", "mix", "self"],
                        help="训练对手（默认 rule；self 表示自我对弈）")
    parser.add_argument("--pool-size", type=int, default=8, help="自我对弈的对手池装几个快照")
    parser.add_argument("--snapshot-every", type=int, default=250, help="每多少局存一个快照")
    parser.add_argument("--bot-ratio", type=float, default=0.2,
                        help="自我对弈时掺多少比例的规则对手保底")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--hidden", type=int, default=128, help="打分网络隐藏层宽度")
    parser.add_argument("--layers", type=int, default=2, help="打分网络隐藏层数量")
    parser.add_argument("--features", default="handcrafted", choices=list(ENCODERS),
                        help="输入编码：handcrafted=手工特征，raw=原始点数网格，both=两者拼接")
    parser.add_argument("--residual", action="store_true",
                        help="隐藏层用残差块 x + f(x)")
    parser.add_argument("--conv", action="store_true",
                        help="原始编码的点数网格走一维卷积（慢 30 倍以上，默认展平进 MLP）")
    parser.add_argument("--norm", default="layer", choices=list(NORMS),
                        help="隐藏层归一化方式（默认 layer=LayerNorm；不提供 BatchNorm，原因见 policy.py）")
    parser.add_argument("--batch", type=int, default=16, help="多少局更新一次")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵奖励系数（鼓励探索）")
    parser.add_argument("--value-coef", type=float, default=0.5, help="价值损失系数")
    parser.add_argument("--clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument("--log-every", type=int, default=500, help="多少局打印一次训练胜率")
    parser.add_argument("--eval-every", type=int, default=0, help="多少局评测一次（0 表示不评）")
    parser.add_argument("--eval-games", type=int, default=300, help="每次评测打多少局")
    parser.add_argument("--final-eval-games", type=int, default=500, help="训练结束后的评测局数")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--quiet", action="store_true", help="不打印训练过程")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--save", metavar="PATH", help="把训练好的权重存到这里")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print(f"算法: {args.algo}   训练对手: {args.opponent}   局数: {args.episodes}   设备: {args.device}")
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
