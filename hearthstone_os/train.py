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
from torch.distributions import Categorical

from .arena import evaluate_all, format_table, play_game
from .batched import BatchedEnv
from .bots import BOTS
from .decks import random_deck, vanilla
from .features import FEATURE_DIM, batch_features
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
        deck = None
        if args.pool == "full":
            deck = random_deck(rng)      # M5：每局从全经典构筑池随机组牌
        result = play_game(players, rng=rng, seed=rng.randrange(1 << 30),
                           deck=deck)
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


def _batch_sample(net, decisions, device):
    """给一批 agent 决策点批量打分采样（M4 批量训练）。

    `decisions` = [(obs, legal_actions, going_first), ...]；返回
    (indices, steps)——padded 矩阵一次前向，每局采样一个动作并记录 Step。
    """
    n = len(decisions)
    widest = max(len(legal) for _, legal, _ in decisions)
    dim = FEATURE_DIM
    features = torch.zeros(n, widest, dim, device=device)
    mask = torch.zeros(n, widest, dtype=torch.bool, device=device)
    for i, (obs, legal, going_first) in enumerate(decisions):
        rows = torch.from_numpy(batch_features(obs, legal, going_first))
        features[i, : rows.shape[0]] = rows
        mask[i, : rows.shape[0]] = True

    indices: List[int] = []
    steps: List[Step] = []
    with torch.no_grad():
        logits, _ = net(features, mask)
        dist = Categorical(logits=logits)
        sampled = dist.sample()
        log_probs = dist.log_prob(sampled)
        for i, (obs, legal, _gf) in enumerate(decisions):
            index = int(sampled[i].item())
            indices.append(index)
            if len(legal) > 1:
                steps.append(Step(
                    features=features[i].cpu(),  # 已 detach（no_grad）
                    action=index,
                    log_prob=float(log_probs[i]),
                ))
    return indices, steps


def train_parallel(args: argparse.Namespace) -> PolicyAgent:
    """批量采样训练（路线图 M4）：N 局同时推进。

    - 决策批量前向：所有 agent 决策点拼成一个 padded 矩阵，一次 forward；
    - 引擎工作走 `orange_stone.BatchEnv` 的单次批量调用（allow_threads 释放
      GIL，多块批量叠线程池可继续并行）；
    - 每局只有一个 GameEnv（结构化观测是行动方视角），省掉单局 Env 的
      双实例锁步。

    采样口径与 `train()` 一致（PPO 更新代码共用 `_update`）。
    """
    n = args.parallel
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    net = UnifiedNet(hidden=args.hidden, layers=args.layers, norm=args.norm,
                     residual=args.residual).to(device)
    if not args.quiet:
        print(f"UnifiedNet: {args.layers} 层 x {args.hidden} 宽，归一化 {args.norm}"
              f"，共 {net.n_params:,} 个参数（批量采样 ×{n}）")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = PolicyAgent(net, device=device, training=False, seed=args.seed)

    # 座位轮换：偶数局坐 P1，奇数局坐 P2
    perspectives = [0 if i % 2 == 0 else 1 for i in range(n)]
    seeds = [rng.randrange(1 << 30) for _ in range(n)]
    batch = BatchedEnv(n, vanilla(), seeds, perspectives=perspectives,
                       bot="none", terminal_reward="health_scaled")
    bots = [build_opponent(args.opponent, rng) for _ in range(n)]

    trajs: List[List[Step]] = [[] for _ in range(n)]       # 每局的轨迹
    batch_steps: List[Step] = []
    batch_episodes: List[Tuple[int, float, int]] = []
    recent_wins: List[int] = []
    episode = 0
    last_logged = -1
    started = time.time()

    for it in range(args.max_iterations):
        obs = batch.observe()
        legal = batch.legal_actions()
        active = batch.active_players()

        # agent 的决策点（行动方 == 该局的 perspective）批量打分采样
        dec_idx = [i for i in range(n)
                   if legal[i] and active[i] == perspectives[i]]
        decisions = [
            (obs[i], legal[i], 1.0 if active[i] == 0 else 0.0)
            for i in dec_idx
        ]
        if decisions:
            sampled_indices, sampled_steps = _batch_sample(net, decisions, device)
            for i, steps in zip(dec_idx, _split_by_env(sampled_steps, decisions)):
                trajs[i].extend(steps)
        else:
            # 边界情况：某一轮所有局都轮到对手行动（批量同步漂移）
            sampled_indices, sampled_steps = [], []

        # 按局号填动作下标；其余决策点交给规则对手
        indices = [0] * n
        for i, idx in zip(dec_idx, sampled_indices):
            indices[i] = idx
        for i in range(n):
            if legal[i] and active[i] != perspectives[i]:
                indices[i] = bots[i].choose(obs[i], legal[i]).index

        batch.step(indices)

        # 收尾完成的局：轨迹 + 终局奖励入更新批，重开新局
        done = batch.done()
        for i in range(n):
            if done[i]:
                episode += 1
                seat = perspectives[i] + 1
                reward = batch.last_reward(i)
                recent_wins.append(int(batch.winners()[i] == perspectives[i]))
                if trajs[i]:
                    batch_steps.extend(trajs[i])
                    batch_episodes.append((len(trajs[i]), reward, seat))
                    trajs[i] = []
                # 只重开这一局（BatchEnv.reset_one 只换 seed，不影响其他局）
                seeds[i] = rng.randrange(1 << 30)
                bots[i] = build_opponent(args.opponent, rng)
                batch.reset_one(i, seeds[i])

        if episode and episode % (args.batch * n) == 0 and batch_steps:
            _update(optimizer, net, batch_steps, batch_episodes, args, device)
            batch_steps, batch_episodes = [], []

        if (not args.quiet and episode and episode % args.log_every == 0
                and episode != last_logged):
            last_logged = episode
            window = recent_wins[-args.log_every:]
            elapsed = time.time() - started
            print(
                f"第 {episode:>6} 局  近 {len(window)} 局胜率 {sum(window) / len(window) * 100:5.1f}%"
                f"  用时 {elapsed:5.1f}s  ({episode / elapsed:5.1f} 局/秒)"
            )

        if episode >= args.episodes:
            break

    return agent


def _split_by_env(steps: List[Step], decisions: List) -> List[List[Step]]:
    """把 `_batch_sample` 的连续 Step 列表按决策点切回每局一份。"""
    out: List[List[Step]] = []
    idx = 0
    for _obs, legal, _gf in decisions:
        count = 1 if len(legal) > 1 else 0
        out.append(steps[idx:idx + count])
        idx += count
    return out


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
    parser.add_argument("--pool", default="vanilla", choices=["vanilla", "full"],
                        help="卡池：vanilla=15 种镜像（M3 口径），full=全经典构筑池随机组牌（M5）")
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
    parser.add_argument("--parallel", type=int, default=1,
                        help="批量采样并发局数（M4；>1 走 BatchedEnv 批量训练）")
    parser.add_argument("--max-iterations", type=int, default=2_000_000,
                        help="批量训练的迭代步数上限（默认足够大，由 episodes 控制局数）")
    parser.add_argument("--save", metavar="PATH", help="把训练好的权重存到这里")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print(f"算法: {args.algo}   训练对手: {args.opponent}   局数: {args.episodes}   设备: {args.device}")
    print("双人游戏，先后手轮换，随机基准胜率 50%\n")

    if args.parallel > 1:
        agent = train_parallel(args)
    else:
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
