#!/usr/bin/env python3
"""用 RosettaStone 真实引擎训练炉石智能体。

用法：
    python -m rosetta.train                                    # 默认 PPO 2000 局
    python -m rosetta.train --episodes 30000 --save rosetta/models/agent.pt
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from . import decks
from .arena import duel as arena_duel
from .bots import BOTS
from .env import ActionType, Env
from .features import STATE_DIM, STATE_OFFSET
from .policy import (
    Batch,
    PolicyAgent,
    Step,
    Trajectory,
    UnifiedNet,
    evaluate_batch,
    gae_advantages,
    make_batch,
    save_agent,
)

OPPONENT_MIX = ["random", "greedy", "rule"]


def build_opponent(kind: str, rng: random.Random, seed: int):
    """造一个规则对手。`mix` 表示每局随机抽。"""
    name = rng.choice(OPPONENT_MIX) if kind == "mix" else kind
    return BOTS[name](seed=seed)


def final_reward(winner: int, seat: int, obs) -> float:
    """终局奖励：赢 +1，平 0，输 −(对方剩血比例)。"""
    if winner == 0:
        return 0.0
    if winner == seat:
        return 1.0
    # 输了：看赢家剩多少血
    winner_health = (obs.me.hero_health if obs.me.hero_health > 0
                     else obs.opponent.hero_health)
    if winner_health <= 0:
        winner_health = obs.opponent.hero_health if obs.opponent.hero_health > 0 else 1
    return -winner_health / 30.0


def play_episode(agent, opponent_name: str, rng: random.Random,
                 episode: int, hero_class: str = "MAGE",
                 max_steps: int = 5000):
    """打一局并收集轨迹。返回 (steps_collected, reward)。"""
    dk = decks.vanilla()
    seed = rng.randrange(1 << 30)

    env = Env(player1_class=hero_class, player2_class=hero_class,
              player1_deck=dk, player2_deck=dk)
    env.reset(seed=seed)

    agent_seat = 1 if episode % 2 == 0 else 2      # 先后手轮换
    opp_seed = seed + 1
    opponent = build_opponent(opponent_name, rng, opp_seed)
    seats = {1: opponent, 2: opponent}
    seats[agent_seat] = agent

    agent.trajectory.clear()
    steps = 0
    while not env.done and steps < max_steps:
        actions = env.legal_actions()
        if not actions:
            break
        obs = env.observe()
        player = env.current_player
        env.step(seats[player].choose(obs, actions))
        steps += 1

    # 拿到终局面貌
    obs_final = env.observe()
    reward = final_reward(env.winner, agent_seat, obs_final)

    collected = len(agent.trajectory.steps)
    return collected, reward


def train(args: argparse.Namespace) -> PolicyAgent:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    net = UnifiedNet(state_dim=STATE_DIM, action_dim=STATE_OFFSET,
                     hidden=args.hidden, layers=args.layers,
                     norm=args.norm, residual=args.residual,
                     oracle_dim=0).to(device)
    if not args.quiet:
        res_str = " + 残差" if args.residual else ""
        print(f"UnifiedNet: {args.layers} 层 x {args.hidden} 宽，归一化 {args.norm}"
              f"{res_str}，共 {net.n_params:,} 个参数")
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent = PolicyAgent(net, device=device, training=True, seed=args.seed)

    batch_steps: List[Step] = []
    batch_episodes: List[Tuple[int, float]] = []

    recent_wins: List[int] = []
    started = time.time()

    for episode in range(1, args.episodes + 1):
        n_steps, reward = play_episode(
            agent, args.opponent, rng, episode,
            hero_class=args.hero, max_steps=args.max_steps)
        recent_wins.append(int(reward > 0))

        if n_steps:
            batch_steps.extend(agent.trajectory.steps)
            batch_episodes.append((n_steps, reward))

        if episode % args.batch == 0 and batch_steps:
            _update(optimizer, net, batch_steps, batch_episodes, args, device)
            batch_steps, batch_episodes = [], []

        if not args.quiet and episode % args.log_every == 0:
            window = recent_wins[-args.log_every:]
            elapsed = time.time() - started
            wr = sum(window) / len(window) * 100
            rate = episode / elapsed
            print(f"第 {episode:>6} 局  近 {len(window)} 局胜率 {wr:5.1f}%"
                  f"  用时 {elapsed:5.1f}s  ({rate:5.1f} 局/秒)")

    return agent


def _update(optimizer, net, steps, episodes, args, device) -> None:
    """PPO 更新。和 hearthstone/train.py 的 _update 逻辑一致。"""
    batch: Batch = make_batch(steps, device)

    with torch.no_grad():
        _, _, old_values = evaluate_batch(net, batch)

    values_np = old_values.cpu().numpy()
    advs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    offset = 0
    for n_steps, reward in episodes:
        adv, tgt = gae_advantages(
            values_np[offset:offset + n_steps], reward, args.gamma, args.gae_lambda
        )
        advs.append(adv)
        targets.append(tgt)
        offset += n_steps
    assert offset == len(values_np), f"步数对不上：{offset} != {len(values_np)}"

    advantage = torch.from_numpy(np.concatenate(advs)).to(device)
    target = torch.from_numpy(np.concatenate(targets)).to(device)
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


def evaluate(agent: PolicyAgent, opponent_name: str, games: int = 200,
             seed: int = 0, hero_class: str = "MAGE") -> dict:
    """评测 agent 打一个对手的胜率。先后手轮换。"""
    dk = decks.vanilla()
    wins = 0
    draws = 0
    total_steps = 0

    for g in range(games):
        bot_seed = seed + g * 2
        opponent = BOTS[opponent_name](seed=bot_seed + 1)
        agent_seat = 1 if g % 2 == 0 else 2

        env = Env(player1_class=hero_class, player2_class=hero_class,
                  player1_deck=dk, player2_deck=dk)
        env.reset(seed=seed + g)

        seats = {1: opponent, 2: opponent}
        seats[agent_seat] = agent.eval_agent()
        steps = 0
        while not env.done and steps < 5000:
            actions = env.legal_actions()
            if not actions:
                break
            obs = env.observe()
            env.step(seats[env.current_player].choose(obs, actions))
            steps += 1

        total_steps += steps
        if env.winner == 0:
            draws += 1
        elif env.winner == agent_seat:
            wins += 1

    return {
        "win_rate": wins / games,
        "draw_rate": draws / games,
        "avg_steps": total_steps / games,
        "games": games,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=2000,
                        help="训练局数（默认 2000）")
    parser.add_argument("--algo", default="ppo", choices=["ppo", "reinforce"])
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--opponent", default="rule",
                        choices=["random", "greedy", "rule", "mix"])
    parser.add_argument("--hero", default="MAGE", help="双方英雄职业")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--norm", default="layer", choices=["none", "layer"])
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--batch", type=int, default=8, help="多少局更新一次")
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", metavar="PATH", help="保存权重")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print(f"算法: {args.algo}   对手: {args.opponent}"
          f"   局数: {args.episodes}   设备: {args.device}")
    print("双人游戏，先后手轮换，随机基准 50%\n")

    agent = train(args)

    print("\n最终评测")
    for opp in ["random", "greedy", "rule"]:
        r = evaluate(agent.eval_agent(), opp, games=400, seed=99999,
                     hero_class=args.hero)
        print(f"  vs {opp:<7} {r['win_rate']*100:5.1f}%"
              f"  平局 {r['draw_rate']*100:4.1f}%"
              f"  {400/r.get('games',400)*r.get('games',400):.0f} 局")

    print("\n规则对手参照")
    for a_name, b_name in [("greedy", "rule"), ("rule", "greedy"),
                            ("rule", "rule"), ("random", "rule")]:
        r = arena_duel(BOTS[a_name], BOTS[b_name], episodes=200,
                       hero_class=args.hero, seed=99999)
        print(f"  {a_name:<7} vs {b_name:<7}  {r['win_rate']*100:5.1f}%")

    if args.save:
        save_agent(args.save, agent.net, meta=vars(args))
        print(f"\n权重已保存到 {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
