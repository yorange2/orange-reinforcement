"""观战 / 评测的命令行入口（路线图 M2，抄 `rosetta/play.py`）。

    .venv/bin/python -m hearthstone_os.play                      # 看两个机器人打一局
    .venv/bin/python -m hearthstone_os.play --bots greedy random
    .venv/bin/python -m hearthstone_os.play --bench 400          # 跑胜率
    .venv/bin/python -m hearthstone_os.play --matrix 200         # 胜率矩阵（对角线 ≈ 50%）
"""

from __future__ import annotations

import argparse
import time

from . import arena, decks
from .bots import BOTS
from .env import Env, describe_action


def _render(obs) -> str:
    def minion(entity) -> str:
        marks = "".join(
            mark
            for flag, mark in (
                (entity.taunt, "嘲"),
                (entity.divine_shield, "盾"),
                (entity.stealth, "潜"),
                (entity.windfury, "风"),
                (entity.charge, "冲"),
                (entity.frozen, "冻"),
            )
            if flag
        )
        idle = "" if entity.can_attack else "z"
        return f"{entity.name} {entity.attack}/{entity.health}{marks}{idle}"

    lines = [
        f"  对手   英雄 {obs.opponent.hero_health:3d}"
        f"  手牌 {obs.opponent.hand_count}  牌堆 {obs.opponent.deck_count}",
        "    场上 " + ("  ".join(minion(e) for e in obs.opponent.field) or "—"),
        "  " + "· " * 30,
        "    场上 " + ("  ".join(minion(e) for e in obs.me.field) or "—"),
        f"  自己   英雄 {obs.me.hero_health:3d}"
        f"  手牌 {obs.me.hand_count}  牌堆 {obs.me.deck_count}"
        f"  水晶 {obs.me.remaining_mana}/{obs.me.total_mana}",
    ]
    return "\n".join(lines)


def watch(bot1_name: str, bot2_name: str, seed: int) -> None:
    deck = decks.vanilla()
    env = Env(deck=deck, seed=seed)
    env.reset(seed=seed)

    seats = {1: BOTS[bot1_name](seed), 2: BOTS[bot2_name](seed + 1)}
    labels = {1: bot1_name, 2: bot2_name}

    turn = -1
    steps = 0
    while not env.done and steps < 5000:
        actions = env.legal_actions()
        if not actions:
            break

        obs = env.observe()
        if env.turn != turn:
            turn = env.turn
            print("\n" + "=" * 62)
            print(f"第 {turn} 个回合    玩家{env.current_player}"
                  f"（{labels[env.current_player]}） 行动")
            print("-" * 62)
            print(_render(obs))
            print("-" * 62)

        action = seats[env.current_player].choose(obs, actions)
        print(f"  玩家{env.current_player} → {describe_action(action, obs)}")
        env.step(action)
        steps += 1

    print("\n" + "=" * 62)
    if env.winner == 0:
        print(f"平局（{steps} 步，{env.turn} 个回合）")
    else:
        print(f"玩家{env.winner}（{labels[env.winner]}）获胜"
              f"（{steps} 步，{env.turn} 个回合）")


def bench(bot1_name: str, bot2_name: str, episodes: int, seed: int) -> None:
    start = time.time()
    result = arena.duel(
        BOTS[bot1_name],
        BOTS[bot2_name],
        episodes=episodes,
        seed=seed,
    )
    elapsed = time.time() - start

    print(f"{bot1_name} vs {bot2_name}    {episodes} 局，先后手轮换")
    print(f"  {bot1_name} 胜率  {result['win_rate']:.1%}")
    print(f"  平局      {result['draw_rate']:.1%}")
    print(f"  平均步数  {result['avg_steps']:.1f}")
    print(f"  耗时      {elapsed:.1f}s（{episodes / elapsed:.0f} 局/秒）")


def show_matrix(episodes: int, seed: int) -> None:
    print(f"胜率矩阵（行 bot 对列 bot，{episodes} 局/对，镜像卡组，对角线应 ≈ 50%）")
    names = sorted(BOTS)
    result = arena.matrix(names, episodes=episodes, seed=seed)
    header = "        " + "".join(f"{n:>8}" for n in names)
    print(header)
    for row in names:
        cells = "".join(f"{result[row][col]:>8.1%}" for col in names)
        print(f"{row:>8}{cells}")


def main() -> None:
    parser = argparse.ArgumentParser(description="orange-stone 炉石对战")
    parser.add_argument("--bots", nargs=2, default=["greedy", "random"],
                        choices=sorted(BOTS), metavar=("BOT1", "BOT2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench", type=int, metavar="N",
                        help="不观战，直接跑 N 局报胜率")
    parser.add_argument("--matrix", type=int, metavar="N",
                        help="跑 N 局/对的胜率矩阵（对角线校准）")
    args = parser.parse_args()

    if args.matrix:
        show_matrix(args.matrix, args.seed)
    elif args.bench:
        bench(args.bots[0], args.bots[1], args.bench, args.seed)
    else:
        watch(args.bots[0], args.bots[1], args.seed)


if __name__ == "__main__":
    main()
