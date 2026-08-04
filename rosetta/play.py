"""观战 / 评测的命令行入口。

    .venv/bin/python -m rosetta.play                      # 看两个机器人打一局
    .venv/bin/python -m rosetta.play --bots greedy random
    .venv/bin/python -m rosetta.play --bench 400          # 跑胜率
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
                (entity.poisonous, "毒"),
                (entity.windfury, "风"),
                (entity.lifesteal, "吸"),
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


def watch(bot1_name: str, bot2_name: str, seed: int, hero_class: str) -> None:
    deck = decks.vanilla()
    env = Env(
        player1_class=hero_class,
        player2_class=hero_class,
        player1_deck=deck,
        player2_deck=deck,
    )
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


def bench(bot1_name: str, bot2_name: str, episodes: int, seed: int,
          hero_class: str) -> None:
    start = time.time()
    result = arena.duel(
        BOTS[bot1_name],
        BOTS[bot2_name],
        episodes=episodes,
        hero_class=hero_class,
        seed=seed,
    )
    elapsed = time.time() - start

    print(f"{bot1_name} vs {bot2_name}    {episodes} 局，先后手轮换")
    print(f"  {bot1_name} 胜率  {result['win_rate']:.1%}")
    print(f"  平局      {result['draw_rate']:.1%}")
    print(f"  平均步数  {result['avg_steps']:.1f}")
    print(f"  耗时      {elapsed:.1f}s（{episodes / elapsed:.0f} 局/秒）")


def main() -> None:
    parser = argparse.ArgumentParser(description="RosettaStone 炉石对战")
    parser.add_argument("--bots", nargs=2, default=["greedy", "random"],
                        choices=sorted(BOTS), metavar=("BOT1", "BOT2"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hero", default="MAGE", help="双方英雄职业")
    parser.add_argument("--bench", type=int, metavar="N",
                        help="不观战，直接跑 N 局报胜率")
    args = parser.parse_args()

    if args.bench:
        bench(args.bots[0], args.bots[1], args.bench, args.seed, args.hero)
    else:
        watch(args.bots[0], args.bots[1], args.seed, args.hero)


if __name__ == "__main__":
    main()
