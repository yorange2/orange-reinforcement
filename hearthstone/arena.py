"""对局评测：两个选手对打，统计胜率。

两家游戏的随机基准是 50%，而且先手有优势，所以评测时先后手对半轮换——
`evaluate` 里同一个种子的一副牌会正反各打一遍，牌运和先手完全对消。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .bots import make_bot
from .game import HERO_HEALTH, GameResult, play_game


@dataclass
class MatchStats:
    games: int
    wins: int
    draws: int
    avg_health: float       # 赢的时候自己平均还剩多少血，越高说明赢得越轻松
    avg_turns: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def draw_rate(self) -> float:
        return self.draws / self.games if self.games else 0.0

    def __str__(self) -> str:
        return (
            f"{self.games} 局: 胜率 {self.win_rate * 100:5.1f}%  "
            f"平局 {self.draw_rate * 100:4.1f}%  "
            f"胜局平均剩血 {self.avg_health:4.1f}  平均回合 {self.avg_turns:4.1f}"
        )


def final_reward(result: GameResult, seat: int, hero_health: int = HERO_HEALTH) -> float:
    """终局回报：赢了 +1，平局 0，输了按对手剩多少血给 0 到 -1 的惩罚。

    输的时候把对面打剩 2 血和一点没碰到显然不是一回事，这个稠密一点的信号
    比纯 ±1 学得快——和跑得快那边按剩牌给分是同一个思路。
    """
    if result.winner == seat:
        return 1.0
    if result.winner is None:
        return 0.0
    return -result.hero_health[1 - seat] / hero_health


def evaluate(
    agent,
    opponent: str = "rule",
    games: int = 400,
    seed: int = 0,
    rotate: bool = True,
) -> MatchStats:
    """`agent` 对 `opponent` 规则机器人，先后手轮换。"""
    rng = random.Random(seed)
    wins = draws = 0
    health: List[int] = []
    turns: List[int] = []

    for game in range(games):
        seat = game % 2 if rotate else 0
        players: List = [make_bot(opponent, seed=seed + game * 2 + i) for i in range(2)]
        players[seat] = agent
        result = play_game(players, rng=rng, first=0)

        if result.winner == seat:
            wins += 1
            health.append(result.hero_health[seat])
        elif result.winner is None:
            draws += 1
        turns.append(result.turns)

    return MatchStats(
        games=games,
        wins=wins,
        draws=draws,
        avg_health=sum(health) / len(health) if health else 0.0,
        avg_turns=sum(turns) / len(turns),
    )


def evaluate_all(
    agent,
    opponents: Sequence[str] = ("random", "greedy", "rule"),
    games: int = 400,
    seed: int = 0,
) -> Dict[str, MatchStats]:
    """依次对三种规则对手评测。"""
    return {name: evaluate(agent, name, games=games, seed=seed) for name in opponents}


def duel(a, b, deals: int = 200, seed: int = 0) -> MatchStats:
    """两个选手直接对打，返回 a 的战绩。

    每副牌打两遍——a 先手一遍、b 先手一遍——所以先手优势和洗牌运气都对消掉了，
    a 的胜率高于 50% 就是真的强。总局数 = deals x 2。
    """
    wins = draws = 0
    health: List[int] = []
    turns: List[int] = []
    games = 0

    for index in range(deals):
        for seat in (0, 1):
            table = [b, b]
            table[seat] = a
            result = play_game(table, rng=random.Random(seed + index), first=0)
            games += 1
            turns.append(result.turns)
            if result.winner == seat:
                wins += 1
                health.append(result.hero_health[seat])
            elif result.winner is None:
                draws += 1

    return MatchStats(
        games=games,
        wins=wins,
        draws=draws,
        avg_health=sum(health) / len(health) if health else 0.0,
        avg_turns=sum(turns) / len(turns),
    )


def format_table(stats: Dict[str, MatchStats]) -> str:
    return "\n".join(f"  vs {name:<7} {stat}" for name, stat in stats.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="规则对手之间的胜率矩阵")
    parser.add_argument("--games", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    names = list(("random", "greedy", "rule"))
    print(f"每格 {args.games} 局，先后手轮换，随机基准 50%\n")
    print("           " + "".join(f"vs {n:<9}" for n in names))
    for a in names:
        row = [f"{a:<9}  "]
        for b in names:
            stat = evaluate(make_bot(a, seed=args.seed), b, games=args.games, seed=args.seed)
            row.append(f"{stat.win_rate * 100:5.1f}%     ")
        print("".join(row))


if __name__ == "__main__":
    main()
