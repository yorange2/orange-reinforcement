"""对局评测：让一个智能体坐一家，另外两家坐规则对手，统计胜率。

三家游戏里随机基准是 33.3%，而且拿 ♦3 的人有先手优势，所以评测时会轮转座位。
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .bots import make_bot
from .game import GameResult, play_game

HAND_SIZE = 16


@dataclass
class MatchStats:
    games: int
    wins: int
    avg_remaining: float   # 输的时候平均还剩几张，越小说明输得越接近
    avg_turns: float

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def __str__(self) -> str:
        return (
            f"{self.games} 局: 胜率 {self.win_rate * 100:5.1f}%  "
            f"负局平均剩牌 {self.avg_remaining:4.1f}  平均回合 {self.avg_turns:4.1f}"
        )


def final_reward(result: GameResult, seat: int, hand_size: int = HAND_SIZE) -> float:
    """终局回报：赢了 +1；输了按剩牌多少给 0 到 -1 的惩罚。

    剩得越多说明输得越惨，这个稠密一点的信号比纯 ±1 学得快不少。
    """
    if result.winner == seat:
        return 1.0
    return -result.remaining[seat] / hand_size


def evaluate(
    agent,
    opponent: str = "rule",
    games: int = 300,
    seed: int = 0,
    rotate: bool = True,
) -> MatchStats:
    """`agent` 坐一家，另外两家是 `opponent` 规则机器人。"""
    rng = random.Random(seed)
    wins = 0
    remaining: List[int] = []
    turns: List[int] = []

    for game in range(games):
        seat = game % 3 if rotate else 0
        players: List = [make_bot(opponent, seed=seed + game * 3 + i) for i in range(3)]
        players[seat] = agent
        result = play_game(players, rng=rng)

        if result.winner == seat:
            wins += 1
        else:
            remaining.append(result.remaining[seat])
        turns.append(result.turns)

    return MatchStats(
        games=games,
        wins=wins,
        avg_remaining=sum(remaining) / len(remaining) if remaining else 0.0,
        avg_turns=sum(turns) / len(turns),
    )


def evaluate_all(
    agent,
    opponents: Sequence[str] = ("random", "greedy", "rule"),
    games: int = 300,
    seed: int = 0,
) -> Dict[str, MatchStats]:
    """依次对三种规则对手评测。"""
    return {name: evaluate(agent, name, games=games, seed=seed) for name in opponents}


def match(
    players: Sequence[Tuple[str, object]],
    deals: int = 300,
    seed: int = 0,
) -> List[Tuple[str, MatchStats]]:
    """三个不同选手同桌混战，返回每个选手的战绩。

    每副牌都会按 6 种座位排列各打一遍，所以牌运和先手完全对消：谁赢下来
    就是真的强。总局数 = deals x 6，每人各自的基准胜率仍是 33.3%。
    """
    if len(players) != 3:
        raise ValueError(f"需要正好 3 个选手，收到 {len(players)} 个")

    wins = [0, 0, 0]
    remaining: List[List[int]] = [[], [], []]
    turns: List[int] = []
    games = 0

    for deal_index in range(deals):
        for seats in itertools.permutations(range(3)):
            # seats[i] 是第 i 号选手这一局坐的位置
            table: List[object] = [None, None, None]
            for i, seat in enumerate(seats):
                table[seat] = players[i][1]

            result = play_game(table, rng=random.Random(seed + deal_index))
            games += 1
            turns.append(result.turns)
            for i, seat in enumerate(seats):
                if result.winner == seat:
                    wins[i] += 1
                else:
                    remaining[i].append(result.remaining[seat])

    return [
        (
            players[i][0],
            MatchStats(
                games=games,
                wins=wins[i],
                avg_remaining=sum(remaining[i]) / len(remaining[i]) if remaining[i] else 0.0,
                avg_turns=sum(turns) / len(turns),
            ),
        )
        for i in range(3)
    ]


def bot_vs_bot(a: str, b: str, games: int = 300, seed: int = 0) -> MatchStats:
    """一家 a 对两家 b，用来校准规则对手之间的强弱。"""
    return evaluate(make_bot(a, seed=seed), b, games=games, seed=seed)


def format_table(stats: Dict[str, MatchStats]) -> str:
    lines = []
    for name, stat in stats.items():
        lines.append(f"  vs 2x{name:<7} {stat}")
    return "\n".join(lines)
