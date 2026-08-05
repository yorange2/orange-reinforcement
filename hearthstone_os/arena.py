"""对局评测：让两个对手打一批，先后手轮换，报胜率（路线图 M2/M3）。

口径抄 `rosetta/arena.py`：同职业镜像 + 同构套牌、两 bot 各先手一半局数，
所以对角线（同 bot 互打）≈ 50% 就是没有优势。orange-stone 没有 start_player
参数（P1 永远先手），先后手轮换用"换座"实现：一半局让 bot1 坐 P1。

M3 加了训练/评测用的 `play_game`（带 bind_env 的智能体驱动）和
`evaluate`/`evaluate_all`/`final_reward`（PPO 训练和战绩表的口径，
抄 `hearthstone/arena.py`）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from . import decks
from .bots import BOTS
from .env import Env

__all__ = ["duel", "matrix", "play_game", "GameResult", "final_reward",
           "evaluate", "evaluate_all", "MatchStats", "format_table"]


@dataclass
class GameResult:
    """一局的结局（hearthstone_os 口径：winner 1=P1 / 2=P2 / 0=平局）。"""

    winner: int
    hero_health: List[int]
    turns: int
    steps: int = 0
    rewards: List[float] = field(default_factory=list)  # 按 seat 索引 [0]=P1, [1]=P2


def play_game(
    players: Sequence,
    rng: Optional[random.Random] = None,
    *,
    seed: int = 0,
    deck: Optional[list[str]] = None,
    max_steps: int = 5000,
) -> GameResult:
    """让 `players`（实现了 choose(obs, actions) 的对象）打完一局。

    按 `rosetta/arena.py` 的换座逻辑驱动：`players[0]` 坐 P1（先手），
    `players[1]` 坐 P2。带 `bind_env` 的智能体（搜索/策略）会自动绑定。
    """
    if deck is None:
        deck = decks.vanilla()
    env = Env(deck=deck, seed=seed)
    env.reset(seed=seed)

    # 搜索型选手需要绑定真实局面才能克隆推演；只看观测的选手没有这个方法，跳过
    for seat, player in enumerate(players, start=1):
        bind = getattr(player, "bind_env", None)
        if bind is not None:
            bind(env, seat)

    steps = 0
    while not env.done and steps < max_steps:
        actions = env.legal_actions()
        if not actions:
            break
        obs = env.observe()
        env.step(players[env.current_player - 1].choose(obs, actions))
        steps += 1

    return GameResult(
        winner=env.winner,
        hero_health=env.hero_healths(),
        turns=env.turn,
        steps=steps,
        rewards=[env.last_reward(0), env.last_reward(1)],
    )


def final_reward(result: GameResult, seat: int) -> float:
    """终局回报：赢了 +1，平局 0，输了按对手剩多少血给 0 到 -1 的惩罚。

    与引擎 `terminal_reward="health_scaled"` 同口径（M1-G7）；训练时直接用
    `Env.last_reward(seat)` 读引擎奖励，这里给统计/对照用。
    """
    if result.winner == seat:
        return 1.0
    if result.winner == 0:
        return 0.0
    return -result.hero_health[result.winner - 1] / 30.0


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


def evaluate(
    agent,
    opponent: str = "rule",
    games: int = 400,
    seed: int = 0,
    rotate: bool = True,
) -> MatchStats:
    """`agent` 对 `opponent` 规则机器人，先后手轮换。"""
    wins = draws = 0
    health: List[int] = []
    turns: List[int] = []

    for game in range(games):
        seat = 1 + (game % 2) if rotate else 1
        players: List = [
            BOTS[opponent](seed=seed + game * 2 + i) for i in range(2)
        ]
        players[seat - 1] = agent
        result = play_game(players, seed=seed + game)

        if result.winner == seat:
            wins += 1
            health.append(result.hero_health[seat - 1])
        elif result.winner == 0:
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


def format_table(stats: Dict[str, MatchStats]) -> str:
    return "\n".join(f"  vs {name:<7} {stat}" for name, stat in stats.items())


def duel(
    bot1_cls,
    bot2_cls,
    episodes: int = 200,
    *,
    seed: int = 0,
    max_steps: int = 5000,
    deck: list[str] | None = None,
) -> dict[str, float]:
    """打 `episodes` 局，一半让 bot1 先手（坐 P1），一半让 bot2 先手。

    返回 bot1 的胜率、平局率和平均步数。同镜像卡组，所以 50% 就是没有优势。
    """
    if deck is None:
        deck = decks.vanilla()
    wins = draws = steps_total = 0

    for episode in range(episodes):
        bot1_first = episode % 2 == 0

        env = Env(deck=deck, seed=seed + episode)
        env.reset(seed=seed + episode)

        # 每局给机器人不同的种子。用固定种子重建的话，每一局的随机选择
        # 序列都一模一样，等于只在少数几条轨迹上反复采样。
        bot1_seed = seed + episode * 2
        bot2_seed = bot1_seed + 1

        # seat 1（P1，先手）/ seat 2（P2，后手）上分别坐着谁
        if bot1_first:
            seats = {1: bot1_cls(bot1_seed), 2: bot2_cls(bot2_seed)}
            bot1_seat = 1
        else:
            seats = {1: bot2_cls(bot2_seed), 2: bot1_cls(bot1_seed)}
            bot1_seat = 2

        steps = 0
        while not env.done and steps < max_steps:
            actions = env.legal_actions()
            if not actions:
                break
            obs = env.observe()
            env.step(seats[env.current_player].choose(obs, actions))
            steps += 1

        steps_total += steps
        if env.winner == 0:
            draws += 1
        elif env.winner == bot1_seat:
            wins += 1

    return {
        "win_rate": wins / episodes,
        "draw_rate": draws / episodes,
        "avg_steps": steps_total / episodes,
        "episodes": episodes,
    }


def matrix(
    bot_names: list[str] | None = None,
    episodes: int = 200,
    *,
    seed: int = 0,
    deck: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """胜率矩阵：行 bot 对列 bot 的胜率。

    同镜像卡组，对角线应 ≈ 50%（M2 验收口径：50% ± 2pp）。
    """
    names = bot_names or list(BOTS)
    return {
        row: {
            col: duel(BOTS[row], BOTS[col], episodes, seed=seed, deck=deck)["win_rate"]
            for col in names
        }
        for row in names
    }
