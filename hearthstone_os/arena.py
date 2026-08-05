"""对局评测：让两个对手打一批，先后手轮换，报胜率（路线图 M2）。

口径抄 `rosetta/arena.py`：同职业镜像 + 同构套牌、两 bot 各先手一半局数，
所以对角线（同 bot 互打）≈ 50% 就是没有优势。orange-stone 没有 start_player
参数（P1 永远先手），先后手轮换用"换座"实现：一半局让 bot1 坐 P1。
"""

from __future__ import annotations

from . import decks
from .bots import BOTS
from .env import Env

__all__ = ["duel", "matrix"]


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
