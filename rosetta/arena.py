"""对局评测：让两个对手打一批，先后手轮换，报胜率。"""

from __future__ import annotations

from . import decks
from .env import Env

__all__ = ["duel"]


def duel(
    bot1_cls,
    bot2_cls,
    episodes: int = 200,
    *,
    hero_class: str = "MAGE",
    seed: int = 0,
    max_steps: int = 5000,
) -> dict[str, float]:
    """打 `episodes` 局，一半让 bot1 先手，一半让 bot2 先手。

    返回 bot1 的胜率、平局率和平均步数。同职业镜像 + 同构套牌，
    所以 50% 就是没有优势。
    """
    deck = decks.vanilla()
    wins = draws = steps_total = 0

    for episode in range(episodes):
        bot1_first = episode % 2 == 0

        env = Env(
            player1_class=hero_class,
            player2_class=hero_class,
            player1_deck=deck,
            player2_deck=deck,
        )
        env.reset(seed=seed + episode)

        # 每局给机器人不同的种子。用固定种子重建的话，每一局的随机选择
        # 序列都一模一样，等于只在少数几条轨迹上反复采样。
        bot1_seed = seed + episode * 2
        bot2_seed = bot1_seed + 1

        # seat 1 / seat 2 上分别坐着谁
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
