"""人为构造的规则对手，也是衡量模型强弱的标尺。

三个难度递增的对手：

    RandomBot  合法动作里随机挑一个
    GreedyBot  跟牌出"最小能压的"，领出出最小的牌，从不留后手
    RuleBot    带手牌拆解估计的启发式：便宜地压、保留炸弹、残局压制、能走就走
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from .cards import Card
from .combos import BOMB, PAIR, SINGLE, Combo, estimate_turns
from .game import Action, Observation


class Bot:
    """对手接口：给一个观测，返回一个合法动作。"""

    name = "bot"

    def choose(self, obs: Observation) -> Action:  # pragma: no cover - 接口
        raise NotImplementedError

    @staticmethod
    def _plays(obs: Observation) -> List[Combo]:
        """合法动作里真正出牌的部分（去掉"过"）。"""
        return [move for move in obs.legal if move is not None]


class RandomBot(Bot):
    """完全随机，胜率基准线。"""

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, obs: Observation) -> Action:
        return self.rng.choice(obs.legal)


class GreedyBot(Bot):
    """能压就压，且总是出最便宜的牌。

    典型的"贪快"打法：不留大牌、不管对手还剩几张。
    """

    name = "greedy"

    def choose(self, obs: Observation) -> Action:
        plays = self._plays(obs)
        if not plays:
            return None
        return min(plays, key=lambda move: (move.kind == BOMB, move.rank, len(move.cards)))


class RuleBot(Bot):
    """启发式对手：会算手牌拆解、会留炸弹、会在残局压人。"""

    name = "rule"

    #: 手牌还多的时候，不用大牌去压对手的小牌
    BIG_RANK = 13
    SMALL_RANK = 8

    def choose(self, obs: Observation) -> Action:
        plays = self._plays(obs)
        if not plays:
            return None

        # 能一把走完就直接赢
        finishing = [m for m in plays if len(m.cards) == len(obs.hand)]
        if finishing:
            return max(finishing, key=lambda m: m.rank)

        danger = min(obs.hand_sizes[i] for i in obs.opponents()) <= 2
        turns_now = estimate_turns(obs.hand)

        if obs.required is None:
            return self._lead(obs, plays, danger)
        return self._follow(obs, plays, danger, turns_now)

    # ------------------------------------------------------------------ 领出

    def _lead(self, obs: Observation, plays: List[Combo], danger: bool) -> Action:
        non_bomb = [m for m in plays if m.kind != BOMB] or plays

        if danger:
            # 对手快走完了：出大牌，尽量让他接不上
            return max(non_bomb, key=lambda m: (m.rank, len(m.cards)))

        # 平时：选出完之后手牌最好拆的那一手，同分时多出几张、点数小的优先
        return min(non_bomb, key=lambda m: (self._turns_after(obs.hand, m), -len(m.cards), m.rank))

    # ------------------------------------------------------------------ 跟牌

    def _follow(self, obs: Observation, plays: List[Combo], danger: bool, turns_now: int) -> Action:
        non_bomb = [m for m in plays if m.kind != BOMB]

        if danger:
            # 残局：能压就用最大的压住，必要时炸
            pool = non_bomb or plays
            return max(pool, key=lambda m: (m.rank, len(m.cards)))

        if not non_bomb:
            return None  # 只能靠炸弹压，对手又不危险，不划算

        best = min(non_bomb, key=lambda m: (self._turns_after(obs.hand, m), m.rank))

        # 拆散了手牌就别压了
        if self._turns_after(obs.hand, best) > turns_now:
            return None

        # 手牌还多的时候，不拿 K/A 去压人家的小牌
        required = obs.required
        if (
            required is not None
            and best.kind in (SINGLE, PAIR)
            and best.rank >= self.BIG_RANK
            and required.rank <= self.SMALL_RANK
            and len(obs.hand) > 5
        ):
            return None

        return best

    @staticmethod
    def _turns_after(hand: Sequence[Card], move: Combo) -> int:
        remaining = list(hand)
        for card in move.cards:
            remaining.remove(card)
        return estimate_turns(remaining)


BOTS = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "rule": RuleBot,
}


def make_bot(name: str, seed: Optional[int] = None) -> Bot:
    """按名字构造对手。"""
    if name not in BOTS:
        raise ValueError(f"未知对手 {name!r}，可选: {', '.join(BOTS)}")
    bot = BOTS[name]
    return bot(seed=seed) if name == "random" else bot()
