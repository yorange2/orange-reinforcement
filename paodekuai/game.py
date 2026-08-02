"""跑得快引擎：发牌、轮转、过牌、判定胜负。

一局的流程：持 ♦3 的人先出且首手必须包含 ♦3；之后按顺序跟牌，跟不起或不想跟
就"过"；其余人全部过之后，最后出牌的人重新自由出牌；谁先出完手牌谁赢。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .cards import Card, DIAMOND_THREE, HAND_SIZE, N_PLAYERS, deal
from .combos import Combo, beats, legal_moves

#: 过牌用 None 表示。
Action = Optional[Combo]


@dataclass
class Observation:
    """一个玩家在某个决策点能看到的全部信息。"""

    player: int                     # 自己是几号位
    hand: List[Card]                # 自己的手牌
    hand_sizes: List[int]           # 每家剩几张（公开信息）
    required: Optional[Combo]       # 当前要压的牌，None 表示自由出牌
    leader: Optional[int]           # 当前牌面是谁打出的
    played_counts: Dict[int, int]   # 各点数已经打出去多少张（公开信息）
    legal: List[Action]             # 合法动作，含"过"（None）
    trick: int                      # 第几个回合

    @property
    def n_players(self) -> int:
        return len(self.hand_sizes)

    def opponents(self) -> List[int]:
        return [i for i in range(self.n_players) if i != self.player]

    def unseen_counts(self) -> Dict[int, int]:
        """对手手里还可能有哪些牌：全牌堆 - 已出的 - 自己手上的。"""
        from .cards import RANKS

        remaining = {rank: 4 for rank in RANKS}
        for rank, count in self.played_counts.items():
            remaining[rank] -= count
        for card in self.hand:
            remaining[card.rank] -= 1
        return remaining


@dataclass
class GameResult:
    winner: int
    remaining: List[int]                    # 每家结束时剩几张
    turns: int
    log: List[str] = field(default_factory=list)


class Game:
    """一局跑得快的状态机。"""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        n_players: int = N_PLAYERS,
        hand_size: int = HAND_SIZE,
        max_turns: int = 400,
    ) -> None:
        self.rng = rng or random.Random()
        self.n_players = n_players
        self.hand_size = hand_size
        self.max_turns = max_turns
        self.reset()

    # ------------------------------------------------------------------ 状态

    def reset(self) -> Observation:
        """发牌开局，返回先手玩家的观测。"""
        self.hands: List[List[Card]] = deal(self.rng, self.n_players, self.hand_size)
        self.required: Optional[Combo] = None
        self.leader: Optional[int] = None
        self.passes = 0
        self.turns = 0
        self.finished = False
        self.winner: Optional[int] = None
        self.played_counts: Dict[int, int] = {}
        self.first_move = True

        # 谁拿到 ♦3 谁先出
        self.current = next(i for i, hand in enumerate(self.hands) if DIAMOND_THREE in hand)
        return self.observe()

    def observe(self, player: Optional[int] = None) -> Observation:
        player = self.current if player is None else player
        return Observation(
            player=player,
            hand=list(self.hands[player]),
            hand_sizes=[len(hand) for hand in self.hands],
            required=self.required,
            leader=self.leader,
            played_counts=dict(self.played_counts),
            legal=self.legal_actions(player),
            trick=self.turns,
        )

    def legal_actions(self, player: Optional[int] = None) -> List[Action]:
        """当前玩家的合法动作。自由出牌时不能过；首手必须包含 ♦3。"""
        player = self.current if player is None else player
        hand = self.hands[player]
        moves: List[Action] = list(legal_moves(hand, self.required))

        if self.first_move:
            moves = [m for m in moves if m is not None and DIAMOND_THREE in m.cards]

        if self.required is not None:
            moves.append(None)  # 跟不起或不想跟就过
        return moves

    # ------------------------------------------------------------------ 推进

    def step(self, action: Action) -> None:
        """执行一个动作，轮到下一家。"""
        if self.finished:
            raise RuntimeError("这一局已经结束了")

        if action is None:
            if self.required is None:
                raise ValueError("自由出牌时不能过牌")
            self.passes += 1
        else:
            self._validate(action)
            hand = self.hands[self.current]
            for card in action.cards:
                hand.remove(card)
                self.played_counts[card.rank] = self.played_counts.get(card.rank, 0) + 1
            self.required = action
            self.leader = self.current
            self.passes = 0
            self.first_move = False

            if not hand:
                self.finished = True
                self.winner = self.current
                return

        self.turns += 1
        if self.turns >= self.max_turns:  # 兜底，正常不会触发
            self.finished = True
            self.winner = min(range(self.n_players), key=lambda i: len(self.hands[i]))
            return

        # 其他人都过了，最后出牌的人重新自由出牌
        if self.passes >= self.n_players - 1:
            self.required = None
            self.passes = 0
            self.current = self.leader if self.leader is not None else self.current
        else:
            self.current = (self.current + 1) % self.n_players

    def _validate(self, action: Combo) -> None:
        hand = self.hands[self.current]
        counts: Dict[Card, int] = {}
        for card in action.cards:
            counts[card] = counts.get(card, 0) + 1
        for card, need in counts.items():
            if hand.count(card) < need:
                raise ValueError(f"手里没有 {card}")
        if not beats(action, self.required):
            raise ValueError(f"{action} 压不过 {self.required}")
        if self.first_move and DIAMOND_THREE not in action.cards:
            raise ValueError("首手必须包含 ♦3")

    def result(self) -> GameResult:
        if not self.finished or self.winner is None:
            raise RuntimeError("这一局还没结束")
        return GameResult(
            winner=self.winner,
            remaining=[len(hand) for hand in self.hands],
            turns=self.turns,
        )


def play_game(players: Sequence, rng: Optional[random.Random] = None, verbose: bool = False) -> GameResult:
    """让 `players`（实现了 choose(obs) 的对象）打完一局。"""
    game = Game(rng=rng, n_players=len(players))
    log: List[str] = []

    if verbose:
        from .cards import hand_to_str

        for i, hand in enumerate(game.hands):
            log.append(f"玩家{i} 起手: {hand_to_str(hand)}")

    while not game.finished:
        obs = game.observe()
        action = players[obs.player].choose(obs)
        if action not in obs.legal:
            raise ValueError(f"玩家{obs.player} 给出了非法动作 {action}")
        if verbose:
            text = "过" if action is None else str(action)
            log.append(f"玩家{obs.player} ({len(obs.hand)}张) -> {text}")
        game.step(action)

    result = game.result()
    result.log = log
    if verbose:
        log.append(f"玩家{result.winner} 获胜，剩余张数 {result.remaining}")
    return result
