"""把"(局面, 某个候选动作)"编码成定长向量。

策略网络对每个候选动作单独打分，再在候选上做 softmax，所以动作空间是变长的也没关系：
每个动作用同一套特征描述，网络学的是"这种局面下这手牌好不好"。

前 26 维描述动作本身，后 14 维描述局面（同一决策点里对所有候选相同）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .combos import BOMB, KIND_INDEX, KINDS, Combo, estimate_turns
from .game import Action, Observation

#: 局面特征在向量里的起始下标，价值网络只吃这一段。
STATE_OFFSET = 28

FEATURE_NAMES: List[str] = (
    [f"kind_{kind}" for kind in KINDS]
    + [
        "is_pass", "is_bomb", "rank", "n_cards", "length", "hand_after",
        "wins_now", "turns_after", "turns_gain", "breaks_pair", "breaks_triple",
        "breaks_bomb", "n_attach", "higher_unseen", "is_top_rank",
        # 带牌本身的点数。少了这两维，"QQQ 带 6" 和 "QQQ 带 K" 在网络眼里
        # 一模一样（牌型/主牌/拆解代价全同），只能瞎猜一个。
        "attach_rank_max", "attach_rank_min",
    ]
    + [
        "is_leading", "hand_size", "min_opp_hand", "opp_le_2", "opp_le_1",
        "turns_now", "next_opp_hand", "prev_opp_hand", "required_rank",
        "required_len", "n_legal", "trick", "leader_is_me", "bias",
    ]
)
FEATURE_DIM = len(FEATURE_NAMES)


def state_features(obs: Observation, turns_now: Optional[int] = None) -> List[float]:
    """只跟局面有关的那 14 维。"""
    if turns_now is None:
        turns_now = estimate_turns(obs.hand)

    opponents = obs.opponents()
    opp_hands = [obs.hand_sizes[i] for i in opponents]
    min_opp = min(opp_hands)
    n = obs.n_players
    next_opp = obs.hand_sizes[(obs.player + 1) % n]
    prev_opp = obs.hand_sizes[(obs.player - 1) % n]
    required = obs.required

    return [
        1.0 if required is None else 0.0,
        len(obs.hand) / 16.0,
        min_opp / 16.0,
        1.0 if min_opp <= 2 else 0.0,
        1.0 if min_opp <= 1 else 0.0,
        turns_now / 10.0,
        next_opp / 16.0,
        prev_opp / 16.0,
        0.0 if required is None else (required.rank - 3) / 11.0,
        0.0 if required is None else len(required.cards) / 5.0,
        len(obs.legal) / 40.0,
        min(obs.trick, 60) / 60.0,
        1.0 if obs.leader == obs.player else 0.0,
        1.0,
    ]


def batch_features(obs: Observation) -> np.ndarray:
    """一次算出所有候选动作的特征，返回 (动作数, FEATURE_DIM) 的矩阵。"""
    turns_now = estimate_turns(obs.hand)
    tail = state_features(obs, turns_now)

    unseen = obs.unseen_counts()
    total_unseen = max(sum(unseen.values()), 1)
    counts = _hand_counts(obs.hand)
    hand_size = len(obs.hand)

    rows = np.empty((len(obs.legal), FEATURE_DIM), dtype=np.float32)
    for i, move in enumerate(obs.legal):
        rows[i, :STATE_OFFSET] = _move_features(
            move, obs, counts, hand_size, turns_now, unseen, total_unseen
        )
        rows[i, STATE_OFFSET:] = tail
    return rows


def _move_features(
    move: Action,
    obs: Observation,
    counts: dict,
    hand_size: int,
    turns_now: int,
    unseen: dict,
    total_unseen: int,
) -> List[float]:
    kind_onehot = [0.0] * len(KINDS)

    if move is None:  # 过牌
        return kind_onehot + [
            1.0, 0.0, 0.0, 0.0, 0.0,
            hand_size / 16.0, 0.0, turns_now / 10.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ]

    kind_onehot[KIND_INDEX[move.kind]] = 1.0
    played = len(move.cards)
    hand_after = hand_size - played
    turns_after = _turns_after(obs.hand, move)

    # 这一手把手里的对子/三张/炸弹拆散了几组
    breaks_pair = breaks_triple = breaks_bomb = 0
    move_counts = _rank_counts(move.cards)
    for rank, used in move_counts.items():
        have = counts[rank]
        if used < have:
            if have == 4:
                breaks_bomb += 1
            elif have == 3:
                breaks_triple += 1
            elif have == 2:
                breaks_pair += 1

    higher_unseen = sum(c for rank, c in unseen.items() if rank > move.rank) / total_unseen

    n_attach = played - _body_size(move)
    # 带出去的是哪几张牌：甩掉一张废牌和搭进去一张 K 差别很大。
    attach_ranks = attachment_ranks(move)
    attach_max = (max(attach_ranks) - 3) / 11.0 if attach_ranks else 0.0
    attach_min = (min(attach_ranks) - 3) / 11.0 if attach_ranks else 0.0

    return kind_onehot + [
        0.0,
        1.0 if move.kind == BOMB else 0.0,
        (move.rank - 3) / 11.0,
        played / 5.0,
        move.length / 6.0,
        hand_after / 16.0,
        1.0 if hand_after == 0 else 0.0,
        turns_after / 10.0,
        (turns_now - turns_after) / 3.0,
        float(breaks_pair),
        float(breaks_triple),
        float(breaks_bomb),
        n_attach / 4.0,
        higher_unseen,
        1.0 if move.rank == 14 else 0.0,
        attach_max,
        attach_min,
    ]


def attachment_ranks(move: Combo) -> List[int]:
    """带牌的点数（三带一/三带二/飞机带牌才有，其余返回空）。

    按点数判定而不是按 `move.cards` 的前后顺序：生成器产出的牌是"主体在前带牌在后"，
    但 `classify()` 会把牌重新排序，靠位置切会切错。
    """
    from .combos import (PLANE_ONE, PLANE_TWO, TRIPLE_ONE, TRIPLE_TWO)

    if move.kind in (TRIPLE_ONE, TRIPLE_TWO):
        body_ranks = {move.rank}
    elif move.kind in (PLANE_ONE, PLANE_TWO):
        body_ranks = set(range(move.rank - move.length + 1, move.rank + 1))
    else:
        return []

    return sorted({card.rank for card in move.cards if card.rank not in body_ranks})


def _body_size(move: Combo) -> int:
    """牌型主体有几张（带牌不算），用来推出带了几张。"""
    from .combos import (PAIR_STRAIGHT, PLANE, PLANE_ONE, PLANE_TWO, STRAIGHT,
                         TRIPLE, TRIPLE_ONE, TRIPLE_TWO)

    if move.kind in (PLANE, PLANE_ONE, PLANE_TWO):
        return 3 * move.length
    if move.kind in (TRIPLE, TRIPLE_ONE, TRIPLE_TWO):
        return 3
    if move.kind == PAIR_STRAIGHT:
        return 2 * move.length
    if move.kind == STRAIGHT:
        return move.length
    return len(move.cards)


def _turns_after(hand: Sequence, move: Combo) -> int:
    remaining = list(hand)
    for card in move.cards:
        remaining.remove(card)
    return estimate_turns(remaining)


def _hand_counts(hand: Sequence) -> dict:
    counts: dict = {}
    for card in hand:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def _rank_counts(cards: Sequence) -> dict:
    counts: dict = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts
