"""把 (局面, 候选动作) 编码成网络的输入。

提供三种编码，用来回答"手工特征到底卡住了多少上限"：

    handcrafted  42 维手工特征。牌力判断已经被 estimate_turns 之类的函数嚼碎喂进去了。
    raw          只给事实：12 个点数 x 6 个通道的网格 + 少量标量，牌好不好得网络自己学。
    both         两者拼一起——不扔掉好特征，同时把原始信号也给它。

三种编码的输出都是一个定长向量，所以下游（补齐成批、掩码、PPO）完全不用改。
布局统一是 [动作相关的块][局面相关的块]，局面块在末尾且长度固定，价值网络切最后
`state_dim` 维就行。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .cards import MAX_RANK, MIN_RANK, RANKS
from .combos import KIND_INDEX, KINDS
from .features import FEATURE_DIM as HANDCRAFTED_DIM
from .features import STATE_OFFSET as HANDCRAFTED_STATE_OFFSET
from .features import batch_features as handcrafted_features
from .features import state_features
from .game import Observation

N_RANKS = len(RANKS)          # 12：3 到 A
GRID_CHANNELS = 6
GRID_SIZE = N_RANKS * GRID_CHANNELS

#: 网格的六个通道，全都是"事实"，没有任何启发式判断。
GRID_CHANNEL_NAMES = (
    "hand",        # 我手上这个点数有几张
    "move",        # 这一手打出去的牌里这个点数有几张
    "hand_after",  # 出完之后我这个点数还剩几张
    "unseen",      # 对手手里可能还有几张
    "played",      # 已经打出去过几张（公开信息）
    "required",    # 当前要压的那手牌，主牌点数在这里标 1
)

RAW_MOVE_SCALARS = (
    [f"kind_{kind}" for kind in KINDS]
    + ["is_pass", "move_rank", "move_cards", "move_length"]
)
RAW_STATE_SCALARS = (
    "is_leading", "hand_size", "next_opp_hand", "prev_opp_hand",
    "required_len", "n_legal", "trick", "leader_is_me", "bias",
)


class Encoder:
    """编码器接口。`dim` 是向量长度，末尾 `state_dim` 维是局面块。"""

    name = "base"
    dim = 0
    state_dim = 0
    #: 网格在向量里的位置 (起点, 终点)，没有网格的编码是 None。
    grid_slice: Optional[Tuple[int, int]] = None

    @property
    def state_offset(self) -> int:
        return self.dim - self.state_dim

    def build(self, obs: Observation) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError


class HandcraftedEncoder(Encoder):
    """现有的 42 维手工特征。"""

    name = "handcrafted"
    dim = HANDCRAFTED_DIM
    state_dim = HANDCRAFTED_DIM - HANDCRAFTED_STATE_OFFSET

    def build(self, obs: Observation) -> np.ndarray:
        return handcrafted_features(obs)


class RawEncoder(Encoder):
    """只给事实的原始编码：点数网格 + 少量标量。

    网格是 (6 通道, 12 个点数)，展平后放在向量最前面。之所以按点数排成一条轴，是因为
    顺子和连对本质上就是这条轴上的连续片段——留给卷积去发现，而不是我们替它算好。
    """

    name = "raw"
    dim = GRID_SIZE + len(RAW_MOVE_SCALARS) + len(RAW_STATE_SCALARS)
    state_dim = len(RAW_STATE_SCALARS)
    grid_slice = (0, GRID_SIZE)

    def build(self, obs: Observation) -> np.ndarray:
        rows = np.zeros((len(obs.legal), self.dim), dtype=np.float32)

        hand_counts = _counts(card.rank for card in obs.hand)
        unseen = obs.unseen_counts()
        played = obs.played_counts
        required = obs.required

        # 所有候选共用的部分先算一次
        base_grid = np.zeros((GRID_CHANNELS, N_RANKS), dtype=np.float32)
        for i, rank in enumerate(RANKS):
            base_grid[0, i] = hand_counts.get(rank, 0) / 4.0
            base_grid[3, i] = max(unseen.get(rank, 0), 0) / 4.0
            base_grid[4, i] = played.get(rank, 0) / 4.0
        if required is not None:
            base_grid[5, _index(required.rank)] = 1.0

        tail = _raw_state_scalars(obs)

        for row, move in enumerate(obs.legal):
            grid = base_grid.copy()
            scalars = [0.0] * len(RAW_MOVE_SCALARS)

            if move is None:
                grid[2] = grid[0]                      # 过牌，手牌不变
                scalars[len(KINDS)] = 1.0              # is_pass
            else:
                move_counts = _counts(card.rank for card in move.cards)
                for rank, count in move_counts.items():
                    idx = _index(rank)
                    grid[1, idx] = count / 4.0
                    grid[2, idx] = (hand_counts.get(rank, 0) - count) / 4.0
                for rank in RANKS:
                    idx = _index(rank)
                    if rank not in move_counts:
                        grid[2, idx] = grid[0, idx]

                scalars[KIND_INDEX[move.kind]] = 1.0
                scalars[len(KINDS) + 1] = (move.rank - MIN_RANK) / (MAX_RANK - MIN_RANK)
                scalars[len(KINDS) + 2] = len(move.cards) / 5.0
                scalars[len(KINDS) + 3] = move.length / 6.0

            rows[row, : GRID_SIZE] = grid.reshape(-1)
            rows[row, GRID_SIZE : GRID_SIZE + len(RAW_MOVE_SCALARS)] = scalars
            rows[row, self.state_offset :] = tail

        return rows


class BothEncoder(Encoder):
    """手工特征 + 原始网格。不扔掉已经好用的东西，同时把原始信号也给网络。

    局面块用手工那份（它是两者里更全的），原始编码那 9 个局面标量就不重复放了。
    """

    name = "both"
    dim = HANDCRAFTED_STATE_OFFSET + GRID_SIZE + len(RAW_MOVE_SCALARS) + (
        HANDCRAFTED_DIM - HANDCRAFTED_STATE_OFFSET
    )
    state_dim = HANDCRAFTED_DIM - HANDCRAFTED_STATE_OFFSET
    grid_slice = (HANDCRAFTED_STATE_OFFSET, HANDCRAFTED_STATE_OFFSET + GRID_SIZE)

    def __init__(self) -> None:
        self._raw = RawEncoder()

    def build(self, obs: Observation) -> np.ndarray:
        handcrafted = handcrafted_features(obs)
        raw = self._raw.build(obs)

        rows = np.empty((len(obs.legal), self.dim), dtype=np.float32)
        cut = HANDCRAFTED_STATE_OFFSET
        raw_move_end = GRID_SIZE + len(RAW_MOVE_SCALARS)

        rows[:, :cut] = handcrafted[:, :cut]                       # 手工的动作块
        rows[:, cut : cut + raw_move_end] = raw[:, :raw_move_end]  # 网格 + 原始标量
        rows[:, cut + raw_move_end :] = handcrafted[:, cut:]       # 手工的局面块
        return rows


ENCODERS = {
    "handcrafted": HandcraftedEncoder,
    "raw": RawEncoder,
    "both": BothEncoder,
}


def make_encoder(name: str) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"未知的编码方式 {name!r}，可选: {', '.join(ENCODERS)}")
    return ENCODERS[name]()


# ------------------------------------------------------------------ 小工具


def _index(rank: int) -> int:
    return rank - MIN_RANK


def _counts(ranks) -> dict:
    counts: dict = {}
    for rank in ranks:
        counts[rank] = counts.get(rank, 0) + 1
    return counts


def _raw_state_scalars(obs: Observation) -> List[float]:
    """原始编码的局面标量。刻意不含任何启发式量（比如手牌还要几轮走完）。"""
    full = state_features(obs)
    names = [
        "is_leading", "hand_size", "next_opp_hand", "prev_opp_hand",
        "required_len", "n_legal", "trick", "leader_is_me", "bias",
    ]
    from .features import FEATURE_NAMES

    lookup = {name: full[i] for i, name in enumerate(FEATURE_NAMES[HANDCRAFTED_STATE_OFFSET:])}
    return [lookup[name] for name in names]
