"""牌型：识别、比大小、以及枚举合法出牌。

支持的牌型（length 表示"连长"：顺子记牌数，连对记对数，飞机记三张数，其余为 1）：

    single        单张
    pair          对子
    triple        三张
    triple_one    三带一
    triple_two    三带二（带一对）
    straight      顺子，>= 5 张连续单牌
    pair_straight 连对，>= 2 组连续对子
    plane         飞机，>= 2 组连续三张
    plane_one     飞机带单（每组三张带 1 张）
    plane_two     飞机带对（每组三张带 1 对）
    bomb          炸弹，四张同点，压任何非炸弹

比较规则：同牌型且同连长时比主牌点数；炸弹压一切非炸弹；炸弹之间比点数。
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .cards import Card

SINGLE = "single"
PAIR = "pair"
TRIPLE = "triple"
TRIPLE_ONE = "triple_one"
TRIPLE_TWO = "triple_two"
STRAIGHT = "straight"
PAIR_STRAIGHT = "pair_straight"
PLANE = "plane"
PLANE_ONE = "plane_one"
PLANE_TWO = "plane_two"
BOMB = "bomb"

KINDS: Tuple[str, ...] = (
    SINGLE, PAIR, TRIPLE, TRIPLE_ONE, TRIPLE_TWO,
    STRAIGHT, PAIR_STRAIGHT, PLANE, PLANE_ONE, PLANE_TWO, BOMB,
)
KIND_INDEX: Dict[str, int] = {kind: i for i, kind in enumerate(KINDS)}

KIND_NAMES_CN: Dict[str, str] = {
    SINGLE: "单张", PAIR: "对子", TRIPLE: "三张", TRIPLE_ONE: "三带一",
    TRIPLE_TWO: "三带二", STRAIGHT: "顺子", PAIR_STRAIGHT: "连对",
    PLANE: "飞机", PLANE_ONE: "飞机带单", PLANE_TWO: "飞机带对", BOMB: "炸弹",
}

MIN_STRAIGHT = 5       # 顺子最少 5 张
MIN_PAIR_STRAIGHT = 2  # 连对最少 2 组
MIN_PLANE = 2          # 飞机最少 2 组三张

#: 飞机带牌时，最多为每个飞机主体生成多少种带牌方案（控制动作空间大小）。
MAX_ATTACH_SETS = 6


@dataclass(frozen=True)
class Combo:
    """一手打出去的牌。`rank` 是用于比大小的主牌点数。"""

    kind: str
    rank: int
    length: int
    cards: Tuple[Card, ...]

    def __len__(self) -> int:
        return len(self.cards)

    def __str__(self) -> str:
        from .cards import hand_to_str

        return f"{KIND_NAMES_CN[self.kind]}[{hand_to_str(self.cards)}]"


def beats(candidate: Combo, current: Optional[Combo]) -> bool:
    """`candidate` 能否压过 `current`（`current` 为 None 表示自由出牌）。"""
    if current is None:
        return True
    if candidate.kind == BOMB and current.kind != BOMB:
        return True
    if current.kind == BOMB and candidate.kind != BOMB:
        return False
    if candidate.kind != current.kind or candidate.length != current.length:
        return False
    return candidate.rank > current.rank


# --------------------------------------------------------------------- 工具


def group_by_rank(cards: Iterable[Card]) -> Dict[int, List[Card]]:
    """按点数分组，组内按花色升序（所以 3 的第一张一定是 ♦3）。"""
    groups: Dict[int, List[Card]] = {}
    for card in cards:
        groups.setdefault(card.rank, []).append(card)
    for cards_of_rank in groups.values():
        cards_of_rank.sort()
    return groups


def _runs(ranks: Sequence[int], min_length: int) -> List[List[int]]:
    """从升序点数里找出所有长度 >= min_length 的连续片段（含所有子片段）。"""
    ordered = sorted(set(ranks))
    result: List[List[int]] = []
    for start in range(len(ordered)):
        run = [ordered[start]]
        if len(run) >= min_length:
            result.append(list(run))
        for nxt in ordered[start + 1 :]:
            if nxt != run[-1] + 1:
                break
            run.append(nxt)
            if len(run) >= min_length:
                result.append(list(run))
    return result


# ----------------------------------------------------------------- 牌型识别


def classify(cards: Sequence[Card]) -> Optional[Combo]:
    """判断这几张牌构成什么牌型，不是合法牌型则返回 None。"""
    if not cards:
        return None

    cards = tuple(sorted(cards))
    if len(set(cards)) != len(cards):
        return None  # 同一张牌不能出两次

    groups = group_by_rank(cards)
    counts = {rank: len(group) for rank, group in groups.items()}
    n = len(cards)

    # 单一点数的牌型
    if len(counts) == 1:
        rank = next(iter(counts))
        if n == 1:
            return Combo(SINGLE, rank, 1, cards)
        if n == 2:
            return Combo(PAIR, rank, 1, cards)
        if n == 3:
            return Combo(TRIPLE, rank, 1, cards)
        if n == 4:
            return Combo(BOMB, rank, 1, cards)
        return None

    # 顺子
    if n >= MIN_STRAIGHT and all(c == 1 for c in counts.values()) and _is_consecutive(counts):
        return Combo(STRAIGHT, max(counts), n, cards)

    # 连对
    if n >= 2 * MIN_PAIR_STRAIGHT and n % 2 == 0 and all(c == 2 for c in counts.values()) and _is_consecutive(counts):
        return Combo(PAIR_STRAIGHT, max(counts), n // 2, cards)

    # 三张打头的牌型：三带一 / 三带二 / 飞机（可带牌）
    triple_ranks = sorted(rank for rank, count in counts.items() if count >= 3)
    for run in sorted(_runs(triple_ranks, 1), key=len, reverse=True):
        k = len(run)
        body = [card for rank in run for card in groups[rank][:3]]
        rest = _remove_cards(cards, body)
        combo = _match_with_attachments(cards, run, k, rest, n)
        if combo is not None:
            return combo
    return None


def _is_consecutive(counts: Dict[int, int]) -> bool:
    ranks = sorted(counts)
    return ranks == list(range(ranks[0], ranks[0] + len(ranks)))


def _match_with_attachments(
    cards: Tuple[Card, ...], run: List[int], k: int, rest: List[Card], n: int
) -> Optional[Combo]:
    """给定 k 组连续三张，看剩下的牌能否构成合法带牌。"""
    top = max(run)
    if n == 3 * k:
        if k == 1:
            return Combo(TRIPLE, top, 1, cards)
        if k >= MIN_PLANE:
            return Combo(PLANE, top, k, cards)
        return None

    if n == 4 * k and len(rest) == k:
        # 三带一 / 飞机带单：带的牌必须是 k 个不同点数
        if len({card.rank for card in rest}) == k:
            if k == 1:
                return Combo(TRIPLE_ONE, top, 1, cards)
            if k >= MIN_PLANE:
                return Combo(PLANE_ONE, top, k, cards)
        return None

    if n == 5 * k and len(rest) == 2 * k:
        # 三带二 / 飞机带对：带的牌必须是 k 个对子
        rest_counts = {rank: len(group) for rank, group in group_by_rank(rest).items()}
        if len(rest_counts) == k and all(count == 2 for count in rest_counts.values()):
            if k == 1:
                return Combo(TRIPLE_TWO, top, 1, cards)
            if k >= MIN_PLANE:
                return Combo(PLANE_TWO, top, k, cards)
        return None

    return None


def _remove_cards(cards: Sequence[Card], to_remove: Sequence[Card]) -> List[Card]:
    remaining = list(cards)
    for card in to_remove:
        remaining.remove(card)
    return remaining


# --------------------------------------------------------------- 出牌枚举


def all_combos(hand: Sequence[Card]) -> List[Combo]:
    """枚举手牌能打出的所有牌型（自由出牌时的动作集合）。"""
    groups = group_by_rank(hand)
    combos: List[Combo] = []

    for rank, cards in sorted(groups.items()):
        combos.append(Combo(SINGLE, rank, 1, tuple(cards[:1])))
        if len(cards) >= 2:
            combos.append(Combo(PAIR, rank, 1, tuple(cards[:2])))
        if len(cards) >= 3:
            combos.append(Combo(TRIPLE, rank, 1, tuple(cards[:3])))
        if len(cards) >= 4:
            combos.append(Combo(BOMB, rank, 1, tuple(cards[:4])))

    combos.extend(_straights(groups))
    combos.extend(_pair_straights(groups))
    combos.extend(_planes(groups))
    combos.extend(_triples_with_attachments(groups))
    return combos


def legal_moves(hand: Sequence[Card], required: Optional[Combo]) -> List[Combo]:
    """当前牌面下所有能打出的牌（不含"过"，过牌由引擎处理）。

    自由出牌时枚举全部牌型；跟牌时只枚举同型同长且更大的，外加所有炸弹。
    """
    if required is None:
        return all_combos(hand)

    groups = group_by_rank(hand)
    moves: List[Combo] = []

    if required.kind == BOMB:
        return [
            Combo(BOMB, rank, 1, tuple(cards[:4]))
            for rank, cards in sorted(groups.items())
            if len(cards) >= 4 and rank > required.rank
        ]

    for combo in _combos_of_kind(groups, required.kind, required.length):
        if combo.rank > required.rank:
            moves.append(combo)

    # 炸弹可以压任何非炸弹牌型
    moves.extend(
        Combo(BOMB, rank, 1, tuple(cards[:4]))
        for rank, cards in sorted(groups.items())
        if len(cards) >= 4
    )
    return moves


def _combos_of_kind(groups: Dict[int, List[Card]], kind: str, length: int) -> List[Combo]:
    """只生成指定牌型和连长的牌，避免枚举整个动作空间。"""
    if kind == SINGLE:
        return [Combo(SINGLE, rank, 1, tuple(cards[:1])) for rank, cards in sorted(groups.items())]
    if kind == PAIR:
        return [Combo(PAIR, rank, 1, tuple(cards[:2])) for rank, cards in sorted(groups.items()) if len(cards) >= 2]
    if kind == TRIPLE:
        return [Combo(TRIPLE, rank, 1, tuple(cards[:3])) for rank, cards in sorted(groups.items()) if len(cards) >= 3]
    if kind in (TRIPLE_ONE, TRIPLE_TWO):
        return [c for c in _triples_with_attachments(groups) if c.kind == kind]
    if kind == STRAIGHT:
        return [c for c in _straights(groups) if c.length == length]
    if kind == PAIR_STRAIGHT:
        return [c for c in _pair_straights(groups) if c.length == length]
    if kind in (PLANE, PLANE_ONE, PLANE_TWO):
        return [c for c in _planes(groups) if c.kind == kind and c.length == length]
    return []


def _straights(groups: Dict[int, List[Card]]) -> List[Combo]:
    combos = []
    for run in _runs(list(groups), MIN_STRAIGHT):
        cards = tuple(groups[rank][0] for rank in run)
        combos.append(Combo(STRAIGHT, max(run), len(run), cards))
    return combos


def _pair_straights(groups: Dict[int, List[Card]]) -> List[Combo]:
    pair_ranks = [rank for rank, cards in groups.items() if len(cards) >= 2]
    combos = []
    for run in _runs(pair_ranks, MIN_PAIR_STRAIGHT):
        cards = tuple(card for rank in run for card in groups[rank][:2])
        combos.append(Combo(PAIR_STRAIGHT, max(run), len(run), cards))
    return combos


def _planes(groups: Dict[int, List[Card]]) -> List[Combo]:
    triple_ranks = [rank for rank, cards in groups.items() if len(cards) >= 3]
    combos: List[Combo] = []

    for run in _runs(triple_ranks, MIN_PLANE):
        k = len(run)
        body = tuple(card for rank in run for card in groups[rank][:3])
        used = set(run)
        combos.append(Combo(PLANE, max(run), k, body))

        for attach in _attachment_sets(groups, used, k, size=1):
            combos.append(Combo(PLANE_ONE, max(run), k, body + attach))
        for attach in _attachment_sets(groups, used, k, size=2):
            combos.append(Combo(PLANE_TWO, max(run), k, body + attach))
    return combos


def _triples_with_attachments(groups: Dict[int, List[Card]]) -> List[Combo]:
    combos: List[Combo] = []
    for rank, cards in sorted(groups.items()):
        if len(cards) < 3:
            continue
        body = tuple(cards[:3])
        for attach in _attachment_sets(groups, {rank}, 1, size=1):
            combos.append(Combo(TRIPLE_ONE, rank, 1, body + attach))
        for attach in _attachment_sets(groups, {rank}, 1, size=2):
            combos.append(Combo(TRIPLE_TWO, rank, 1, body + attach))
    return combos


def _attachment_sets(
    groups: Dict[int, List[Card]], used_ranks: set, count: int, size: int
) -> List[Tuple[Card, ...]]:
    """挑 `count` 个点数，每个点数出 `size` 张作为带牌。

    候选按"拆得起"排序（先用零散单张，再用小牌），飞机带牌时最多生成
    MAX_ATTACH_SETS 种方案，避免动作空间爆炸。
    """
    candidates = [
        rank for rank, cards in groups.items()
        if rank not in used_ranks and len(cards) >= size
    ]
    if len(candidates) < count:
        return []

    candidates.sort(key=lambda rank: (len(groups[rank]), rank))
    if count == 1:
        chosen = [(rank,) for rank in sorted(candidates)]
    else:
        pool = candidates[: max(count, min(len(candidates), count + 2))]
        chosen = list(itertools.combinations(sorted(pool), count))[:MAX_ATTACH_SETS]

    return [tuple(card for rank in combo for card in groups[rank][:size]) for combo in chosen]


# ------------------------------------------------------- 手牌拆解（启发式）


def estimate_turns(hand: Sequence[Card]) -> int:
    """贪心估计"打完这手牌至少还要出几轮"，越小越好。

    规则机器人和策略网络都用它衡量一次出牌把手牌拆得好不好。花色不影响结果，
    所以按"每个点数几张"做签名缓存——训练时每步要算几十次，这个缓存很关键。
    """
    if not hand:
        return 0
    items = sorted(group_by_rank(hand).items())
    return _estimate_turns_cached(
        tuple(rank for rank, _ in items), tuple(len(cards) for _, cards in items)
    )


@functools.lru_cache(maxsize=1 << 17)
def _estimate_turns_cached(ranks: Tuple[int, ...], signature: Tuple[int, ...]) -> int:
    counts = dict(zip(ranks, signature))
    turns = 0

    # 炸弹整体留着，算一轮
    for rank in [r for r, c in counts.items() if c == 4]:
        counts.pop(rank)
        turns += 1

    # 顺子：长的优先
    while True:
        runs = _runs([r for r, c in counts.items() if c >= 1], MIN_STRAIGHT)
        if not runs:
            break
        run = max(runs, key=len)
        for rank in run:
            counts[rank] -= 1
            if counts[rank] == 0:
                counts.pop(rank)
        turns += 1

    # 连对
    while True:
        runs = _runs([r for r, c in counts.items() if c >= 2], MIN_PAIR_STRAIGHT)
        if not runs:
            break
        run = max(runs, key=len)
        for rank in run:
            counts[rank] -= 2
            if counts[rank] == 0:
                counts.pop(rank)
        turns += 1

    triples = sum(1 for c in counts.values() if c == 3)
    pairs = sum(1 for c in counts.values() if c == 2)
    singles = sum(1 for c in counts.values() if c == 1)

    turns += triples + pairs + singles
    # 三带一可以顺手带走零散单张，省下的轮次要扣掉
    turns -= min(triples, singles)
    return max(turns, 1)
