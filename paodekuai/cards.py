"""牌的表示：点数、花色、整副牌与发牌。

三人跑得快变体：54 张去掉大小王和四个 2，剩 48 张，每人 16 张。
点数顺序 3 < 4 < ... < 10 < J < Q < K < A，没有 2 和王，所以顺子不需要处理回绕。
"""

from __future__ import annotations

import random
from typing import Dict, List, NamedTuple, Sequence

# 点数 3..14（14 = A）。2 已被移出牌堆。
MIN_RANK = 3
MAX_RANK = 14
RANKS: List[int] = list(range(MIN_RANK, MAX_RANK + 1))
RANK_NAMES: Dict[int, str] = {
    3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9", 10: "10",
    11: "J", 12: "Q", 13: "K", 14: "A",
}

# 花色只影响"谁先出"（方块 3 先手），不参与大小比较。
# 顺序固定为方块最小，这样按花色排序取到的 3 一定是 ♦3。
SUIT_SYMBOLS: List[str] = ["♦", "♣", "♥", "♠"]
DIAMOND = 0

N_PLAYERS = 3
HAND_SIZE = 16


class Card(NamedTuple):
    """一张牌。按 (点数, 花色) 排序。"""

    rank: int
    suit: int

    def __str__(self) -> str:
        return f"{SUIT_SYMBOLS[self.suit]}{RANK_NAMES[self.rank]}"


#: 首出必须包含的那张牌。
DIAMOND_THREE = Card(MIN_RANK, DIAMOND)


def full_deck() -> List[Card]:
    """48 张牌的整副牌。"""
    return [Card(rank, suit) for rank in RANKS for suit in range(len(SUIT_SYMBOLS))]


def deal(rng: random.Random, n_players: int = N_PLAYERS, hand_size: int = HAND_SIZE) -> List[List[Card]]:
    """洗牌并发牌，返回每家排好序的手牌。"""
    deck = full_deck()
    if n_players * hand_size > len(deck):
        raise ValueError(f"{n_players} 家 x {hand_size} 张 超过了牌堆的 {len(deck)} 张")

    rng.shuffle(deck)
    return [sorted(deck[i * hand_size : (i + 1) * hand_size]) for i in range(n_players)]


def hand_to_str(cards: Sequence[Card]) -> str:
    """把一手牌排序后拼成可读字符串。"""
    return " ".join(str(card) for card in sorted(cards))


def rank_counts(cards: Sequence[Card]) -> Dict[int, int]:
    """统计每个点数有几张。"""
    counts: Dict[int, int] = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def parse_card(text: str) -> Card:
    """解析 '♦3' / 'd3' / '3' 这类写法，花色缺省为方块。"""
    text = text.strip()
    if not text:
        raise ValueError("空的牌面")

    suit = DIAMOND
    letters = {"d": 0, "c": 1, "h": 2, "s": 3}
    if text[0] in SUIT_SYMBOLS:
        suit = SUIT_SYMBOLS.index(text[0])
        text = text[1:]
    elif text[0].lower() in letters and len(text) > 1:
        suit = letters[text[0].lower()]
        text = text[1:]

    name = text.strip().upper()
    for rank, rank_name in RANK_NAMES.items():
        if rank_name == name:
            return Card(rank, suit)
    raise ValueError(f"无法识别的牌面: {text!r}")
