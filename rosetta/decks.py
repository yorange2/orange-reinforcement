"""套牌定义。

用卡 ID 而不是卡名——同一个名字在 RosettaStone 里往往有四五张卡
（`CS2_179` / `CORE_CS2_179` / `TU5_CS2_179` / `VAN_CS2_179` 都叫
"Sen'jin Shieldmasta"），按名字找到哪一张不确定。
"""

from __future__ import annotations

DECK_SIZE = 30

#: 经典白板套牌：15 种中立随从各两张，只有关键词、没有战吼亡语。
#: 刻意和 `hearthstone/` 那个自研卡池同源，方便两边的胜率互相参照。
VANILLA = [
    "CS2_231",   # 幽灵          0费 1/1
    "EX1_008",   # 银色侍从      1费 1/1 圣盾
    "CS2_171",   # 石牙野猪      1费 1/1 冲锋
    "CS2_172",   # 血沼迅猛龙    2费 3/2
    "CS2_173",   # 蓝腮战士      2费 2/1 冲锋
    "CS2_121",   # 霜狼步兵      2费 2/2 嘲讽
    "NEW1_023",  # 精灵龙        2费 3/2 扰咒
    "CS2_124",   # 狼骑兵        3费 3/1 冲锋
    "CS2_125",   # 铁鬃灰熊      3费 3/3 嘲讽
    "EX1_020",   # 血色十字军战士 3费 3/1 圣盾
    "CS2_182",   # 冰风雪人      4费 4/5
    "CS2_179",   # 森金持盾卫士  4费 3/5 嘲讽
    "EX1_023",   # 银月城卫兵    4费 3/3 圣盾
    "EX1_028",   # 荆棘谷猛虎    5费 5/5 潜行
    "CS2_200",   # 石拳食人魔    6费 6/7
]


def vanilla() -> list[str]:
    """经典白板套牌，30 张（每种两张）。"""
    deck = [card_id for card_id in VANILLA for _ in range(2)]
    assert len(deck) == DECK_SIZE, f"套牌是 {len(deck)} 张，应该是 {DECK_SIZE} 张"
    return deck
