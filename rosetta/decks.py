"""套牌定义。

用卡 ID 而不是卡名——同一个名字在 RosettaStone 里往往有四五张卡
（`CS2_179` / `CORE_CS2_179` / `TU5_CS2_179` / `VAN_CS2_179` 都叫
"Sen'jin Shieldmasta"），按名字找到哪一张不确定。
"""

from __future__ import annotations

import random

DECK_SIZE = 30

# ================================================================ Vanilla 固定套牌

#: 经典白板套牌：15 种中立随从各两张，只有关键词、没有战吼亡语。
VANILLA_IDS = [
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

# ================================================================ Core 固定套牌

#: Core 系列固定套牌（手动精选的 15 种卡，用于对照实验）。
CORE_DECK_IDS = [
    "CORE_BOT_453",   # 流星          1费 对一个随从及相邻随从造成1点伤害
    "CORE_CS2_023",   # 奥术智慧      3费 抽2张牌
    "CORE_CS2_029",   # 火球术        4费 造成6点伤害
    "CORE_CS2_032",   # 烈焰风暴      7费 对所有敌方随从造成5点伤害
    "CORE_EX1_010",   # 狼人渗透者    2/1 潜行
    "CORE_CS2_189",   # 精灵弓箭手    1/1 战吼：造成1点伤害
    "CORE_CS2_188",   # 叫嚣的中士    1/1 战吼：本回合使一个随从+2攻
    "CORE_GVG_085",   # 吵吵机器人    1/2 嘲讽 圣盾
    "CORE_NEW1_023",  # 精灵龙        3/2 扰咒
    "CORE_EX1_096",   # 战利品贮藏者  2/1 亡语：抽一张牌
    "CORE_UNG_928",   # 焦油爬行者    1/5 嘲讽
    "CORE_CS2_203",   # 铁喙猫头鹰    2/1 战吼：沉默一个随从
    "CORE_CS2_182",   # 冰风雪人      4/5
    "CORE_CS2_179",   # 森金持盾卫士  3/5 嘲讽
    "CORE_EX1_284",   # 碧蓝幼龙      4/5 法术伤害+1 战吼：抽一张牌
]

# ================================================================ Core 随机池

#: Core 随机池——已通过 RuleBot 200 局压力测试的卡牌。
#: 传说最多带 1 张，其他 2 张。部分卡牌因 RosettaStone C++ 内存 bug 被剔除。
_CORE_POOL_RAW: dict[str, int] = {
    # --- Mage 法术 ---
    "CORE_CS2_023": 2,   # 奥术智慧      3费 抽2张
    "CORE_CS2_029": 2,   # 火球术        4费 6伤
    "CORE_CS2_032": 2,   # 烈焰风暴      7费 全体5伤
    # --- 中立 0 费 ---
    "CORE_LOEA10_3": 2,  # 鱼人小鳍      1/1
    # --- 中立 1 费 ---
    "CORE_CS2_188": 2,   # 叫嚣的中士    1/1 战吼+2攻
    "CORE_ULD_191": 2,   # 微笑的助教    1/2 战吼+2生命值
    "CORE_CS2_189": 2,   # 精灵弓箭手    1/1 战吼1伤
    "CORE_EX1_011": 2,   # 巫医          2/1 战吼回2血
    "CORE_EX1_010": 2,   # 狼人渗透者    2/1 潜行
    # --- 中立 2 费 ---
    "CORE_GVG_085": 2,   # 吵吵机器人    1/2 嘲讽圣盾
    "CORE_NEW1_023": 2,  # 精灵龙        3/2 扰咒
    "CORE_ULD_271": 2,   # 受伤的托维尔  2/6 嘲讽战吼自伤3
    "CORE_CS2_142": 2,   # 狗头人地卜师  2/2 法伤+1
    "CORE_EX1_096": 2,   # 战利品贮藏者  2/1 亡语抽牌
    "CORE_FP1_007": 2,   # 蛛魔之卵      0/2 亡语招4/4
    "CORE_EX1_049": 2,   # 年轻的酒仙    3/2 战吼回收随从
    # --- 中立 3 费 ---
    "CORE_UNG_844": 2,   # 巨型刃叶      4/8 无法攻击
    "CORE_CS2_203": 2,   # 铁喙猫头鹰    2/1 战吼沉默
    "CORE_EX1_017": 2,   # 丛林猎豹      4/2 潜行
    "CORE_UNG_928": 2,   # 焦油爬行者    1/5 嘲讽
    # --- 中立 4 费 ---
    "CORE_EX1_005": 2,   # 王牌猎人      4/2 战吼杀7+攻
    "CORE_CS2_182": 2,   # 冰风雪人      4/5
    "CORE_EX1_046": 2,   # 黑铁矮人      4/4 战吼+2攻
    "CORE_EX1_093": 2,   # 阿古斯防御者  3/3 战吼相邻+1/+1嘲讽
    "CORE_YOD_006": 2,   # 逃脱的刃豹    3/5 潜行
    "CORE_CS2_179": 2,   # 森金持盾卫士  3/5 嘲讽
    # --- 中立 5 费 ---
    "CORE_EX1_284": 2,   # 碧蓝幼龙      4/5 法伤+1战吼抽牌
    "CORE_EX1_028": 2,   # 荆棘谷猛虎    5/5 潜行
}

#: 加权池：每张卡按可带张数展开。
CORE_POOL_WEIGHTED: list[str] = [
    cid for cid, copies in _CORE_POOL_RAW.items()
    for _ in range(copies)
]


def random_core(rng: random.Random | None = None) -> list[str]:
    """从 Core 池随机组 30 张牌。

    每张卡不超过上限（传说 1 张，其他 2 张）。
    每局独立抽——模型要适应不同的对局分布。
    """
    if rng is None:
        rng = random.Random()
    pool = list(CORE_POOL_WEIGHTED)
    rng.shuffle(pool)
    return pool[:DECK_SIZE]


# ================================================================ 公共接口

#: 所有可用套牌的注册表（固定 ID 列表，会经过 build_deck ×2）。
DECKS: dict[str, list[str]] = {
    "vanilla": VANILLA_IDS,
    "core": CORE_DECK_IDS,
}


def vanilla() -> list[str]:
    """经典白板套牌，30 张（每种两张）。"""
    return build_deck(VANILLA_IDS)


def core() -> list[str]:
    """Core 系列固定套牌，30 张（每种两张）。"""
    return build_deck(CORE_DECK_IDS)


def build_deck(ids: list[str]) -> list[str]:
    """把卡牌 ID 列表变成 30 张实牌（每种两张）。"""
    deck = [card_id for card_id in ids for _ in range(2)]
    assert len(deck) == DECK_SIZE, f"套牌是 {len(deck)} 张，应该是 {DECK_SIZE} 张"
    return deck
