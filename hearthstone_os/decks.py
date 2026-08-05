"""套牌定义（G9 子集）。

卡 ID 一律用 orange-stone 的（如 `CLASSIC_001`）；对拍测试要映射回
`hearthstone/` 简化引擎时用 `SUBSET_MAP`（os 卡 ID → 简版卡名）。

G9 子集的入选标准（路线图 §5 风险对策）：**只收两个引擎语义一致的卡**——
白板 + 基础关键词（冲锋/嘲讽/圣盾/风怒）。orange-stone 的潜行/扰咒关键词
目前没实现（2026-08 实测：丛林豹/荆棘谷猛虎/精灵龙在结构化视图里没有对应
字段），这几张不进子集，等 M5 引擎补上再扩。
"""

from __future__ import annotations

DECK_SIZE = 30

#: os 卡 ID → 简版引擎卡名（`hearthstone.cards.POOL` 里的名字）。
#: 每张卡的费用/身材/关键词都经两引擎实证核对一致（费用见 os 卡面，
#: 身材/关键词见各自实现，对拍测试兜底验证）。
SUBSET_MAP: dict[str, str] = {
    # ---- 白板
    "NEUTRAL_T01": "幽灵",           # 0/1/1
    "NEUTRAL_B02": "鱼人袭击者",     # 1/2/1
    "CLASSIC_001": "血沼迅猛龙",     # 2/3/2
    "NEUTRAL_T05": "河鳄",           # 2/2/3
    "NEUTRAL_B09": "岩浆暴怒者",     # 3/5/1
    "NEUTRAL_T08": "冰风雪人",       # 4/4/5
    "NEUTRAL_B13": "绿洲钳嘴龟",     # 4/2/7
    "NEUTRAL_T09": "石拳食人魔",     # 6/6/7
    "NEUTRAL_025": "熔火恶犬",       # 7/9/5
    "NEUTRAL_T11": "战争傀儡",       # 7/7/7
    # ---- 冲锋
    "NEUTRAL_B03": "石牙野猪",       # 1/1/1
    "CLASSIC_002": "蓝腮战士",       # 2/2/1
    "CLASSIC_017": "狼骑兵",         # 3/3/1
    "NEUTRAL_B14": "暴风城骑士",     # 4/2/5
    "NEUTRAL_023": "鲁莽火箭兵",     # 6/5/2
    # ---- 嘲讽
    "NEUTRAL_T03": "闪金镇步兵",     # 1/1/2
    "NEUTRAL_B05": "霜狼步兵",       # 2/2/2
    "NEUTRAL_B08": "铁鬃灰熊",       # 3/3/3
    "NEUTRAL_B11": "银背族长",       # 3/1/4
    "CLASSIC_008": "森金持盾卫士",   # 4/3/5
    "NEUTRAL_016": "魔古山守望者",   # 4/1/7
    "NEUTRAL_B15": "藏宝海湾保镖",   # 5/5/4
    "NEUTRAL_022": "竞技场主宰",     # 6/6/5
    # ---- 圣盾
    "NEUTRAL_C01": "银色侍从",       # 1/1/1
    "NEUTRAL_009": "血色十字军战士", # 3/3/1
    "NEUTRAL_014": "银月城卫兵",     # 4/3/3
    # ---- 风怒
    "NEUTRAL_C04": "年轻的多头龙鹰", # 1/1/1
    "CLASSIC_016": "风怒鹰身人",     # 6/4/5
    # ---- 潜行（orange-stone #72 之后实现）
    "NEUTRAL_C10": "丛林豹",         # 3/4/2
    "NEUTRAL_T14": "荆棘谷猛虎",     # 5/5/5
}

#: G9 子集卡 ID（供对拍测试构造卡池）。
SUBSET_IDS: list[str] = list(SUBSET_MAP)

#: 固定 15 种镜像套牌：白板 + 基础关键词，两个引擎语义一致（M2 默认评测口径）。
VANILLA_IDS: list[str] = [
    "NEUTRAL_T01",   # 幽灵          0/1/1
    "NEUTRAL_C01",   # 银色侍从      1/1/1 圣盾
    "NEUTRAL_B03",   # 石牙野猪      1/1/1 冲锋
    "NEUTRAL_B02",   # 鱼人袭击者    1/2/1
    "CLASSIC_001",   # 血沼迅猛龙    2/3/2
    "CLASSIC_002",   # 蓝腮战士      2/2/1 冲锋
    "NEUTRAL_B05",   # 霜狼步兵      2/2/2 嘲讽
    "CLASSIC_017",   # 狼骑兵        3/3/1 冲锋
    "NEUTRAL_B08",   # 铁鬃灰熊      3/3/3 嘲讽
    "NEUTRAL_009",   # 血色十字军战士 3/3/1 圣盾
    "NEUTRAL_T08",   # 冰风雪人      4/4/5
    "CLASSIC_008",   # 森金持盾卫士  4/3/5 嘲讽
    "NEUTRAL_014",   # 银月城卫兵    4/3/3 圣盾
    "NEUTRAL_C10",   # 丛林豹        3/4/2 潜行（orange-stone #72 后）
    "CLASSIC_016",   # 风怒鹰身人    6/4/5 风怒
]


def vanilla() -> list[str]:
    """15 种 × 2 张 = 30 张镜像套牌（M2 的默认评测口径）。"""
    return build_deck(VANILLA_IDS)


def build_deck(ids: list[str]) -> list[str]:
    """把卡牌 ID 列表变成 30 张实牌（每种两张）。"""
    deck = [card_id for card_id in ids for _ in range(2)]
    assert len(deck) == DECK_SIZE, f"套牌是 {len(deck)} 张，应该是 {DECK_SIZE} 张"
    return deck
