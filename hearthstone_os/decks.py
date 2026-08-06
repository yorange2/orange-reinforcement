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


# ---------------------------------------------------------------- M5 全卡池

#: 引擎侧有记录简化债的卡（orange-stone 源码 "simplified" 注释，2026-08 核对 68 处）：
#: 语义与真实炉石有偏差，RL 训练卡池不用（路线图 §5 风险对策：只收已实现且
#: 语义一致的卡）。
#: 权威清单在 orange-stone/docs/finished/fidelity-debt.md（F4/F5 持续审计账本，已归档）——
#: 卡离开账本（实现 + F5 差分验证）后，删掉源码注释里的 "simplified" 字样，
#: 本提取器就会自动把它放回卡池；注意失效 ~/.cache/orange_stone_debt_ids.txt 缓存。
DEBT_IDS: set[str] = {
    # 示例：Tauren Warrior（enrage 简化成只有嘲讽）——完整列表以账本为准。
}

#: 全经典卡池（ALL_CARDS 410 张，含硬币/衍生物；过滤掉不干净的定义后由
#: `full_pool()` 给出可用的构筑池）。
def full_pool() -> list[str]:
    """M5 全经典构筑池：ALL_CARDS 里所有可入套牌的卡。

    过滤规则（路线图 §5 风险对策——训练卡池只用已实现且语义一致的卡）：
    - 去掉硬币（GAME_005）与纯衍生物（id 以 't' 结尾）
    - 去掉引擎侧有简化债注释的卡（DEBT_IDS 由 orange-stone 源码核对）
    实测每张卡都能正常打出（tools/orange_stone_m5_smoke.py 的卡池压力测试）。
    """
    import orange_stone as os

    ids = os.GameEnv.all_card_ids()
    debt = _load_debt_ids()
    out = [
        cid for cid in ids
        if cid not in debt
        and not cid.endswith("t")
        and cid != "GAME_005"
    ]
    return out


def _load_debt_ids() -> set[str]:
    """从 orange-stone 源码提取简化债卡 ID（构建期核对一次，结果缓存）。

    每处带 "simplified" 的文档注释都紧贴它所描述的卡牌常量（`///` 在
    `pub const` 正上方），所以从注释行向后找**下一个** `pub const` 才是
    正确的卡。之前按整块正则切分会把注释记到**上一张**卡头上（2026-08-06
    审计发现：321 卡池里混进约 12 张简化卡、漏掉约 15 张干净卡）。
    """
    import os as _os

    cache = _os.path.expanduser("~/.cache/orange_stone_debt_ids.txt")
    if _os.path.exists(cache):
        return set(open(cache).read().split())
    # 运行时解析源码（仓库就在旁边）
    src_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "..", "..", "orange-stone", "src", "cards")
    ids: set[str] = set()
    import glob as _glob
    import re as _re
    for f in _glob.glob(_os.path.join(src_dir, "classic_*.rs")):
        lines = open(f).read().split("\n")
        for i, line in enumerate(lines):
            if "simplified" not in line or "///" not in line:
                continue
            # 注释正下方的常量定义：pub const NAME: CardDef = vanilla!(...) 或 = CardDef { ... }
            m = None
            for j in range(i + 1, min(i + 8, len(lines))):
                m = _re.search(r'pub const \w+: CardDef = vanilla!\("([^"]+)"', lines[j])
                if m:
                    break
                m = _re.search(r'pub const \w+: CardDef = CardDef', lines[j])
                if m:
                    m = None
                    for k in range(j, min(j + 10, len(lines))):
                        m = _re.search(r'id: "([^"]+)"', lines[k])
                        if m:
                            break
                    break
            if m:
                ids.add(m.group(1))
    try:
        _os.makedirs(_os.path.dirname(cache), exist_ok=True)
        open(cache, "w").write("\n".join(sorted(ids)))
    except OSError:
        pass
    return ids


def random_deck(rng: "random.Random | None" = None) -> list[str]:
    """从全经典构筑池随机组 30 张（M5 套牌构筑逻辑，模型要适应不同对局分布）。"""
    import random as _random

    if rng is None:
        rng = _random.Random()
    pool = full_pool()
    rng.shuffle(pool)
    return pool[:DECK_SIZE]
