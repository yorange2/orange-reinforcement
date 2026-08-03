"""卡池、关键词与套牌。

收录标准：卡面上**只有关键词**，没有战吼、亡语、光环、触发等任何额外文本。
费用、身材、关键词全部照抄原版炉石。支持的关键词：

    冲锋  出场当回合就能攻击，随从和英雄都能打
    突袭  出场当回合能攻击随从，但不能打脸
    嘲讽  对方必须先打它（潜行状态下不生效）
    潜行  不能被攻击；自己发动攻击后失去潜行
    圣盾  第一次受到伤害时完全免疫，然后圣盾消失
    剧毒  被它造成伤害的随从直接死亡（伤害被圣盾挡下则不触发）
    风怒  每回合可以攻击两次
    吸血  它造成多少伤害，自己的英雄就回多少血（不超过上限）
    复生  第一次死亡后带着 1 点血回到场上，且不再拥有复生
    扰咒  不能成为法术和英雄技能的目标
    法术增强  自己的法术伤害 +1

最后两个在当前版本里**不产生任何效果**——这版没有法术（除了幸运币）也没有英雄技能，
"不能被指向"和"法术伤害 +1"都无从触发。卡还是照原样收进来了，关键词会显示、也会进
特征向量，等以后加了法术就直接生效。
"""

from __future__ import annotations

import random
from typing import Dict, List, NamedTuple, Sequence, Tuple

# ---------------------------------------------------------------- 关键词

CHARGE = "冲锋"
RUSH = "突袭"
TAUNT = "嘲讽"
STEALTH = "潜行"
DIVINE_SHIELD = "圣盾"
POISONOUS = "剧毒"
WINDFURY = "风怒"
LIFESTEAL = "吸血"
REBORN = "复生"
ELUSIVE = "扰咒"
SPELL_DAMAGE = "法术增强"

#: 全部关键词，顺序固定——编码 one-hot 特征时直接用下标。
KEYWORDS: Tuple[str, ...] = (
    CHARGE, RUSH, TAUNT, STEALTH, DIVINE_SHIELD,
    POISONOUS, WINDFURY, LIFESTEAL, REBORN, ELUSIVE, SPELL_DAMAGE,
)
KEYWORD_INDEX: Dict[str, int] = {word: i for i, word in enumerate(KEYWORDS)}

#: 这两个在当前版本里没有效果，原因见模块文档。
INERT_KEYWORDS: Tuple[str, ...] = (ELUSIVE, SPELL_DAMAGE)


class CardDef(NamedTuple):
    """一张卡。`spell` 为真时是法术（这版只有幸运币一张）。"""

    name: str
    cost: int
    attack: int = 0
    health: int = 0
    keywords: Tuple[str, ...] = ()
    spell: bool = False

    @property
    def stats(self) -> int:
        """攻血总和，衡量随从体量最粗糙的一个数。"""
        return self.attack + self.health

    def has(self, keyword: str) -> bool:
        return keyword in self.keywords

    @property
    def text(self) -> str:
        return "、".join(self.keywords)

    def __str__(self) -> str:
        if self.spell:
            return f"{self.name}({self.cost}费)"
        tail = f" {self.text}" if self.keywords else ""
        return f"{self.name}({self.cost}费 {self.attack}/{self.health}{tail})"


#: 后手的补偿，和炉石一致：本回合额外获得一个法力水晶。
THE_COIN = CardDef("幸运币", 0, spell=True)


def _m(name: str, cost: int, attack: int, health: int, *keywords: str) -> CardDef:
    return CardDef(name, cost, attack, health, tuple(keywords))


#: 卡池，按费用排序。索引即卡牌 id，编码特征时可以直接用。
POOL: List[CardDef] = [
    # ---- 白板：卡面一个字都没有
    _m("幽灵", 0, 1, 1),
    _m("鱼人袭击者", 1, 2, 1),
    _m("血沼迅猛龙", 2, 3, 2),
    _m("河鳄", 2, 2, 3),
    _m("岩浆暴怒者", 3, 5, 1),
    _m("冰风雪人", 4, 4, 5),
    _m("绿洲钳嘴龟", 4, 2, 7),
    _m("石拳食人魔", 6, 6, 7),
    _m("熔火恶犬", 7, 9, 5),
    _m("战争傀儡", 7, 7, 7),
    # ---- 冲锋
    _m("石牙野猪", 1, 1, 1, CHARGE),
    _m("蓝腮战士", 2, 2, 1, CHARGE),
    _m("狼骑兵", 3, 3, 1, CHARGE),
    _m("暴风城骑士", 4, 2, 5, CHARGE),
    _m("鲁莽火箭兵", 6, 5, 2, CHARGE),
    # ---- 嘲讽
    _m("闪金镇步兵", 1, 1, 2, TAUNT),
    _m("霜狼步兵", 2, 2, 2, TAUNT),
    _m("铁鬃灰熊", 3, 3, 3, TAUNT),
    _m("银背族长", 3, 1, 4, TAUNT),
    _m("森金持盾卫士", 4, 3, 5, TAUNT),
    _m("魔古山守望者", 4, 1, 7, TAUNT),
    _m("藏宝海湾保镖", 5, 5, 4, TAUNT),
    _m("竞技场主宰", 6, 6, 5, TAUNT),
    _m("铁木树人", 8, 8, 8, TAUNT),
    # ---- 潜行
    _m("丛林豹", 3, 4, 2, STEALTH),
    _m("荆棘谷猛虎", 5, 5, 5, STEALTH),
    # ---- 圣盾
    _m("银色侍从", 1, 1, 1, DIVINE_SHIELD),
    _m("血色十字军战士", 3, 3, 1, DIVINE_SHIELD),
    _m("银月城卫兵", 4, 3, 3, DIVINE_SHIELD),
    # ---- 风怒
    _m("年轻的多头龙鹰", 1, 1, 1, WINDFURY),
    _m("风怒鹰身人", 6, 4, 5, WINDFURY),
    # ---- 剧毒
    _m("蛇皇", 3, 2, 3, POISONOUS),
    _m("玛克扎尔", 6, 2, 8, POISONOUS),
    # ---- 吸血
    _m("沼泽水蛭", 1, 2, 1, LIFESTEAL),
    _m("凶恶的鳞甲兽", 2, 1, 3, LIFESTEAL, RUSH),
    # ---- 突袭
    _m("不安分的木乃伊", 4, 3, 2, RUSH, REBORN),
    _m("阿曼尼狂战熊", 7, 5, 7, RUSH, TAUNT),
    # ---- 复生
    _m("骸骨怨灵", 4, 2, 5, TAUNT, REBORN),
    _m("荒野刺客", 5, 4, 2, STEALTH, REBORN),
    # ---- 扰咒 / 法术增强（当前版本里没有效果）
    _m("精灵龙", 2, 3, 2, ELUSIVE),
    _m("狗头人地卜师", 2, 2, 2, SPELL_DAMAGE),
    _m("达拉然法师", 3, 1, 4, SPELL_DAMAGE),
    _m("食人魔法师", 4, 4, 4, SPELL_DAMAGE),
    _m("大法师", 6, 4, 7, SPELL_DAMAGE),
]

#: 卡名 -> 卡池下标。
CARD_INDEX: Dict[str, int] = {card.name: i for i, card in enumerate(POOL)}

#: 和炉石一致：套牌 30 张，同名卡最多 2 张。
DECK_SIZE = 30
COPIES = 2
DISTINCT = DECK_SIZE // COPIES

MAX_COST = max(card.cost for card in POOL)


def build_decklist(rng: random.Random) -> List[CardDef]:
    """随机构筑一份 30 张的套牌：卡池里挑 15 张不同的，每张两份。

    没有做曲线优化——想固定套牌就自己传一份 decklist 给 Game。
    """
    picked = rng.sample(POOL, DISTINCT)
    return [card for card in picked for _ in range(COPIES)]


def shuffled(decklist: Sequence[CardDef], rng: random.Random) -> List[CardDef]:
    """洗好的一副牌。列表末尾是牌堆顶，抽牌时 pop()。"""
    deck = list(decklist)
    rng.shuffle(deck)
    return deck


def by_keyword(keyword: str) -> List[CardDef]:
    """卡池里带某个关键词的卡。"""
    return [card for card in POOL if card.has(keyword)]


def hand_to_str(hand: Sequence[CardDef]) -> str:
    """把一手牌拼成可读字符串，按费用排序。"""
    return " ".join(str(card) for card in sorted(hand, key=lambda c: (c.cost, c.name)))


def parse_card(text: str) -> CardDef:
    """按卡名或卡池下标找一张卡，卡名支持唯一前缀。"""
    text = text.strip()
    if not text:
        raise ValueError("空的卡名")

    if text.isdigit():
        index = int(text)
        if 0 <= index < len(POOL):
            return POOL[index]
        raise ValueError(f"卡池只有 {len(POOL)} 张，没有第 {index} 张")

    hits = [card for card in POOL if card.name.startswith(text)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"卡池里没有 {text!r}")
    raise ValueError(f"{text!r} 有歧义: {', '.join(card.name for card in hits)}")
