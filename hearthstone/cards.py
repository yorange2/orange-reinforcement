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

十个关键词全部生效。扰咒挡的是**指定目标**——AoE、溅射和奥术飞弹的随机伤害照样
打得到它；法术增强只加**伤害**，变形、消灭、乱斗、抽牌都不受影响。
"""

from __future__ import annotations

import random
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

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

#: 曾经有两个关键词没实现，现在全部生效了。留空元组是为了不破坏外部引用。
INERT_KEYWORDS: Tuple[str, ...] = ()


class Effect(NamedTuple):
    """一段可结算的效果。**法术、战吼、亡语共用同一套结算**。

    这样"对目标造成 3 点伤害"的战吼和火球术就是同一段代码，加卡的时候只需要描述
    效果，不需要再往引擎里塞分支。结算见 `game.Game._resolve`。

    `needs_target` 为真的效果在出牌时要指定目标；扰咒的随从不能被指定。
    """

    # --- 伤害（都吃法术增强，且都在 _resolve 里统一 +bonus）---
    damage: int = 0                 # 对指定目标
    splash: int = 0                 # 目标以外的敌方随从（横扫）
    missiles: int = 0               # 随机 1 点伤害 N 次
    aoe_enemy_minions: int = 0
    aoe_all_enemies: int = 0        # 敌方随从 + 英雄
    aoe_all: int = 0                # 所有角色，含自己和自己的随从

    # --- 恢复与增益 ---
    heal: int = 0                   # 对指定目标
    buff_attack: int = 0
    buff_health: int = 0
    #: 增益/治疗作用于谁："target" / "friendly"（己方全体）/ "friendly_others"（己方其他）
    scope: str = "target"
    grant: Tuple[str, ...] = ()     # 赋予关键词

    # --- 牌与场面 ---
    draw: int = 0
    summon: str = ""                # 衍生物卡名，见 TOKENS
    summon_count: int = 0

    # --- 冻结 ---
    freeze_target: bool = False     # 冻结指定的那个角色（寒冰箭、冰霜元素）
    freeze_enemy_minions: bool = False   # 冻结所有敌方随从（冰霜新星、暴风雪）

    # --- 特殊 ---
    transform: bool = False         # 变成 1/1 绵羊
    destroy_target: bool = False
    destroy_all: bool = False
    brawl: bool = False             # 随机只留一个随从

    @property
    def needs_target(self) -> bool:
        """要不要在出牌时指定一个目标。"""
        return bool(
            self.damage or self.transform or self.destroy_target or self.freeze_target
            or (self.heal and self.scope == "target")
            or ((self.buff_attack or self.buff_health or self.grant)
                and self.scope == "target")
        )

    @property
    def deals_damage(self) -> bool:
        return bool(self.damage or self.splash or self.missiles
                    or self.aoe_enemy_minions or self.aoe_all_enemies or self.aoe_all)


#: 光环的作用范围。
AURA_OTHERS = "friendly_others"   # 你的**其他**随从（团队领袖、暴风城勇士）
AURA_FRIENDLY = "friendly"        # 你的所有随从，含自己
AURA_ADJACENT = "adjacent"        # 左右相邻的随从（炎锤先锋）


class Aura(NamedTuple):
    """光环：随从在场期间**持续**改变别的随从的属性，离场立刻还原。

    和 `Effect` 里的 `buff_*` 是两回事——那个是一次性永久增益，这个是持续的、
    随光环来源在不在场而生灭。所以它单独存一层（`Minion.aura_*`），由
    `Game._refresh_auras` 整体重算，不做增量加减，避免累积漂移。
    """

    attack: int = 0
    health: int = 0
    scope: str = AURA_OTHERS
    #: 只影响带这个关键词的随从（留空表示不限），例如"你的其他亡灵随从 +1/+1"
    only_keyword: str = ""


class CardDef(NamedTuple):
    """一张卡。`spell` 为真时是法术，`weapon` 为真时是武器。

    卡面文本一律用 `Effect` 描述，三个入口共用同一套结算：

        effect       法术本身的效果（`spell=True` 时）
        battlecry    随从/武器出场时触发
        deathrattle  随从死亡时触发
    """

    name: str
    cost: int
    attack: int = 0
    health: int = 0
    keywords: Tuple[str, ...] = ()
    spell: bool = False
    weapon: bool = False
    effect: Optional[Effect] = None
    battlecry: Optional[Effect] = None
    deathrattle: Optional[Effect] = None
    aura: Optional[Aura] = None

    @property
    def fx(self) -> "Effect":
        """卡面效果，没有就给一个全零的 `Effect`。

        让调用方可以直接写 `card.fx.damage` 而不用每处判空——白板随从的每个字段
        都是 0，语义上正好。
        """
        return self.effect or _NO_EFFECT

    @property
    def needs_target(self) -> bool:
        """出这张牌要不要指定目标——法术看 effect，随从/武器看 battlecry。"""
        source = self.effect if self.spell else self.battlecry
        return source is not None and source.needs_target

    @property
    def stats(self) -> int:
        """攻 + 血/耐久，衡量体量最粗糙的一个数。"""
        return self.attack + self.health

    @property
    def durability(self) -> int:
        """武器耐久度——和 health 是同一个字段，这个别名让调用方更可读。"""
        return self.health

    def has(self, keyword: str) -> bool:
        return keyword in self.keywords

    @property
    def text(self) -> str:
        return "、".join(self.keywords)

    def __str__(self) -> str:
        if self.spell:
            return f"{self.name}({self.cost}费)"
        if self.weapon:
            tail = f" {self.text}" if self.keywords else ""
            return f"{self.name}({self.cost}费 {self.attack}/{self.durability}{tail})"
        tail = f" {self.text}" if self.keywords else ""
        return f"{self.name}({self.cost}费 {self.attack}/{self.health}{tail})"


#: `CardDef.fx` 在卡面没有效果时返回它——每个字段都是 0。
_NO_EFFECT = Effect()

#: 后手的补偿，和炉石一致：本回合额外获得一个法力水晶。
THE_COIN = CardDef("幸运币", 0, spell=True)

#: 变形术的产物。
SHEEP = CardDef("绵羊", 1, 1, 1)

#: 衍生物：只会被召唤到场上，**永远不会进手牌**，所以不进 POOL、也不参与构筑。
#:
#: 卡池必须是"闭"的——任何能进手牌的卡都得在 POOL 里，否则智能体会拿到特征编码
#: 里从没见过的东西。衍生物只出现在场上，而场上随从的特征用的是身材和关键词、
#: 不是卡牌身份，所以它们天然可编码。发现和随机生成卡牌的效果则一律不收。
TOKENS: Dict[str, CardDef] = {
    card.name: card
    for card in [
        SHEEP,
        CardDef("恐狼", 1, 1, 1),
        CardDef("小鬼", 1, 1, 1),
        CardDef("骷髅", 1, 1, 1),
        CardDef("鱼人斥候", 1, 1, 1),
        CardDef("镀银之手新兵", 1, 1, 1),
    ]
}


def _m(name: str, cost: int, attack: int, health: int, *keywords: str) -> CardDef:
    return CardDef(name, cost, attack, health, tuple(keywords))


def _w(name: str, cost: int, attack: int, durability: int) -> CardDef:
    return CardDef(name, cost, attack, durability, weapon=True)


def _spell(name: str, cost: int, **effect) -> CardDef:
    """一张法术。效果字段直接透传给 `Effect`。"""
    return CardDef(name, cost, spell=True, effect=Effect(**effect))


def _mb(name: str, cost: int, attack: int, health: int, *keywords: str,
        **battlecry) -> CardDef:
    """带战吼的随从。"""
    return CardDef(name, cost, attack, health, tuple(keywords),
                   battlecry=Effect(**battlecry))


def _md(name: str, cost: int, attack: int, health: int, *keywords: str,
        **deathrattle) -> CardDef:
    """带亡语的随从。"""
    return CardDef(name, cost, attack, health, tuple(keywords),
                   deathrattle=Effect(**deathrattle))


def _ma(name: str, cost: int, attack: int, health: int, aura: Aura,
        *keywords: str) -> CardDef:
    """带光环的随从。光环显式传 `Aura`——它自己也有 attack/health，跟位置参数会撞名。"""
    return CardDef(name, cost, attack, health, tuple(keywords), aura=aura)


#: 卡池，按费用排序。索引即卡牌 id，编码特征时可以直接用。
POOL: List[CardDef] = [
    # ---- 法术
    _spell("奥术飞弹", 1, missiles=3),
    _spell("刀扇", 3, aoe_enemy_minions=1, draw=1),
    _spell("奥术智慧", 3, draw=2),
    _spell("火球术", 4, damage=6),
    _spell("变形术", 4, transform=True),
    _spell("奉献", 4, aoe_all_enemies=2),
    _spell("地狱烈焰", 4, aoe_all=3),
    _spell("横扫", 4, damage=4, splash=1),
    _spell("绝命乱斗", 5, brawl=True),
    _spell("烈焰风暴", 7, aoe_enemy_minions=4),
    _spell("疾跑", 7, draw=4),
    _spell("扭曲虚空", 8, destroy_all=True),
    # ---- 武器：纯白板，只有攻/耐久
    _w("圣光的正义", 1, 1, 4),
    _w("炽炎战斧", 2, 3, 2),
    _w("刺客之刃", 4, 2, 5),
    _w("奥金斧", 5, 5, 2),
    # ---- 白板随从：卡面一个字都没有
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
    # ---- 扰咒 / 法术增强
    _m("精灵龙", 2, 3, 2, ELUSIVE),
    _m("狗头人地卜师", 2, 2, 2, SPELL_DAMAGE),
    _m("达拉然法师", 3, 1, 4, SPELL_DAMAGE),
    _m("食人魔法师", 4, 4, 4, SPELL_DAMAGE),
    _m("大法师", 6, 4, 7, SPELL_DAMAGE),
    # ---- 战吼（核心系列，第一批）
    _mb("工程师学徒", 1, 1, 1, draw=1),
    _mb("闪金镇步兵长", 2, 2, 2, buff_attack=1, scope="friendly_others"),
    _mb("北郡牧师", 3, 3, 2, heal=2, scope="target"),
    _mb("阿古斯防御者", 4, 2, 3, grant=(TAUNT,), scope="friendly_others"),
    _mb("火车王里诺艾", 5, 4, 2, damage=2),
    _mb("恐怖的奴隶主", 5, 5, 4, buff_attack=1, buff_health=1, scope="friendly_others"),
    # ---- 亡语（核心系列，第一批）
    _md("鱼人猎潮者", 2, 2, 1, summon="鱼人斥候", summon_count=1),
    _md("恐狼前锋", 2, 2, 2, summon="恐狼", summon_count=1),
    _md("巫毒医生", 2, 2, 1, summon="骷髅", summon_count=1),
    _md("憎恶", 5, 4, 4, aoe_all=2),
    _md("石爪野猪", 3, 3, 2, summon="小鬼", summon_count=1),
    # ---- 冻结（法术）
    _spell("冰霜震击", 1, damage=1, freeze_target=True),
    _spell("寒冰箭", 2, damage=3, freeze_target=True),
    _spell("冰霜新星", 3, freeze_enemy_minions=True),
    _spell("暴风雪", 6, aoe_enemy_minions=2, freeze_enemy_minions=True),
    # ---- 冻结（战吼）
    _mb("冰川裂片", 1, 2, 1, freeze_target=True),
    _mb("冰霜元素", 6, 5, 5, freeze_target=True),
    # ---- 光环
    _ma("团队领袖", 3, 2, 2, Aura(attack=1)),
    _ma("石堡卫士", 4, 3, 4, Aura(health=1)),
    _ma("暴风城勇士", 7, 6, 6, Aura(attack=1, health=1)),
    _ma("炎锤先锋", 4, 3, 3, Aura(attack=1, scope=AURA_ADJACENT)),
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
