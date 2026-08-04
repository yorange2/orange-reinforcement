"""简化版炉石引擎：法力水晶、抽牌、疲劳、出随从、攻击、关键词结算、判定胜负。

规则尽量和炉石对齐：

    英雄 30 点血，法力水晶每回合上限 +1 并回满，上限 10
    套牌 30 张（同名最多 2 张），先手 3 张起手、后手 4 张 + 一张幸运币
    场上最多 7 个随从，手牌上限 10 张（超出的牌抽出来就烧掉）
    随从出场当回合不能攻击（召唤失调），之后每回合能攻击一次
    攻击是双向伤害：攻击方和被攻击方同时结算，血量 <= 0 的随从死亡
    牌堆抽空后每次抽牌吃 1、2、3... 点递增的疲劳伤害
    先把对方英雄打到 0 血的一方获胜；同时归零算平局

一个回合的流程：法力上限 +1 并回满 → 自己场上的随从解除召唤失调、恢复攻击次数 →
抽一张牌（牌堆空了就吃疲劳伤害）→ 玩家自由地出牌和攻击，直到主动结束回合。

和炉石不一样的地方，只有三处：

    没有起手调度（换牌）
    没有职业、英雄技能、法术（除幸运币）和武器
    随从站位不影响任何东西，出牌一律接在场上最右边

关键词的实现见 cards.py 的文档，其中扰咒和法术增强在没有法术的前提下不产生效果。
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence

from .cards import (
    CHARGE,
    DIVINE_SHIELD,
    LIFESTEAL,
    POISONOUS,
    REBORN,
    RUSH,
    STEALTH,
    TAUNT,
    THE_COIN,
    WINDFURY,
    CardDef,
    build_decklist,
    hand_to_str,
    shuffled,
)

HERO_HEALTH = 30
MAX_MANA = 10
BOARD_LIMIT = 7
HAND_LIMIT = 10

#: 和炉石一致：先手 3 张，后手 4 张外加一张幸运币。
STARTING_HAND = (3, 4)

#: 幸运币给的临时水晶。和炉石一致，1 点。
COIN_MANA = 1

#: 打脸时 Action.target 用这个值。
HERO = -1

#: 英雄拿武器攻击时 Action.source 用这个值。
HERO_SOURCE = -2

N_PLAYERS = 2

# 动作类型
PLAY = "play"
ATTACK = "attack"
END = "end"


class Action(NamedTuple):
    """一个动作。三种：出牌、攻击、结束回合。

    source  出牌时是手牌下标，攻击时是自己场上随从的下标
    target  攻击时是对方场上随从的下标，HERO(-1) 表示打脸
    """

    kind: str
    source: int = -1
    target: int = -1

    def __str__(self) -> str:
        if self.kind == END:
            return "结束回合"
        if self.kind == PLAY:
            return f"出 #{self.source}"
        who = "英雄" if self.target == HERO else f"随从#{self.target}"
        return f"随从#{self.source} 攻击 {who}"


def play(hand_index: int) -> Action:
    return Action(PLAY, hand_index, HERO)


def attack(attacker: int, target: int) -> Action:
    return Action(ATTACK, attacker, target)


def hero_attack(target: int) -> Action:
    return Action(ATTACK, HERO_SOURCE, target)


END_TURN = Action(END)


@dataclass
class Minion:
    """场上的一个随从。

    关键词里有三个是**会变的状态**，所以单独存一份而不是每次去卡面上查：
    圣盾会被打掉、潜行会因为攻击而失去、复生只能触发一次。
    """

    card: CardDef
    attack: int
    health: int
    max_health: int
    attacks_left: int = 1
    just_played: bool = True        # 本回合出场
    divine_shield: bool = False
    stealth: bool = False
    reborn: bool = False
    uid: int = 0

    @staticmethod
    def max_attacks(card: CardDef) -> int:
        return 2 if card.has(WINDFURY) else 1

    @classmethod
    def summon(cls, card: CardDef, uid: int = 0) -> "Minion":
        return cls(
            card=card,
            attack=card.attack,
            health=card.health,
            max_health=card.health,
            attacks_left=cls.max_attacks(card),
            just_played=True,
            divine_shield=card.has(DIVINE_SHIELD),
            stealth=card.has(STEALTH),
            reborn=card.has(REBORN),
            uid=uid,
        )

    def has(self, keyword: str) -> bool:
        return self.card.has(keyword)

    @property
    def name(self) -> str:
        return self.card.name

    @property
    def damaged(self) -> bool:
        return self.health < self.max_health

    @property
    def asleep(self) -> bool:
        """召唤失调：出场当回合动不了，冲锋和突袭除外。"""
        return self.just_played and not self.has(CHARGE) and not self.has(RUSH)

    @property
    def can_attack(self) -> bool:
        return self.attacks_left > 0 and self.attack > 0 and not self.asleep

    @property
    def can_hit_face(self) -> bool:
        """突袭随从出场当回合只能打随从，不能打脸。"""
        return self.can_attack and not (
            self.just_played and self.has(RUSH) and not self.has(CHARGE)
        )

    @property
    def taunting(self) -> bool:
        """潜行状态下的嘲讽不强制对方攻击它。"""
        return self.has(TAUNT) and not self.stealth

    @property
    def value(self) -> int:
        """当前体量：还剩多少攻血。给交易估价用的最粗糙的一个数。"""
        return self.attack + self.health

    def state_words(self) -> List[str]:
        """当前**还生效**的关键词，圣盾破了、潜行掉了就不再列出来。"""
        words = []
        for word in self.card.keywords:
            if word == DIVINE_SHIELD and not self.divine_shield:
                continue
            if word == STEALTH and not self.stealth:
                continue
            if word == REBORN and not self.reborn:
                continue
            words.append(word)
        return words

    def __str__(self) -> str:
        words = self.state_words()
        tail = " " + "".join(words) if words else ""
        mark = "" if self.can_attack else "z"     # z = 这回合动不了
        extra = f"x{self.attacks_left}" if self.attacks_left > 1 else ""
        return f"{self.name} {self.attack}/{self.health}{tail}{mark}{extra}"


@dataclass
class Observation:
    """一个玩家在某个决策点能看到的全部信息。

    看不到的东西一律只给数量：对手的手牌内容、两边牌堆的具体顺序。
    """

    player: int
    turn: int                       # 第几个半回合（每人一次算一个）
    mana: int                       # 自己当前可用的法力
    max_mana: int
    hand: List[CardDef]
    board: List[Minion]             # 自己场上，顺序与 Action.source 对应
    enemy_board: List[Minion]       # 对方场上，顺序与 Action.target 对应
    hero_health: int
    enemy_hero_health: int
    hero_weapon_attack: int         # 0 = 没有武器
    hero_weapon_durability: int
    hero_attacked: bool             # 这回合英雄是否已攻击
    enemy_weapon_attack: int
    enemy_weapon_durability: int
    deck_size: int
    enemy_deck_size: int
    enemy_hand_size: int
    fatigue: int                    # 下次抽空牌堆会吃多少伤害的计数
    enemy_fatigue: int
    going_first: bool               # 是否先手
    legal: List[Action]

    #: **先知字段——违反上面那条"看不到的只给数量"的约定。**
    #:
    #: 对手的真实手牌。只允许非对称 actor-critic 的价值头在**训练时**读取：价值头
    #: 不参与线上决策（推理时算都不算），所以给它作弊不会泄漏到实际打法里。
    #:
    #: 任何进入策略的东西都不许碰它——`features.state_features` /
    #: `features.action_features` 和所有 `Bot` 一律不读，只有
    #: `features.oracle_features` 读。改动时务必守住这条线。
    enemy_hand: List[CardDef] = field(default_factory=list)

    @property
    def opponent(self) -> int:
        return 1 - self.player

    def playable(self) -> List[Action]:
        return [a for a in self.legal if a.kind == PLAY]

    def attacks(self) -> List[Action]:
        return [a for a in self.legal if a.kind == ATTACK]

    @property
    def has_weapon(self) -> bool:
        return self.hero_weapon_attack > 0

    def enemy_taunts(self) -> List[int]:
        """挡在前面的嘲讽随从的下标。"""
        return [i for i, m in enumerate(self.enemy_board) if m.taunting]

    def face_damage(self) -> int:
        """这回合还能打到对方脸上的总伤害（风怒算两次）。"""
        return sum(m.attack * m.attacks_left for m in self.board if m.can_hit_face)

    def has_lethal(self) -> bool:
        """全部打脸能不能直接结束比赛。有嘲讽挡着就不算。"""
        if self.enemy_taunts():
            return False
        return self.face_damage() >= self.enemy_hero_health


@dataclass
class GameResult:
    winner: Optional[int]           # None = 平局
    hero_health: List[int]
    turns: int
    log: List[str] = field(default_factory=list)


class Game:
    """一局简化版炉石的状态机。"""

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        hero_health: int = HERO_HEALTH,
        max_turns: int = 120,
        first: int = 0,
        decklists: Optional[Sequence[Sequence[CardDef]]] = None,
        mirror: bool = True,
    ) -> None:
        """`mirror` 为真时两家用同一份构筑，只是洗牌顺序不同。

        默认镜像是为了评测干净：胜率差异只来自打法，不掺卡组强弱。想研究构筑就
        传 `mirror=False`，或者直接用 `decklists` 指定两份牌。
        """
        self.rng = rng or random.Random()
        self.hero_health_start = hero_health
        self.max_turns = max_turns
        self.first = first
        self.decklists = decklists
        self.mirror = mirror
        self.reset()

    # ------------------------------------------------------------------ 状态

    def reset(self) -> Observation:
        """构筑、洗牌、发起手牌，返回先手玩家的观测。"""
        if self.decklists is not None:
            lists = [list(self.decklists[i]) for i in range(N_PLAYERS)]
        elif self.mirror:
            shared = build_decklist(self.rng)
            lists = [list(shared) for _ in range(N_PLAYERS)]
        else:
            lists = [build_decklist(self.rng) for _ in range(N_PLAYERS)]

        self.decks: List[List[CardDef]] = [shuffled(deck, self.rng) for deck in lists]
        self.hands: List[List[CardDef]] = [[] for _ in range(N_PLAYERS)]
        self.boards: List[List[Minion]] = [[] for _ in range(N_PLAYERS)]
        self.hero_health: List[int] = [self.hero_health_start] * N_PLAYERS
        self.mana: List[int] = [0, 0]
        self.max_mana: List[int] = [0, 0]
        self.fatigue: List[int] = [0, 0]
        self.burned: List[List[CardDef]] = [[], []]
        self.weapons: List[Optional[CardDef]] = [None, None]
        self.weapon_durability: List[int] = [0, 0]
        self.hero_attacked: List[bool] = [False, False]
        self.turns = 0
        self.finished = False
        self.winner: Optional[int] = None
        self._next_uid = 0

        for player in range(N_PLAYERS):
            going_first = player == self.first
            for _ in range(STARTING_HAND[0] if going_first else STARTING_HAND[1]):
                self._draw(player)
            if not going_first:
                self.hands[player].append(THE_COIN)      # 后手的补偿

        self.current = self.first
        self._begin_turn()
        return self.observe()

    def clone(self, rng: Optional[random.Random] = None) -> "Game":
        """复制一份可以随便乱走的局面，给搜索用。

        比 `copy.deepcopy` 快约 200 倍（实测 1µs vs 191µs）：`CardDef` 是不可变的
        NamedTuple，可以直接共享；只有 `Minion` 和几个 list 需要真的复制。

        `rng` **不共享**——否则搜索会消耗掉真实对局的随机数，把真实牌序也搅乱。
        默认给一个固定种子的新实例，让搜索可复现；代价是奥术飞弹和绝命乱斗这两张
        带随机的卡在搜索里只按一种结果评估。
        """
        twin = object.__new__(Game)
        twin.__dict__.update(self.__dict__)      # 先搬常量和标量
        twin.rng = rng if rng is not None else random.Random(0)
        # 会变的部分逐个复制
        twin.decks = [list(deck) for deck in self.decks]
        twin.hands = [list(hand) for hand in self.hands]
        twin.boards = [[copy.copy(m) for m in board] for board in self.boards]
        twin.burned = [list(b) for b in self.burned]
        twin.hero_health = list(self.hero_health)
        twin.mana = list(self.mana)
        twin.max_mana = list(self.max_mana)
        twin.fatigue = list(self.fatigue)
        twin.weapons = list(self.weapons)
        twin.weapon_durability = list(self.weapon_durability)
        twin.hero_attacked = list(self.hero_attacked)
        return twin

    def observe(self, player: Optional[int] = None) -> Observation:
        player = self.current if player is None else player
        enemy = 1 - player
        my_weapon = self.weapons[player]
        en_weapon = self.weapons[enemy]
        return Observation(
            player=player,
            turn=self.turns,
            mana=self.mana[player],
            max_mana=self.max_mana[player],
            hand=list(self.hands[player]),
            board=list(self.boards[player]),
            enemy_board=list(self.boards[enemy]),
            hero_health=self.hero_health[player],
            enemy_hero_health=self.hero_health[enemy],
            hero_weapon_attack=my_weapon.attack if my_weapon else 0,
            hero_weapon_durability=self.weapon_durability[player],
            hero_attacked=self.hero_attacked[player],
            enemy_weapon_attack=en_weapon.attack if en_weapon else 0,
            enemy_weapon_durability=self.weapon_durability[enemy],
            deck_size=len(self.decks[player]),
            enemy_deck_size=len(self.decks[enemy]),
            enemy_hand_size=len(self.hands[enemy]),
            fatigue=self.fatigue[player],
            enemy_fatigue=self.fatigue[enemy],
            going_first=(player == self.first),
            legal=self.legal_actions(player),
            enemy_hand=list(self.hands[enemy]),      # 先知字段，见 Observation 的说明
        )

    def legal_actions(self, player: Optional[int] = None) -> List[Action]:
        """当前玩家的合法动作。结束回合永远可选，所以这个列表不会是空的。"""
        player = self.current if player is None else player
        moves: List[Action] = []

        # 出牌：付得起；手里同名卡只留一个（伤害法术除外，每目标各一个动作）
        board_full = len(self.boards[player]) >= BOARD_LIMIT
        seen = set()
        enemy_board = self.boards[1 - player]
        for i, card in enumerate(self.hands[player]):
            if card.cost > self.mana[player]:
                continue
            # 随从和武器
            if not card.spell:
                if card.name in seen:
                    continue
                if not card.weapon and board_full:
                    continue  # 随从需要场地
                seen.add(card.name)
                moves.append(play(i))
            # 伤害法术 / 变形术 / 横扫：需要指定目标（敌方随从 + 英雄），无视嘲讽和潜行
            elif card.spell_damage > 0 or card.spell_transform:
                if card.name in seen:
                    continue
                seen.add(card.name)
                for j in range(len(enemy_board)):
                    moves.append(Action(PLAY, i, j))
                moves.append(Action(PLAY, i, HERO))
            # 抽牌 / 飞弹 / AoE / 乱斗 / 扭曲虚空：无需指定目标
            elif card.spell_draw > 0 or card.spell_missiles > 0 or card.spell_aoe_enemy_minions > 0 or card.spell_aoe_all_enemies > 0 or card.spell_aoe_all > 0 or card.spell_destroy_all or card.spell_brawl:
                if card.name in seen:
                    continue
                seen.add(card.name)
                moves.append(play(i))
            # 幸运币等
            else:
                if card.name in seen:
                    continue
                seen.add(card.name)
                moves.append(play(i))

        # 攻击：潜行的随从不能被指定；有嘲讽挡着就只能打嘲讽
        enemy_board = self.boards[1 - player]
        targets = [i for i, m in enumerate(enemy_board) if not m.stealth]
        taunts = [i for i in targets if enemy_board[i].taunting]
        if taunts:
            targets = taunts
        face_open = not taunts

        for i, minion in enumerate(self.boards[player]):
            if not minion.can_attack:
                continue
            if face_open and minion.can_hit_face:
                moves.append(attack(i, HERO))
            for j in targets:
                moves.append(attack(i, j))

        # 英雄武器攻击：一回合最多一次
        weapon = self.weapons[player]
        if weapon is not None and weapon.attack > 0 and not self.hero_attacked[player]:
            if face_open:
                moves.append(hero_attack(HERO))
            for j in targets:
                moves.append(hero_attack(j))

        moves.append(END_TURN)
        return moves

    # ------------------------------------------------------------------ 推进

    def step(self, action: Action) -> None:
        """执行一个动作。除了"结束回合"，执行完还是同一个人行动。"""
        if self.finished:
            raise RuntimeError("这一局已经结束了")

        if action.kind == PLAY:
            self._play_card(action.source, action.target)
        elif action.kind == ATTACK:
            self._attack(action.source, action.target)
        elif action.kind == END:
            self._end_turn()
        else:
            raise ValueError(f"未知的动作类型 {action.kind!r}")

    def _play_card(self, hand_index: int, target: int = HERO) -> None:
        player = self.current
        hand = self.hands[player]
        if not 0 <= hand_index < len(hand):
            raise ValueError(f"手牌里没有第 {hand_index} 张")
        card = hand[hand_index]
        if card.cost > self.mana[player]:
            raise ValueError(f"法力不够：{card} 要 {card.cost}，只剩 {self.mana[player]}")
        if card.weapon:
            pass  # 武器不占随从位
        elif not card.spell and len(self.boards[player]) >= BOARD_LIMIT:
            raise ValueError(f"场上已经有 {BOARD_LIMIT} 个随从了")

        hand.pop(hand_index)
        self.mana[player] -= card.cost
        if card.spell:
            self._cast(player, card, target)
        elif card.weapon:
            self.weapons[player] = card
            self.weapon_durability[player] = card.durability
        else:
            self.boards[player].append(Minion.summon(card, self._take_uid()))

    def _cast(self, player: int, card: CardDef, target: int = HERO) -> None:
        if card.name == THE_COIN.name:
            self.mana[player] += COIN_MANA
            return
        if card.spell_draw > 0:
            for _ in range(card.spell_draw):
                self._draw(player)
        if card.spell_destroy_all:
            self.boards[0] = []
            self.boards[1] = []
        if card.spell_brawl:
            all_minions = [(p, j, m) for p in range(N_PLAYERS)
                           for j, m in enumerate(self.boards[p])]
            if all_minions:
                survivor = self.rng.choice(all_minions)
                for p in range(N_PLAYERS):
                    self.boards[p] = [m for m in self.boards[p] if m.uid == survivor[2].uid]
        if card.spell_transform:
            enemy_board = self.boards[1 - player]
            if 0 <= target < len(enemy_board):
                sheep = CardDef("绵羊", 1, 1, 1)
                transformed = Minion.summon(sheep, self._take_uid())
                transformed.just_played = enemy_board[target].just_played
                transformed.attacks_left = enemy_board[target].attacks_left
                enemy_board[target] = transformed
        if card.spell_aoe_enemy_minions > 0:
            dmg = card.spell_aoe_enemy_minions
            for m in self.boards[1 - player]:
                self._hit(m, dmg)
        if card.spell_aoe_all_enemies > 0:
            dmg = card.spell_aoe_all_enemies
            for m in self.boards[1 - player]:
                self._hit(m, dmg)
            if dmg > 0:
                self._damage_hero(1 - player, dmg)
        if card.spell_aoe_all > 0:
            dmg = card.spell_aoe_all
            for p in range(N_PLAYERS):
                for m in self.boards[p]:
                    self._hit(m, dmg)
            self._damage_hero(0, dmg)
            self._damage_hero(1, dmg)
        if card.spell_damage > 0:
            if target == HERO:
                self._damage_hero(1 - player, card.spell_damage)
            else:
                enemy_board = self.boards[1 - player]
                if not 0 <= target < len(enemy_board):
                    raise ValueError(f"对方场上没有第 {target} 个随从")
                self._hit(enemy_board[target], card.spell_damage)
        if card.spell_splash > 0:
            for j, m in enumerate(self.boards[1 - player]):
                if j != target:
                    self._hit(m, card.spell_splash)
            if target != HERO:
                self._damage_hero(1 - player, card.spell_splash)
        if card.spell_missiles > 0:
            for _ in range(card.spell_missiles):
                candidates: List[int] = []
                for j in range(len(self.boards[1 - player])):
                    candidates.append(j)
                if self.hero_health[1 - player] > 0:
                    candidates.append(HERO)
                if not candidates:
                    break
                t = self.rng.choice(candidates)
                if t == HERO:
                    self._damage_hero(1 - player, 1)
                else:
                    self._hit(self.boards[1 - player][t], 1)
        self._clear_dead()
        self._check_over()

    def _attack(self, attacker_index: int, target_index: int) -> None:
        player = self.current

        if attacker_index == HERO_SOURCE:
            self._hero_weapon_attack(player, target_index)
            return

        board = self.boards[player]
        enemy_board = self.boards[1 - player]

        if not 0 <= attacker_index < len(board):
            raise ValueError(f"场上没有第 {attacker_index} 个随从")
        attacker = board[attacker_index]
        if not attacker.can_attack:
            raise ValueError(f"{attacker.name} 这回合不能攻击")

        if target_index == HERO:
            if not attacker.can_hit_face:
                raise ValueError(f"{attacker.name} 出场当回合不能打脸（突袭）")
            if any(m.taunting for m in enemy_board):
                raise ValueError("对面有嘲讽随从挡着")
        else:
            if not 0 <= target_index < len(enemy_board):
                raise ValueError(f"对方场上没有第 {target_index} 个随从")
            defender = enemy_board[target_index]
            if defender.stealth:
                raise ValueError(f"{defender.name} 处于潜行状态，不能被攻击")
            if defender.taunting is False and any(m.taunting for m in enemy_board):
                raise ValueError("对面有嘲讽随从挡着")

        attacker.attacks_left -= 1
        attacker.stealth = False        # 发动攻击就会脱离潜行

        if target_index == HERO:
            self._damage_hero(1 - player, attacker.attack)
            self._drain(player, attacker, attacker.attack)
        else:
            defender = enemy_board[target_index]
            # 双向同时结算：两边的伤害都按打之前的攻击力算
            to_defender, to_attacker = attacker.attack, defender.attack
            dealt_out = self._hit(defender, to_defender)
            dealt_in = self._hit(attacker, to_attacker)

            if dealt_out and attacker.has(POISONOUS):
                defender.health = 0     # 剧毒：只要伤害进去了就直接死
            if dealt_in and defender.has(POISONOUS):
                attacker.health = 0

            self._drain(player, attacker, dealt_out)
            self._drain(1 - player, defender, dealt_in)
            self._clear_dead()

        self._check_over()

    def _hero_weapon_attack(self, player: int, target_index: int) -> None:
        """英雄用武器攻击。"""
        weapon = self.weapons[player]
        if weapon is None:
            raise ValueError("没有装备武器")
        if self.hero_attacked[player]:
            raise ValueError("英雄这回合已经攻击过了")

        enemy_board = self.boards[1 - player]
        if target_index == HERO:
            if any(m.taunting for m in enemy_board):
                raise ValueError("对面有嘲讽随从挡着")
            self._damage_hero(1 - player, weapon.attack)
        else:
            if not 0 <= target_index < len(enemy_board):
                raise ValueError(f"对方场上没有第 {target_index} 个随从")
            defender = enemy_board[target_index]
            if defender.stealth:
                raise ValueError(f"{defender.name} 处于潜行状态，不能被攻击")
            if not defender.taunting and any(m.taunting for m in enemy_board):
                raise ValueError("对面有嘲讽随从挡着")
            # 英雄受到对方随从的反伤
            self._damage_hero(player, defender.attack)
            # 武器伤害打过去
            dealt = self._hit(defender, weapon.attack)
            if dealt and weapon.has("剧毒"):
                defender.health = 0
            self._drain(player, weapon, dealt)
            self._clear_dead()

        self.weapon_durability[player] -= 1
        if self.weapon_durability[player] <= 0:
            self.weapons[player] = None
        self.hero_attacked[player] = True
        self._check_over()

    def _end_turn(self) -> None:
        self.turns += 1
        if self.turns >= self.max_turns:        # 兜底，正常打不到
            self._finish_by_health()
            return
        self.current = 1 - self.current
        self._begin_turn()

    def _begin_turn(self) -> None:
        player = self.current
        self.max_mana[player] = min(self.max_mana[player] + 1, MAX_MANA)
        self.mana[player] = self.max_mana[player]
        self.hero_attacked[player] = False
        for minion in self.boards[player]:
            minion.just_played = False          # 召唤失调解除
            minion.attacks_left = Minion.max_attacks(minion.card)
        self._draw(player)
        self._check_over()

    # ------------------------------------------------------------------ 结算

    def _take_uid(self) -> int:
        self._next_uid += 1
        return self._next_uid

    def _hit(self, minion: Minion, amount: int) -> int:
        """给随从造成伤害，返回**真正打进去**的伤害。

        圣盾把这一次伤害整个吃掉并消失，返回 0——剧毒和吸血都是按这个返回值判定的，
        所以"圣盾挡下剧毒"和"打在圣盾上不回血"自然就对了。
        """
        if amount <= 0:
            return 0
        if minion.divine_shield:
            minion.divine_shield = False
            return 0
        minion.health -= amount
        return amount

    def _drain(self, player: int, source, dealt: int) -> None:
        """吸血：造成多少伤害，自己的英雄回多少血，不超过上限。"""
        if dealt <= 0 or not source.has(LIFESTEAL):
            return
        self.hero_health[player] = min(self.hero_health[player] + dealt, self.hero_health_start)

    def _draw(self, player: int) -> Optional[CardDef]:
        """抽一张。牌堆空了吃疲劳伤害，手牌满了把抽到的牌烧掉。"""
        deck = self.decks[player]
        if not deck:
            self.fatigue[player] += 1
            self._damage_hero(player, self.fatigue[player])
            return None

        card = deck.pop()
        if len(self.hands[player]) >= HAND_LIMIT:
            self.burned[player].append(card)
            return None
        self.hands[player].append(card)
        return card

    def _damage_hero(self, player: int, amount: int) -> None:
        self.hero_health[player] -= amount

    def _clear_dead(self) -> None:
        """清场。带复生的随从第一次死亡后带 1 点血回来，且不再有复生。

        复生有场地限制：如果死亡时棋盘是满的，复生不会触发（没有空间）。
        这和炉石一致——死亡结算时，死掉的随从还没有真正离开棋盘。
        """
        for player in range(N_PLAYERS):
            board = self.boards[player]
            n_before = len(board)
            survivors: List[Minion] = []
            for minion in board:
                if minion.health > 0:
                    survivors.append(minion)
                elif minion.reborn and n_before < BOARD_LIMIT and len(survivors) < BOARD_LIMIT:
                    back = Minion.summon(minion.card, self._take_uid())
                    back.health = 1
                    back.reborn = False
                    survivors.append(back)
            self.boards[player] = survivors

    def _check_over(self) -> None:
        dead = [p for p in range(N_PLAYERS) if self.hero_health[p] <= 0]
        if not dead:
            return
        self.finished = True
        self.winner = None if len(dead) == N_PLAYERS else 1 - dead[0]

    def _finish_by_health(self) -> None:
        self.finished = True
        if self.hero_health[0] == self.hero_health[1]:
            self.winner = None
        else:
            self.winner = max(range(N_PLAYERS), key=lambda p: self.hero_health[p])

    def result(self) -> GameResult:
        if not self.finished:
            raise RuntimeError("这一局还没结束")
        return GameResult(
            winner=self.winner,
            hero_health=list(self.hero_health),
            turns=self.turns,
        )


def play_game(
    players: Sequence,
    rng: Optional[random.Random] = None,
    first: int = 0,
    verbose: bool = False,
    **kwargs,
) -> GameResult:
    """让 `players`（实现了 choose(obs) 的对象）打完一局。"""
    game = Game(rng=rng, first=first, **kwargs)
    log: List[str] = []

    # 搜索型选手需要真实局面才能克隆推演；只看观测的选手没有这个方法，跳过即可
    for seat, player in enumerate(players):
        bind = getattr(player, "bind_game", None)
        if bind is not None:
            bind(game, seat)

    if verbose:
        for i in range(N_PLAYERS):
            log.append(f"玩家{i} 起手: {hand_to_str(game.hands[i])}")

    last_turn = -1
    while not game.finished:
        obs = game.observe()
        if verbose and obs.turn != last_turn:
            last_turn = obs.turn
            log.append(
                f"-- 第{obs.turn + 1}个回合 玩家{obs.player} "
                f"{obs.mana}/{obs.max_mana}水晶 血{obs.hero_health}:{obs.enemy_hero_health}"
            )
        action = players[obs.player].choose(obs)
        if action not in obs.legal:
            raise ValueError(f"玩家{obs.player} 给出了非法动作 {action}")
        if verbose:
            log.append(f"   玩家{obs.player} {describe(obs, action)}")
        game.step(action)

    result = game.result()
    result.log = log
    if verbose:
        who = "平局" if result.winner is None else f"玩家{result.winner} 获胜"
        log.append(f"{who}，血量 {result.hero_health}，共 {result.turns} 个回合")
    return result


def describe(obs: Observation, action: Action) -> str:
    """把动作翻译成带卡名的一句话，供日志和界面使用。"""
    if action.kind == END:
        return "结束回合"
    if action.kind == PLAY:
        card = obs.hand[action.source]
        if card.spell and (card.spell_damage > 0 or card.spell_transform):
            who = "英雄" if action.target == HERO else f"随从#{action.target}"
            return f"{card.name} → {who}"
        return f"{'用' if card.spell else '出'} {card}"
    if action.source == HERO_SOURCE:
        weapon_atk = obs.hero_weapon_attack
        if action.target == HERO:
            return f"英雄({weapon_atk}) 打脸"
        defender = obs.enemy_board[action.target]
        return f"英雄({weapon_atk}) 攻击 {defender.name}({defender.attack}/{defender.health})"
    attacker = obs.board[action.source]
    if action.target == HERO:
        return f"{attacker.name}({attacker.attack}) 打脸"
    defender = obs.enemy_board[action.target]
    return (
        f"{attacker.name}({attacker.attack}/{attacker.health}) "
        f"攻击 {defender.name}({defender.attack}/{defender.health})"
    )
