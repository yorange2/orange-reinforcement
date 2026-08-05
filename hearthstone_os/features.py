"""把 (局面, 某个候选动作) 编码成定长向量（v7+，223 维）。

v7 是把 `hearthstone/features.py` 的 v5/v6 思路（48→76→197→251 的演进）
移植到 orange-stone 结构化视图上的版本（199 维）；**v7+（M5）在引擎补上
卡面文本视图字段后把 A_TEXT / S_TEXT 块加了回来，223 维**：

- **关键词**：视图暴露 5 个（嘲讽/圣盾/潜行/风怒/冲锋）+ 冻结/能动/可出；
  M5 补了扰咒（elusive，orange-stone #73）
- **卡面文本**（M5）：`A_TEXT` 16 维 = 标签（战吼/亡语/光环/触发）+ 战吼/
  法术量级（damage/draw/summon/buff/heal/freeze/destroy）+ 亡语量级 + 光环
  量级；`S_TEXT` 8 维 = 双方场上带文本随从数。战吼/法术共用 battlecry 槽。
- **无先知特征**（v6 的 oracle）：orange-stone 绑定层不暴露对手手牌
  （`opponent.hand` 恒为空），非对称价值头无从谈起。
- **无英雄技能/职业**：当前引擎英雄没有英雄技能（M2 核实）。
- **无疲劳**：视图没有疲劳字段。

特征布局沿用 v6 的三段式：动作特征（类型 + 出牌/攻击细节 + 交易结果）+
共享局面尾（聚合 + 逐随从槽 + 逐手牌 + 斩杀检测 + 杂项）。交易结果的
"这一刀下去谁死、场面差怎么变"是 v2 起积累的核心信号，逐槽编码从 v3 起
就是局面特征的主体，都保留。

`going_first` 不在视图里，由调用方显式传入（`play_game` 通过 `bind_env`
把当前行动方喂给智能体）：当前行动方是 P1 时 1.0，否则 0.0。
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .env import Action

#: orange-stone 结构化视图暴露的关键词，顺序固定（one-hot 下标即字段序）。
OS_KEYWORDS = ("taunt", "divine_shield", "stealth", "windfury", "charge")

# ---------------------------------------------------------------- 维度布局

# 动作特征
A_TYPE = 3                     # 动作类型 one-hot (play/attack/end)
A_CARD = 1 + 2 + 5             # 出牌：费用 + 攻/血 + 关键词 one-hot（视图的 5 个）
#: 卡面文本（v7+，M5）：标签（战吼/亡语/光环/触发）+ 战吼/法术量级 + 亡语量级
#: + 光环量级。v6 的 A_TEXT 思路重建在结构化视图的 effect 字段上（M5）。
A_TEXT = 4 + 7 + 3 + 2
A_ATTACKER = 2 + 1 + 1 + 1 + 1  # 攻击者：攻/血 + 剩余次数 + 圣盾/风怒/冲锋
A_TARGET = 1 + 2 + 1 + 1 + 1    # 目标：人脸/攻/血 + 嘲讽/圣盾/潜行
A_TRADE = 2 + 1 + 1            # 交易结果：谁死 + 过杀 + 场面差
A_PLAY_EFFECT = 2 + 1           # 出牌效果：剩余水晶 + 用光/铺满
A_LEGAL = 1                    # 候选动作数

ACTION_DIM = (A_TYPE + A_CARD + A_TEXT + A_ATTACKER + A_TARGET + A_TRADE
              + A_PLAY_EFFECT + A_LEGAL)

# 局面特征
S_BASE = 4                     # 水晶（当前/上限）+ 双方血量
S_WEAPON = 4                   # 武器：自己攻/耐久 + 对方攻/耐久
S_HAND = 1 + 1 + 2 + 1 + 1     # 手牌：张数 + 可出数 + 可出总攻/总血 + 有冲锋/有嘲讽
S_BOARD = 2 + 2 + 2 + 1        # 场面：随从数 + 总攻 + 总血 + 嘲讽挡脸
S_BOARD_SLOTS = 7 * 9 + 7 * 9  # 双方场上逐随从（各 7 槽）：攻/血/能动/嘲/盾/潜/风/冲/冻
S_HAND_CARDS = 5 * 3           # 手牌逐卡（前 3 低费可出）：费/攻/血/冲锋/潜行
#: 场上文本感知（v7+）：双方各有多少带战吼/亡语/光环/触发的随从（M5）
S_TEXT = 4 + 4
S_LETHAL = 2                   # 斩杀检测
S_OTHER = 4                    # 牌堆/对手手牌/先后手/bias

STATE_DIM = (S_BASE + S_WEAPON + S_HAND + S_BOARD + S_BOARD_SLOTS
             + S_HAND_CARDS + S_TEXT + S_LETHAL + S_OTHER)

STATE_OFFSET = ACTION_DIM
FEATURE_DIM = ACTION_DIM + STATE_DIM


# ---------------------------------------------------------------- 对外接口

def batch_features(obs, actions: Sequence[Action], going_first: float = 0.5) -> np.ndarray:
    """一个决策点的特征矩阵：(候选数, FEATURE_DIM)，局面尾共享。"""
    tail = state_features(obs, going_first)
    rows = np.empty((len(actions), FEATURE_DIM), dtype=np.float32)
    for i, action in enumerate(actions):
        rows[i, :STATE_OFFSET] = action_features(obs, action, actions)
        rows[i, STATE_OFFSET:] = tail
    return rows


def action_features(obs, action: Action, actions: Sequence[Action]) -> List[float]:
    feats = _base_features(obs, action)
    # 公共：候选动作数
    feats.append(min(len(actions), 30) / 30.0)
    assert len(feats) == ACTION_DIM, f"{len(feats)} != {ACTION_DIM}"
    return feats


def state_features(obs, going_first: float = 0.5) -> List[float]:
    feats = _state_features(obs, going_first)
    assert len(feats) == STATE_DIM, f"{len(feats)} != {STATE_DIM}"
    return feats


# ---------------------------------------------------------------- 动作

def _card_text(card) -> List[float]:
    """卡面文本的 A_TEXT 维编码（v7+，M5）：标签 + 战吼/法术量级 + 亡语量级 + 光环量级。

    战吼/法术共用 battlecry 槽，所以 bc_* 对两类都成立。白板随从整块为 0。
    """
    return [
        # 标签
        1.0 if card.has_battlecry else 0.0,
        1.0 if card.has_deathrattle else 0.0,
        1.0 if card.has_aura else 0.0,
        1.0 if card.has_trigger else 0.0,
        # 战吼/法术量级
        card.bc_damage / 10.0,
        card.bc_draw / 3.0,
        card.bc_summon / 3.0,
        card.bc_buff / 6.0,
        card.bc_heal / 10.0,
        1.0 if card.bc_freeze else 0.0,
        1.0 if card.bc_destroy else 0.0,
        # 亡语量级
        card.dr_damage / 10.0,
        card.dr_draw / 3.0,
        card.dr_summon / 3.0,
        # 光环量级
        card.aura_attack / 3.0,
        card.aura_health / 3.0,
    ]


#: 攻击和结束回合没有"卡面文本"可言，整块置零。
_NO_TEXT = [0.0] * A_TEXT


def _base_features(obs, action: Action) -> List[float]:
    if action.kind == "play":
        return _play(obs, action)
    if action.kind == "attack":
        return _attack(obs, action)
    return _end()


def _play(obs, action: Action) -> List[float]:
    card = obs.me.hand[action.card_index]
    left = obs.me.remaining_mana - card.cost
    target = _find_target(action, obs)
    target_is_hero = target is None
    target_minion = None if target_is_hero else target

    if target_minion is not None:
        def_atk, def_hp = target_minion.attack / 10.0, target_minion.health / 10.0
        def_taunt = 1.0 if target_minion.taunt else 0.0
        def_shield = 1.0 if target_minion.divine_shield else 0.0
        def_stealth = 1.0 if target_minion.stealth else 0.0
        kills_def = 1.0  # 有目标的出牌（法术/战吼）按"能打死"处理
    else:
        def_atk = def_hp = 0.0
        def_taunt = def_shield = def_stealth = 0.0
        kills_def = 0.0

    feats = [
        # 类型
        1.0, 0.0, 0.0,
        # 卡牌
        card.cost / 10.0,
        card.attack / 10.0,
        card.health / 10.0,
        *_kw_vec(card),
        # 卡面文本
        *_card_text(card),
        # 攻击者（出牌不涉及）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # 目标
        1.0 if target_is_hero else 0.0,
        def_atk,
        def_hp,
        def_taunt,
        def_shield,
        def_stealth,
        # 交易结果
        kills_def, 0.0, 0.0, 0.0,
        # 出牌效果
        left / 10.0,
        1.0 if card.cost == obs.me.remaining_mana and card.cost > 0 else 0.0,
        1.0 if len(obs.me.field) >= 6 else 0.0,
    ]
    return feats


def _attack(obs, action: Action) -> List[float]:
    attacker = _find_attacker(action, obs)
    target = _find_target(action, obs)
    target_is_hero = target is None
    is_hero_attack = attacker is None

    att_atk = obs.me.hero_attack if is_hero_attack else attacker.attack
    att_hp = 30.0 if is_hero_attack else float(attacker.health)
    att_shield = 0.0 if is_hero_attack else (1.0 if attacker.divine_shield else 0.0)
    att_windfury = 0.0 if is_hero_attack else (1.0 if attacker.windfury else 0.0)
    att_charge = 0.0 if is_hero_attack else (1.0 if attacker.charge else 0.0)

    if target_is_hero:
        def_atk = def_hp = 0.0
        def_taunt = def_shield = def_stealth = 0.0
        kills_def = kills_att = 0.0
        overkill = 0.0
        board_after = (_board_attack(obs.me.field) - _board_attack(obs.opponent.field)) / 20.0
    else:
        def_atk = target.attack / 10.0
        def_hp = target.health / 10.0
        def_taunt = 1.0 if target.taunt else 0.0
        def_shield = 1.0 if target.divine_shield else 0.0
        def_stealth = 1.0 if target.stealth else 0.0

        # 交易结果（等效血量 = 血 + 圣盾）
        def_eff_hp = target.health + (1 if target.divine_shield else 0)
        att_eff_hp = att_hp + (1 if att_shield else 0)
        kills_def = 1.0 if att_atk >= def_eff_hp else 0.0
        kills_att = 1.0 if target.attack >= att_eff_hp and not is_hero_attack else 0.0
        overkill = max(0.0, att_atk - def_eff_hp) / 5.0

        # 交易后场面差（把参与交易的两个随从从场面里去掉）
        my_rest = _board_attack(obs.me.field) - attacker.attack if not is_hero_attack else _board_attack(obs.me.field)
        en_rest = _board_attack(obs.opponent.field) - target.attack
        if not kills_def:
            en_rest += target.attack
        if not kills_att and not is_hero_attack:
            my_rest += attacker.attack
        board_after = (my_rest - en_rest) / 20.0

    feats = [
        # 类型
        0.0, 1.0, 0.0,
        # 卡牌（攻击时不涉及手牌）
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        *_NO_TEXT,
        # 攻击者
        att_atk / 10.0,
        att_hp / 10.0,
        float(_attacks_left(action, obs) - 1),
        att_shield,
        att_windfury,
        att_charge,
        # 目标
        1.0 if target_is_hero else 0.0,
        def_atk,
        def_hp,
        def_taunt,
        def_shield,
        def_stealth,
        # 交易结果
        kills_def,
        kills_att,
        overkill,
        board_after,
        # 出牌效果（攻击时不涉及）
        0.0, 0.0, 0.0,
    ]
    return feats


def _end() -> List[float]:
    feats = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        *_NO_TEXT,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    return feats


# ---------------------------------------------------------------- 局面

def _board_slot_feature(minion) -> List[float]:
    """单个随从的 9 维编码：攻/血/能动/嘲讽/圣盾/潜行/风怒/冲锋/冻结。"""
    return [
        minion.attack / 10.0,
        minion.health / 10.0,
        1.0 if minion.can_attack else 0.0,
        1.0 if minion.taunt else 0.0,
        1.0 if minion.divine_shield else 0.0,
        1.0 if minion.stealth else 0.0,
        1.0 if minion.windfury else 0.0,
        1.0 if minion.charge else 0.0,
        1.0 if minion.frozen else 0.0,
    ]


def _board_slots(field: Sequence, n: int = 7) -> List[float]:
    """场上按出现顺序的前 n 个随从的编码，不足补零。"""
    feats: List[float] = []
    for i in range(n):
        if i < len(field):
            feats.extend(_board_slot_feature(field[i]))
        else:
            feats.extend([0.0] * 9)
    return feats


def _hand_card_feature(card) -> List[float]:
    """手牌中单张可出牌的 5 维编码：费/攻/血/冲锋/潜行。"""
    return [
        card.cost / 10.0,
        card.attack / 10.0,
        card.health / 10.0,
        1.0 if card.charge else 0.0,
        1.0 if card.stealth else 0.0,
    ]


def _hand_cards(obs, n: int = 3) -> List[float]:
    """可出牌中费用最低的 n 张的编码，不足补零。"""
    playable = sorted(
        (c for c in obs.me.hand if c.playable),
        key=lambda c: (c.cost, -(c.attack + c.health)),
    )
    feats: List[float] = []
    for i in range(n):
        if i < len(playable):
            feats.extend(_hand_card_feature(playable[i]))
        else:
            feats.extend([0.0] * 5)
    return feats


def _state_features(obs, going_first: float) -> List[float]:
    my = obs.me.field
    en = obs.opponent.field
    playable = [c for c in obs.me.hand if c.playable]
    play_atk = sum(c.attack for c in playable)
    play_hp = sum(c.health for c in playable)
    has_charge = 1.0 if any(c.charge for c in playable) else 0.0
    has_taunt = 1.0 if any(c.taunt for c in playable) else 0.0

    enemy_taunt = 1.0 if any(m.taunt for m in en) else 0.0

    return [
        # 水晶血量
        obs.me.remaining_mana / 10.0,
        obs.me.total_mana / 10.0,
        obs.me.hero_health / 30.0,
        obs.opponent.hero_health / 30.0,
        # 武器
        obs.me.weapon_attack / 10.0,
        obs.me.weapon_durability / 5.0,
        obs.opponent.weapon_attack / 10.0,
        obs.opponent.weapon_durability / 5.0,
        # 手牌质量
        len(playable) / 6.0,
        obs.me.hand_count / 10.0,
        play_atk / 20.0,
        play_hp / 30.0,
        has_charge,
        has_taunt,
        # 手牌逐卡（前 3 低费可出）
        *_hand_cards(obs),
        # 场面大小
        len(my) / 7.0,
        len(en) / 7.0,
        sum(m.attack for m in my) / 20.0,
        sum(m.attack for m in en) / 20.0,
        sum(m.health for m in my) / 30.0,
        sum(m.health for m in en) / 30.0,
        enemy_taunt,
        # 双方场上逐随从（各 7 槽）
        *_board_slots(my),
        *_board_slots(en),
        # 场上文本感知（v7+，M5）：带战吼/亡语/光环/触发的随从数
        float(sum(1 for m in my if m.has_battlecry)),
        float(sum(1 for m in my if m.has_deathrattle)),
        float(sum(1 for m in my if m.has_aura)),
        float(sum(1 for m in my if m.has_trigger)),
        float(sum(1 for m in en if m.has_battlecry)),
        float(sum(1 for m in en if m.has_deathrattle)),
        float(sum(1 for m in en if m.has_aura)),
        float(sum(1 for m in en if m.has_trigger)),
        # 斩杀检测
        _can_kill(obs, mine=True),
        _can_kill(obs, mine=False),
        # 其他
        obs.me.deck_count / 30.0,
        obs.opponent.hand_count / 10.0,
        going_first,
        1.0,  # bias
    ]


def _can_kill(obs, mine: bool) -> float:
    """给定一方，所有能动的随从打脸能不能杀死对面。有嘲讽挡着就不算。"""
    if mine:
        board, enemy_health = obs.me.field, obs.opponent.hero_health
        taunts = [m for m in obs.opponent.field if m.taunt]
    else:
        board, enemy_health = obs.opponent.field, obs.me.hero_health
        taunts = [m for m in obs.me.field if m.taunt]
    if taunts:
        return 0.0
    damage = sum(m.attack * _attacks_left_from(m) for m in board if m.can_attack)
    return 1.0 if damage >= enemy_health else 0.0


# ---------------------------------------------------------------- 辅助

def _kw_vec(card) -> List[float]:
    return [
        1.0 if getattr(card, kw) else 0.0
        for kw in OS_KEYWORDS
    ]


def _find_attacker(action: Action, obs):
    """攻击者随从；英雄攻击返回 None。"""
    for minion in obs.me.field:
        if minion.entity_id == action.entity_id:
            return minion
    return None


def _find_target(action: Action, obs):
    """目标随从；打脸返回 None。"""
    for minion in obs.opponent.field:
        if minion.entity_id == action.target_id:
            return minion
    return None


def _attacks_left(action: Action, obs) -> int:
    """攻击者还能打几次（风怒按 2 估）。"""
    minion = _find_attacker(action, obs)
    if minion is None:
        return 1 if obs.me.hero_attack > 0 else 0
    if not minion.can_attack:
        return 0
    return 2 if minion.windfury else 1


def _attacks_left_from(minion) -> int:
    return 2 if minion.windfury else 1


def _board_attack(field: Sequence) -> float:
    return sum(m.attack for m in field)
