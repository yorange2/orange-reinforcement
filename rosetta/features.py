"""把 (Observation, Action) 编码成定长向量。

刻意和 `hearthstone/features.py` 保持相同的总维度（251），这样 `UnifiedNet`
可以不改一行代码直接复用在 rosetta 环境上。但两个模块的特征布局独立——
rosetta 的 Observation/Action 数据模型不同。

布局：前 46 维是动作特征，后 205 维是局面特征。
"""

from __future__ import annotations

import zlib
from typing import List

import numpy as np

from .env import Action, ActionType

# ================================================================ 维度布局

# 动作特征 (0..45)
A_TYPE       = 5     # one-hot PLAY_CARD / ATTACK / HERO_POWER / END_TURN / CHOOSE
A_CARD       = 12    # 打出牌: cost/10, attack/10, health/10 + 9 关键词 bool
A_CARD_ID    = 2     # crc32 哈希分桶，区分同身材不同名的卡
A_CHOOSE     = 1     # CHOOSE: (choice % 128) / 128
A_ATTACKER   = 7     # ATTACK: attack/10, health/10, can_attack, shield, poison, lifesteal, windfury
A_TARGET     = 8     # target_side, is_hero, attack/10, health/10, taunt, shield, stealth, poison
A_TRADE      = 4     # ATTACK/HP: kills_def, kills_att, overkill/5, board_after/20
A_HP         = 2     # HERO_POWER: damage/5, mana_left/10
A_PLAY       = 4     # PLAY_CARD: mana_left/10, used_all_mana, field_full, field_pos/8
A_LEGAL      = 1     # min(len(actions), 30) / 30

ACTION_DIM = (A_TYPE + A_CARD + A_CARD_ID + A_CHOOSE + A_ATTACKER
              + A_TARGET + A_TRADE + A_HP + A_PLAY + A_LEGAL)
STATE_OFFSET = ACTION_DIM

# 局面特征 (46..250)
S_BASE           = 6     # 水晶/血量/护甲
S_WEAPON         = 6     # 武器状态
S_HERO           = 2     # 英雄技能可用
S_HAND           = 7     # 手牌聚合
S_HAND_CARDS     = 25    # 5 张最低费可出牌 × 5 维
S_BOARD          = 7     # 场面聚合
S_BOARD_SLOTS    = 126   # 双方各 7 槽 × 9 维
S_KEYWORDS       = 14    # 双方场上关键词计数
S_LEGAL          = 1     # 候选数
S_LETHAL         = 2     # 斩杀检测
S_TURN           = 2     # 回合/先后手
S_DECK           = 2     # 牌堆
S_HAND_EXTRA     = 2     # playable 总费/最高费
S_AWAITING       = 1     # 等待选择
S_CHOICE_CNT     = 1     # choice 动作数
S_BIAS           = 1     # bias

STATE_DIM = (S_BASE + S_WEAPON + S_HERO + S_HAND + S_HAND_CARDS + S_BOARD
             + S_BOARD_SLOTS + S_KEYWORDS + S_LEGAL + S_LETHAL + S_TURN
             + S_DECK + S_HAND_EXTRA + S_AWAITING + S_CHOICE_CNT + S_BIAS)
FEATURE_DIM = ACTION_DIM + STATE_DIM

# 关键词在 action 和 slot 编码中的固定顺序
_KEYWORD_ORDER = ("taunt", "divine_shield", "stealth", "poisonous",
                  "windfury", "lifesteal", "rush", "charge", "frozen")

# ================================================================ 对外接口


def batch_features(obs, actions: List[Action]) -> np.ndarray:
    """(候选数, FEATURE_DIM) 矩阵。局面特征只算一次，所有行共享。"""
    tail = state_features(obs, actions)
    rows = np.empty((len(actions), FEATURE_DIM), dtype=np.float32)
    for i, action in enumerate(actions):
        rows[i, :STATE_OFFSET] = action_features(obs, action, actions)
        rows[i, STATE_OFFSET:] = tail
    return rows


def action_features(obs, action: Action, actions: List[Action]) -> List[float]:
    feats: List[float] = []

    # --- A_TYPE: 5-dim one-hot ---
    at = action.type
    feats += [
        1.0 if at == ActionType.PLAY_CARD else 0.0,
        1.0 if at == ActionType.ATTACK else 0.0,
        1.0 if at == ActionType.HERO_POWER else 0.0,
        1.0 if at == ActionType.END_TURN else 0.0,
        1.0 if at == ActionType.CHOOSE else 0.0,
    ]

    # --- A_CARD (12 dims) + A_CARD_ID (2 dims) ---
    if at == ActionType.PLAY_CARD:
        card = obs.me.hand[action.hand_idx]
        feats += [
            card.cost / 10.0,
            card.attack / 10.0,
            card.health / 10.0,
            *[1.0 if getattr(card, kw, False) else 0.0
              for kw in _KEYWORD_ORDER],
        ]
        # crc32 hash → two buckets (0..15 each)
        h = zlib.crc32(card.card_id.encode()) & 0xFFFFFFFF
        feats += [(h & 0xF) / 16.0, ((h >> 4) & 0xF) / 16.0]
    else:
        feats += [0.0] * (A_CARD + A_CARD_ID)

    # --- A_CHOOSE (1 dim) ---
    feats.append(((action.choice % 128) / 128.0)
                 if at == ActionType.CHOOSE else 0.0)

    # --- A_ATTACKER (7 dims) ---
    if at == ActionType.ATTACK:
        if action.source_pos < 0:
            feats += [obs.me.hero_attack / 10.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        else:
            src = obs.me.field[action.source_pos]
            feats += [
                src.attack / 10.0, src.health / 10.0,
                1.0 if src.can_attack else 0.0,
                1.0 if src.divine_shield else 0.0,
                1.0 if src.poisonous else 0.0,
                1.0 if src.lifesteal else 0.0,
                1.0 if src.windfury else 0.0,
            ]
    else:
        feats += [0.0] * A_ATTACKER

    # --- A_TARGET (8 dims) ---
    needs_target = at in (ActionType.ATTACK, ActionType.HERO_POWER) or (
        at == ActionType.PLAY_CARD and action.target_side >= 0)
    if needs_target:
        tgt = _resolve_target(obs, action)
        target_is_hero = (action.target_pos < 0)
        feats += [
            float(action.target_side),
            1.0 if target_is_hero else 0.0,
            tgt["attack"] / 10.0 if isinstance(tgt, dict) else (tgt.attack / 10.0 if tgt else 0.0),
            tgt["health"] / 10.0 if isinstance(tgt, dict) else (tgt.health / 10.0 if tgt else 0.0),
            1.0 if (tgt and not isinstance(tgt, dict) and tgt.taunt) else 0.0,
            1.0 if (tgt and not isinstance(tgt, dict) and tgt.divine_shield) else 0.0,
            1.0 if (tgt and not isinstance(tgt, dict) and tgt.stealth) else 0.0,
            1.0 if (tgt and not isinstance(tgt, dict) and tgt.poisonous) else 0.0,
        ]
    else:
        feats += [0.0] * A_TARGET

    # --- A_TRADE (4 dims) ---
    if at in (ActionType.ATTACK, ActionType.HERO_POWER) and action.target_pos >= 0:
        attacker_dmg = _attacker_damage(obs, action)
        target_field = obs.me.field if action.target_side == 0 else obs.opponent.field
        target_pos = action.target_pos
        if target_pos >= len(target_field):
            feats += [0.0] * A_TRADE  # 安全降级：目标已不在场
        else:
            defender = target_field[target_pos]
            def_eff = defender.health + (1 if defender.divine_shield else 0)
            kills_def = 1.0 if attacker_dmg >= def_eff else 0.0
            kills_att = 0.0  # hero power / hero attack don't die from trading
            overkill = max(0.0, attacker_dmg - def_eff) / 5.0

            src = _get_source(obs, action) if at == ActionType.ATTACK else None
            my_rest = sum(m.attack for m in obs.me.field if m is not src)
            en_rest_field = obs.opponent.field if action.target_side == 1 else obs.me.field
            en_rest = sum(m.attack for m in en_rest_field if m is not defender)
            is_enemy_target = (action.target_side == 1)
            if not kills_def and is_enemy_target:
                en_rest += defender.attack
            board_after = (my_rest - en_rest) / 20.0 if is_enemy_target else 0.0

            feats += [kills_def, kills_att, overkill, board_after]
    else:
        feats += [0.0] * A_TRADE

    # --- A_HP (2 dims) ---
    if at == ActionType.HERO_POWER:
        dmg = 1.0 if action.target_side == 1 else 0.0  # 法师火冲 = 1 伤
        mana_after = obs.me.remaining_mana - 2
        feats += [dmg / 5.0, max(0.0, mana_after) / 10.0]
    else:
        feats += [0.0] * A_HP

    # --- A_PLAY (4 dims) ---
    if at == ActionType.PLAY_CARD:
        card = obs.me.hand[action.hand_idx]
        left = obs.me.remaining_mana - card.cost
        feats += [
            max(0.0, left) / 10.0,
            1.0 if card.cost == obs.me.remaining_mana and card.cost > 0 else 0.0,
            1.0 if len(obs.me.field) >= 7 else 0.0,
            (action.field_pos + 1) / 8.0 if action.field_pos >= 0 else 0.125,
        ]
    else:
        feats += [0.0] * A_PLAY

    # --- A_LEGAL (1 dim) ---
    feats.append(min(len(actions), 30) / 30.0)

    assert len(feats) == ACTION_DIM, f"{len(feats)} != {ACTION_DIM}"
    return feats


def state_features(obs, actions: List[Action]) -> List[float]:
    me = obs.me
    opp = obs.opponent

    # ---- S_BASE (6) ----
    feats: List[float] = [
        me.remaining_mana / 10.0,
        me.total_mana / 10.0,
        me.hero_health / 30.0,
        opp.hero_health / 30.0,
        me.hero_armor / 10.0,
        opp.hero_armor / 10.0,
    ]

    # ---- S_WEAPON (6) ----
    feats += [
        me.weapon_attack / 10.0,
        me.weapon_durability / 5.0,
        me.hero_attack / 10.0,
        opp.weapon_attack / 10.0,
        opp.weapon_durability / 5.0,
        opp.hero_attack / 10.0,
    ]

    # ---- S_HERO (2) ----
    feats += [
        1.0 if me.hero_power_usable else 0.0,
        1.0 if opp.hero_power_usable else 0.0,
    ]

    # ---- S_HAND (7) ----
    playable = [c for c in me.hand if c.playable]
    feats += [
        me.hand_count / 10.0,
        len(playable) / 6.0,
        sum(c.attack for c in playable) / 20.0,
        sum(c.health for c in playable) / 30.0,
        1.0 if any(c.charge for c in playable) else 0.0,
        1.0 if any(c.taunt for c in playable) else 0.0,
        1.0 if any(c.rush for c in playable) else 0.0,
    ]

    # ---- S_HAND_CARDS (5 × 5) ----
    sorted_playable = sorted(playable, key=lambda c: (c.cost, -(c.attack + c.health)))
    for i in range(5):
        if i < len(sorted_playable):
            c = sorted_playable[i]
            feats += [c.cost / 10.0, c.attack / 10.0, c.health / 10.0,
                      1.0 if c.charge else 0.0, 1.0 if c.rush else 0.0]
        else:
            feats += [0.0] * 5

    # ---- S_BOARD (7) ----
    my_field = me.field
    en_field = opp.field
    feats += [
        len(my_field) / 7.0,
        len(en_field) / 7.0,
        sum(m.attack for m in my_field) / 20.0,
        sum(m.attack for m in en_field) / 20.0,
        sum(m.health for m in my_field) / 30.0,
        sum(m.health for m in en_field) / 30.0,
        1.0 if any(m.taunt for m in en_field) else 0.0,
    ]

    # ---- S_BOARD_SLOTS (2 × 7 × 9) ----
    for field in (my_field, en_field):
        for i in range(7):
            if i < len(field):
                m = field[i]
                feats += [
                    m.attack / 10.0, m.health / 10.0,
                    1.0 if m.can_attack else 0.0,
                    1.0 if m.taunt else 0.0,
                    1.0 if m.divine_shield else 0.0,
                    1.0 if m.stealth else 0.0,
                    1.0 if m.poisonous else 0.0,
                    1.0 if m.windfury else 0.0,
                    1.0 if m.lifesteal else 0.0,
                ]
            else:
                feats += [0.0] * 9

    # ---- S_KEYWORDS (2 × 7) ----
    for field in (my_field, en_field):
        for kw in ("taunt", "divine_shield", "stealth", "poisonous",
                    "windfury", "lifesteal", "frozen"):
            feats.append(float(sum(1 for m in field if getattr(m, kw, False))))

    # ---- S_LEGAL (1) ----
    feats.append(min(len(actions), 30) / 30.0)

    # ---- S_LETHAL (2) ----
    feats.append(1.0 if _my_lethal(obs) else 0.0)
    feats.append(1.0 if _enemy_lethal(obs) else 0.0)

    # ---- S_TURN (2) ----
    feats += [obs.turn / 20.0, 1.0 if obs.turn % 2 == 1 else 0.0]

    # ---- S_DECK (2) ----
    feats += [me.deck_count / 30.0, opp.deck_count / 30.0]

    # ---- S_HAND_EXTRA (2) ----
    feats += [
        sum(c.cost for c in playable) / 20.0,
        (max((c.cost for c in playable), default=0)) / 10.0,
    ]

    # ---- S_AWAITING (1) + S_CHOICE_CNT (1) ----
    feats.append(1.0 if obs.awaiting_choice else 0.0)
    choice_count = sum(1 for a in actions if a.type == ActionType.CHOOSE)
    feats.append(min(choice_count, 8) / 8.0)

    # ---- S_BIAS (1) ----
    feats.append(1.0)

    assert len(feats) == STATE_DIM, f"{len(feats)} != {STATE_DIM} (diff={len(feats) - STATE_DIM})"
    return feats


# ================================================================ 辅助


def _resolve_target(obs, action: Action):
    """从 (side, pos) 解出目标的大致属性（Hero 没有 EntityView，用 dict 代替）。"""
    if action.target_side < 0:
        return None
    side = obs.me if action.target_side == 0 else obs.opponent
    if action.target_pos < 0:
        # 目标是英雄
        return {
            "attack": side.hero_attack,
            "health": side.hero_health,
            "taunt": False, "divine_shield": False,
            "stealth": False, "poisonous": False,
        }
    field = side.field
    if 0 <= action.target_pos < len(field):
        return field[action.target_pos]
    return None


def _get_source(obs, action: Action):
    """从 source_pos 取攻击者 Entity（英雄返回 None）。"""
    if action.source_pos < 0:
        return None
    return obs.me.field[action.source_pos]


def _attacker_damage(obs, action: Action) -> int:
    """攻击动作的伤害量。"""
    if action.type == ActionType.HERO_POWER:
        return 1  # 法师火冲 = 1 伤
    if action.type == ActionType.ATTACK:
        if action.source_pos < 0:
            return obs.me.hero_attack
        return obs.me.field[action.source_pos].attack
    return 0


def _my_lethal(obs) -> bool:
    """我能用能动随从 + 武器直接斩杀对方吗（无嘲讽挡着）？"""
    if any(m.taunt for m in obs.opponent.field):
        return False
    dmg = sum(m.attack for m in obs.me.field if m.can_attack)
    dmg += obs.me.hero_attack
    return dmg >= obs.opponent.hero_health


def _enemy_lethal(obs) -> bool:
    """对方能斩杀我吗（考虑对方场上随从的 burst + 武器）。

    注意 rosetta 对方随从的 `can_attack` 在我方回合恒为 False，
    所以改用"有冲锋/突袭 + 武器"做 burst 估计。
    """
    if any(m.taunt for m in obs.me.field):
        return False
    dmg = sum(m.attack for m in obs.opponent.field
              if m.charge or m.rush)
    dmg += obs.opponent.hero_attack
    return dmg >= obs.me.hero_health
