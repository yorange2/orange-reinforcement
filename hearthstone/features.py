"""把 (局面, 某个候选动作) 编码成定长向量。

v2：比初版加了约 20 维，集中在四个地方——
1. 交易结果（这一刀下去谁死、场面差怎么变）
2. 手牌质量（不只看有几张，还看能出什么）
3. 场上关键词分布（剧毒、吸血、复生严重影响交易决策）
4. 斩杀检测（自己和对手分别能不能在下一次攻击中结束比赛）
"""

from __future__ import annotations

from typing import List

import numpy as np

from .cards import KEYWORDS
from .game import ATTACK, END, HERO, HERO_SOURCE, PLAY, Action, Observation

N_KEYWORDS = len(KEYWORDS)

# ---------------------------------------------------------------- 维度布局

# 动作特征
A_TYPE = 3                     # 动作类型 one-hot (play/attack/end)
A_CARD = 1 + 2 + N_KEYWORDS    # 出牌：费用 + 攻/血 + 关键词 one-hot
A_ATTACKER = 2 + 1 + 1 + 1 + 1 + 1  # 攻击者：攻/血 + 剩余次数 + 圣盾/剧毒/吸血/风怒
A_TARGET = 1 + 2 + 1 + 1 + 1    # 目标：人脸/攻/血 + 嘲讽/圣盾/剧毒
A_TRADE = 2 + 1 + 1 + 1 + 1 + 1 + 1  # 交易结果：谁死 + 过杀 + 场面差 + 复生/吸血/冲锋突袭
A_PLAY_EFFECT = 2 + 1           # 出牌效果：剩余水晶 + 用光/铺满
A_LEGAL = 1                    # 候选动作数

ACTION_DIM = A_TYPE + A_CARD + A_ATTACKER + A_TARGET + A_TRADE + A_PLAY_EFFECT + A_LEGAL

# 局面特征
S_BASE = 1 + 1 + 1 + 1 + 1    # 水晶、血量
S_WEAPON = 5                    # 武器：自己攻/耐久/已攻击 + 对方攻/耐久
S_HAND = 1 + 1 + 2 + 1 + 1 + 1  # 手牌：张数 + 可出数 + 可出总攻/总血 + 有冲锋/有嘲讽/有突袭
S_BOARD = 2 + 2 + 2 + 1        # 场面：随从数 + 总攻 + 总血 + 嘲讽挡脸
S_KEYWORDS = 4 + 4              # 双方场上关键词计数（剧毒/吸血/风怒/复生）
S_LETHAL = 2                    # 斩杀检测
S_OTHER = 1 + 1 + 1 + 1 + 1     # 牌堆/对手手牌/疲劳/bias/先后手

STATE_DIM = S_BASE + S_WEAPON + S_HAND + S_BOARD + S_KEYWORDS + S_LETHAL + S_OTHER

STATE_OFFSET = ACTION_DIM
FEATURE_DIM = ACTION_DIM + STATE_DIM


# ---------------------------------------------------------------- 对外接口

def batch_features(obs: Observation) -> np.ndarray:
    tail = state_features(obs)
    rows = np.empty((len(obs.legal), FEATURE_DIM), dtype=np.float32)
    for i, action in enumerate(obs.legal):
        rows[i, :STATE_OFFSET] = action_features(obs, action)
        rows[i, STATE_OFFSET:] = tail
    return rows


def action_features(obs: Observation, action: Action) -> List[float]:
    feats = _base_features(obs, action)
    # 公共：候选动作数
    feats.append(min(len(obs.legal), 30) / 30.0)
    assert len(feats) == ACTION_DIM, f"{len(feats)} != {ACTION_DIM}"
    return feats


def state_features(obs: Observation) -> List[float]:
    feats = _state_features(obs)
    assert len(feats) == STATE_DIM, f"{len(feats)} != {STATE_DIM}"
    return feats


# ---------------------------------------------------------------- 动作

def _base_features(obs: Observation, action: Action) -> List[float]:
    if action.kind == PLAY:
        return _play(obs, action)
    if action.kind == ATTACK:
        return _attack(obs, action)
    return _end()


def _play(obs: Observation, action: Action) -> List[float]:
    card = obs.hand[action.source]
    kw = _kw_vec(card.keywords)
    left = obs.mana - card.cost
    target_is_hero = action.target == HERO
    target_is_minion = not target_is_hero and card.spell_damage > 0

    if target_is_minion and action.target < len(obs.enemy_board):
        defender = obs.enemy_board[action.target]
        def_atk, def_hp = defender.attack / 10.0, defender.health / 10.0
        def_taunt = 1.0 if defender.taunting else 0.0
        def_shield = 1.0 if defender.divine_shield else 0.0
        def_poison = 1.0 if defender.has("剧毒") else 0.0
        kills_def = 1.0 if card.spell_damage >= (defender.health + (1 if defender.divine_shield else 0)) else 0.0
    else:
        def_atk, def_hp = 0.0, 0.0
        def_taunt, def_shield, def_poison = 0.0, 0.0, 0.0
        kills_def = 0.0

    feats = [
        # 类型
        1.0, 0.0, 0.0,
        # 卡牌
        card.cost / 10.0,
        card.attack / 10.0,
        card.health / 10.0,
        *kw,
        # 攻击者（法术不需要）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # 目标（伤害法术直接写目标信息，其他置零）
        1.0 if target_is_hero else 0.0,
        def_atk,
        def_hp,
        def_taunt,
        def_shield,
        def_poison,
        # 交易结果
        kills_def, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        # 出牌效果
        left / 10.0,
        1.0 if card.cost == obs.mana and card.cost > 0 else 0.0,
        1.0 if len(obs.board) >= 6 else 0.0,
    ]
    return feats


def _attack(obs: Observation, action: Action) -> List[float]:
    if action.source == HERO_SOURCE:
        return _hero_attack_features(obs, action)

    att = obs.board[action.source]
    target_is_hero = action.target == HERO

    # 目标信息
    if target_is_hero:
        def_atk, def_hp = 0.0, 0.0
        def_taunt, def_shield, def_poison = 0.0, 0.0, 0.0
        def_reborn, def_lifesteal = 0.0, 0.0
        kills_def, kills_att = 0.0, 0.0
        overkill = 0.0
        board_after = (sum(m.attack for m in obs.board if m.uid != att.uid)
                       - sum(m.attack for m in obs.enemy_board)) / 20.0
    else:
        defender = obs.enemy_board[action.target]
        def_atk = defender.attack / 10.0
        def_hp = defender.health / 10.0
        def_taunt = 1.0 if defender.taunting else 0.0
        def_shield = 1.0 if defender.divine_shield else 0.0
        def_poison = 1.0 if defender.has("剧毒") else 0.0
        def_reborn = 1.0 if defender.reborn else 0.0
        def_lifesteal = 1.0 if defender.has("吸血") else 0.0

        # 交易结果
        def_eff_hp = defender.health + (1 if defender.divine_shield else 0)
        att_eff_hp = att.health + (1 if att.divine_shield else 0)
        kills_def = 1.0 if att.attack >= def_eff_hp else 0.0
        kills_att = 1.0 if defender.attack >= att_eff_hp else 0.0
        overkill = max(0.0, att.attack - def_eff_hp) / 5.0

        # 交易后场面差（把当前参与的两个随从从场面里去掉）
        my_rest = sum(m.attack for m in obs.board if m.uid != att.uid)
        en_rest = sum(m.attack for m in obs.enemy_board if m.uid != defender.uid)
        if not kills_def:
            en_rest += defender.attack  # 没打死，还在
        if not kills_att:
            my_rest += att.attack
        board_after = (my_rest - en_rest) / 20.0

    feats = [
        # 类型
        0.0, 1.0, 0.0,
        # 卡牌（攻击时不涉及手牌）
        0.0, 0.0, 0.0,
        *([0.0] * N_KEYWORDS),
        # 攻击者
        att.attack / 10.0,
        att.health / 10.0,
        float(att.attacks_left - 1),
        1.0 if att.divine_shield else 0.0,
        1.0 if att.has("剧毒") else 0.0,
        1.0 if att.has("吸血") else 0.0,
        1.0 if att.has("风怒") else 0.0,
        # 目标
        1.0 if target_is_hero else 0.0,
        def_atk,
        def_hp,
        def_taunt,
        def_shield,
        def_poison,
        # 交易结果
        kills_def,
        kills_att,
        overkill,
        board_after,
        def_reborn,
        def_lifesteal,
        1.0 if att.reborn else 0.0,
        1.0 if att.has("冲锋") or att.has("突袭") else 0.0,
        # 出牌效果（攻击时不涉及）
        0.0, 0.0, 0.0,
    ]
    return feats


def _hero_attack_features(obs: Observation, action: Action) -> List[float]:
    """英雄拿武器攻击时的特征。"""
    weapon_atk = obs.hero_weapon_attack
    target_is_hero = action.target == HERO

    if target_is_hero:
        def_atk, def_hp = 0.0, 0.0
        def_taunt, def_shield, def_poison = 0.0, 0.0, 0.0
        def_reborn, def_lifesteal = 0.0, 0.0
        kills_def, kills_att = 0.0, 0.0
        overkill = 0.0
        board_after = (sum(m.attack for m in obs.board)
                       - sum(m.attack for m in obs.enemy_board)) / 20.0
    else:
        defender = obs.enemy_board[action.target]
        def_atk = defender.attack / 10.0
        def_hp = defender.health / 10.0
        def_taunt = 1.0 if defender.taunting else 0.0
        def_shield = 1.0 if defender.divine_shield else 0.0
        def_poison = 1.0 if defender.has("剧毒") else 0.0
        def_reborn = 1.0 if defender.reborn else 0.0
        def_lifesteal = 1.0 if defender.has("吸血") else 0.0
        def_eff_hp = defender.health + (1 if defender.divine_shield else 0)
        kills_def = 1.0 if weapon_atk >= def_eff_hp else 0.0
        kills_att = 0.0  # 英雄不会因为攻击随从死亡
        overkill = max(0.0, weapon_atk - def_eff_hp) / 5.0
        my_rest = sum(m.attack for m in obs.board)
        en_rest = sum(m.attack for m in obs.enemy_board if m.uid != defender.uid)
        board_after = (my_rest - en_rest) / 20.0

    feats = [
        0.0, 1.0, 0.0,    # 类型：attack
        0.0, 0.0, 0.0,     # 卡牌
        *([0.0] * N_KEYWORDS),
        weapon_atk / 10.0,  # 攻击者（武器攻击）
        0.0,                # 攻击者血量（英雄血量在局面里）
        0.0, 1.0 if weapon_atk > 0 else 0.0, 0.0, 0.0, 0.0,  # 圣盾/剧毒/吸血/风怒
        1.0 if target_is_hero else 0.0,
        def_atk, def_hp,
        def_taunt, def_shield, def_poison,
        kills_def, kills_att, overkill, board_after,
        def_reborn, def_lifesteal,
        0.0, 0.0,            # 复生/冲锋突袭（英雄没有）
        0.0, 0.0, 0.0,       # 出牌效果
    ]
    return feats


def _end() -> List[float]:
    feats = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        *([0.0] * N_KEYWORDS),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]
    return feats


# ---------------------------------------------------------------- 局面

def _state_features(obs: Observation) -> List[float]:
    my = obs.board
    en = obs.enemy_board

    # 手牌质量
    playable = obs.playable()
    play_atk = sum(obs.hand[a.source].attack for a in playable)
    play_hp = sum(obs.hand[a.source].health for a in playable)
    has_charge = 1.0 if any(obs.hand[a.source].has("冲锋") for a in playable) else 0.0
    has_taunt = 1.0 if any(obs.hand[a.source].has("嘲讽") for a in playable) else 0.0
    has_rush = 1.0 if any(obs.hand[a.source].has("突袭") for a in playable) else 0.0

    # 斩杀
    i_have_lethal = 1.0 if _can_kill(obs, obs.player) else 0.0
    opp_has_lethal = 1.0 if _can_kill(obs, obs.opponent) else 0.0

    # 先后手（通过水晶差推断：第一个回合 mana 上限都是 1，但后手有幸运币
    # 所以看 mana 上限 + 手牌中是否有幸运币来推）
    going_first = 1.0 if obs.max_mana == obs.turn // 2 + 1 else 0.0
    # 更简单的推断：先手 max_mana ≈ (turn+2)//2，后手 max_mana ≈ (turn+1)//2
    # 直接用 deck_size 差：后手起手多一张，牌堆少一张
    # 最准的：先手牌堆 = 后手牌堆 + 1（开局时），中期牌堆差接近 1
    if obs.deck_size > obs.enemy_deck_size:
        going_first = 1.0
    elif obs.deck_size < obs.enemy_deck_size:
        going_first = 0.0
    else:
        going_first = 0.5

    return [
        # 水晶血量
        obs.mana / 10.0,
        obs.max_mana / 10.0,
        obs.hero_health / 30.0,
        obs.enemy_hero_health / 30.0,
        obs.fatigue / 10.0,
        # 武器
        obs.hero_weapon_attack / 10.0,
        obs.hero_weapon_durability / 5.0,
        1.0 if obs.hero_attacked else 0.0,
        obs.enemy_weapon_attack / 10.0,
        obs.enemy_weapon_durability / 5.0,
        # 手牌质量
        len(obs.hand) / 10.0,
        len(playable) / 6.0,
        play_atk / 20.0,
        play_hp / 30.0,
        has_charge,
        has_taunt,
        has_rush,
        # 场面大小
        len(my) / 7.0,
        len(en) / 7.0,
        sum(m.attack for m in my) / 20.0,
        sum(m.attack for m in en) / 20.0,
        sum(m.health for m in my) / 30.0,
        sum(m.health for m in en) / 30.0,
        1.0 if obs.enemy_taunts() else 0.0,
        # 双方场上关键词
        float(sum(1 for m in my if m.has("剧毒"))),
        float(sum(1 for m in my if m.has("吸血"))),
        float(sum(1 for m in my if m.has("风怒"))),
        float(sum(1 for m in my if m.reborn)),
        float(sum(1 for m in en if m.has("剧毒"))),
        float(sum(1 for m in en if m.has("吸血"))),
        float(sum(1 for m in en if m.has("风怒"))),
        float(sum(1 for m in en if m.reborn)),
        # 斩杀检测
        i_have_lethal,
        opp_has_lethal,
        # 其他
        obs.deck_size / 30.0,
        obs.enemy_hand_size / 10.0,
        obs.enemy_fatigue / 10.0,
        going_first,
        1.0,  # bias
    ]


def _can_kill(obs: Observation, player: int) -> bool:
    """给定玩家在当前局面下，用所有能动的随从打脸能不能杀死对面。

    注意：不模拟对手回合，只看"如果现在所有攻击都打脸"。
    """
    if player == obs.player:
        board = obs.board
        enemy_health = obs.enemy_hero_health
    else:
        board = obs.enemy_board
        enemy_health = obs.hero_health

    # 有嘲讽挡着就不算（要模拟清嘲讽太复杂，保守一点）
    taunts = [m for m in (obs.enemy_board if player == obs.player else obs.board) if m.taunting]
    if taunts:
        return False

    damage = sum(m.attack * m.attacks_left for m in board if m.can_hit_face)
    return damage >= enemy_health


# ---------------------------------------------------------------- 辅助

def _kw_vec(keywords) -> List[float]:
    from .cards import KEYWORD_INDEX

    vec = [0.0] * N_KEYWORDS
    for word in keywords:
        idx = KEYWORD_INDEX.get(word)
        if idx is not None:
            vec[idx] = 1.0
    return vec
