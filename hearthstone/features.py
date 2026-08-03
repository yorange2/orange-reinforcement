"""把 (局面, 某个候选动作) 编码成定长向量。

和跑得快一样：策略网络对每个候选动作单独打分，再在候选上做 softmax，动作空间变长也不怕。

前 36 维描述动作本身，后 18 维描述局面（同一决策点里对所有候选相同）。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .cards import KEYWORDS
from .game import ATTACK, END, HERO, PLAY, Action, Observation

N_KEYWORDS = len(KEYWORDS)

#: 动作特征维数
ACTION_DIM = 3 + 1 + 2 + N_KEYWORDS + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 1 + 1 + 1
#  = is_play/is_attack/is_end (3)
#  + card_cost (1)
#  + card_attack, card_health (2)
#  + card_keyword_onehot (N_KEYWORDS)
#  + attacker_attack, attacker_health (2)
#  + attacker_attacks_left (1)
#  + attacker_divine_shield (1)
#  + attacker_poisonous (1)
#  + attacker_lifesteal (1)
#  + attacker_windfury (1)
#  + target_is_hero (1)
#  + defender_attack, defender_health (2)
#  + defender_taunt (1)
#  + defender_divine_shield (1)
#  + defender_poisonous (1)

#: 局面特征维数
STATE_DIM = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1

#: 局面特征起始下标
STATE_OFFSET = ACTION_DIM

FEATURE_DIM = ACTION_DIM + STATE_DIM


def batch_features(obs: Observation) -> np.ndarray:
    """一次算出所有候选动作的特征，返回 (候选数, FEATURE_DIM) 的矩阵。"""
    tail = state_features(obs)
    rows = np.empty((len(obs.legal), FEATURE_DIM), dtype=np.float32)
    for i, action in enumerate(obs.legal):
        rows[i, :STATE_OFFSET] = action_features(obs, action)
        rows[i, STATE_OFFSET:] = tail
    return rows


# ---------------------------------------------------------------- 动作

def action_features(obs: Observation, action: Action) -> List[float]:
    if action.kind == PLAY:
        return _play_features(obs, action)
    if action.kind == ATTACK:
        return _attack_features(obs, action)
    if action.kind == END:
        return _end_features()
    raise ValueError(f"未知动作 {action.kind}")


def _play_features(obs: Observation, action: Action) -> List[float]:
    card = obs.hand[action.source]
    keywords = _keyword_vector(card.keywords)
    return [
        # 动作类型
        1.0, 0.0, 0.0,
        # 卡牌本身
        card.cost / 10.0,
        card.attack / 10.0,
        card.health / 10.0,
        *keywords,
        # 攻击相关（出牌时全是 0）
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]


def _attack_features(obs: Observation, action: Action) -> List[float]:
    attacker = obs.board[action.source]
    target_is_hero = action.target == HERO
    if target_is_hero:
        def_atk, def_hp = 0.0, 0.0
        def_taunt, def_shield, def_poison = 0.0, 0.0, 0.0
    else:
        defender = obs.enemy_board[action.target]
        def_atk, def_hp = defender.attack / 10.0, defender.health / 10.0
        def_taunt = 1.0 if defender.has("嘲讽") else 0.0
        def_shield = 1.0 if defender.divine_shield else 0.0
        def_poison = 1.0 if defender.has("剧毒") else 0.0

    return [
        # 动作类型
        0.0, 1.0, 0.0,
        # 卡牌相关（攻击时全是 0）
        0.0, 0.0, 0.0,
        *([0.0] * N_KEYWORDS),
        # 攻击者
        attacker.attack / 10.0,
        attacker.health / 10.0,
        float(attacker.attacks_left - 1) / 1.0,   # 0 = 最后一次，1 = 还能打一次(风怒)
        1.0 if attacker.divine_shield else 0.0,
        1.0 if attacker.has("剧毒") else 0.0,
        1.0 if attacker.has("吸血") else 0.0,
        1.0 if attacker.has("风怒") else 0.0,
        # 目标
        1.0 if target_is_hero else 0.0,
        def_atk,
        def_hp,
        def_taunt,
        def_shield,
        def_poison,
    ]


def _end_features() -> List[float]:
    return [
        0.0, 0.0, 1.0,
        0.0, 0.0, 0.0,
        *([0.0] * N_KEYWORDS),
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
    ]


# ---------------------------------------------------------------- 局面

def state_features(obs: Observation) -> List[float]:
    """只跟局面有关的特征。"""
    my_board = obs.board
    en_board = obs.enemy_board

    return [
        obs.mana / 10.0,
        obs.max_mana / 10.0,
        obs.hero_health / 30.0,
        obs.enemy_hero_health / 30.0,
        len(obs.hand) / 10.0,
        len(obs.playable()) / 6.0,
        len(my_board) / 7.0,
        len(en_board) / 7.0,
        sum(m.attack for m in my_board) / 20.0,
        sum(m.attack for m in en_board) / 20.0,
        sum(m.health for m in my_board) / 30.0,
        sum(m.health for m in en_board) / 30.0,
        float(sum(1 for m in en_board if m.taunting)),
        float(sum(1 for m in my_board if m.divine_shield)),
        float(sum(1 for m in en_board if m.divine_shield)),
        obs.deck_size / 30.0,
        obs.enemy_hand_size / 10.0,
        1.0,  # bias
    ]


# ---------------------------------------------------------------- 辅助

def _keyword_vector(keywords) -> List[float]:
    vec = [0.0] * N_KEYWORDS
    for word in keywords:
        from .cards import KEYWORD_INDEX
        idx = KEYWORD_INDEX.get(word)
        if idx is not None:
            vec[idx] = 1.0
    return vec
