"""把 (局面, 某个候选动作) 编码成定长向量。

v2：比初版加了约 20 维，集中在四个地方——
1. 交易结果（这一刀下去谁死、场面差怎么变）
2. 手牌质量（不只看有几张，还看能出什么）
3. 场上关键词分布（剧毒、吸血、复生严重影响交易决策）
4. 斩杀检测（自己和对手分别能不能在下一次攻击中结束比赛）

v3：从 81 维升到 129 维（+48 维局面特征），打破纯聚合编码——
5. 逐随从编码（双方场上各前 3 大随从的攻/血/能动/嘲讽/圣盾）
6. 逐手牌编码（可出牌中前 3 低费的费/攻/血/冲锋/突袭）
7. 法术感知（手牌中是否有直伤/AOE/硬解）
8. 先后手改为 Observation 显式字段，不再用牌堆差推断

先知特征（oracle）：另一套**只给价值头**的特征，编码对手的真实手牌。它不进
`state_features`，所以策略永远看不到，推理时也不计算——见 `oracle_features`。
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
S_BOARD_SLOTS = 7 * 7 + 7 * 7  # 双方场上逐随从（各7槽全覆盖）：攻/血/能动/嘲讽/圣盾/剧毒/吸血
S_HAND_CARDS = 5 * 3           # 手牌逐卡（前3低费可出）：费/攻/血/冲锋/突袭
S_SPELLS = 3                    # 法术感知：直伤/AOE/硬解
S_KEYWORDS = 4 + 4              # 双方场上关键词计数（剧毒/吸血/风怒/复生）
S_LETHAL = 2                    # 斩杀检测
S_OTHER = 1 + 1 + 1 + 1 + 1     # 牌堆/对手手牌/疲劳/bias/先后手

STATE_DIM = S_BASE + S_WEAPON + S_HAND + S_BOARD + S_BOARD_SLOTS + S_HAND_CARDS + S_SPELLS + S_KEYWORDS + S_LETHAL + S_OTHER

STATE_OFFSET = ACTION_DIM
FEATURE_DIM = ACTION_DIM + STATE_DIM

# ---------------------------------------------------------------- 先知维度布局

O_AGG = 6                       # 对手手牌聚合：总攻/总血/可出数/可出攻/可出血/均费
O_FLAGS = 3                     # 对手下回合能出的牌里有冲锋/嘲讽/突袭
O_CARDS = 3 * 5                 # 对手手牌逐卡（前 3 低费可出）：费/攻/血/冲锋/突袭
O_SPELLS = 3                    # 对手手牌法术感知：直伤/AOE/硬解
O_BURST = 2                     # 对手手牌爆发：伤害总量 + 是否够斩杀我
O_BIAS = 1

ORACLE_DIM = O_AGG + O_FLAGS + O_CARDS + O_SPELLS + O_BURST + O_BIAS


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


def oracle_features(obs: Observation) -> np.ndarray:
    """先知特征：把**对手的真实手牌**编码成定长向量。

    只给非对称 actor-critic 的价值头用。价值头不参与线上决策，所以这里的作弊
    不会泄漏到实际打法——详见 `Observation.enemy_hand` 的说明。

    对手下回合的水晶用 `min(自己上限 + 1, 10)` 估计：双方水晶上限最多差 1，
    这个近似足够判断"他下回合能拍下什么"。
    """
    hand = obs.enemy_hand
    if not hand:
        return np.zeros(ORACLE_DIM, dtype=np.float32)

    mana = min(obs.max_mana + 1, 10)
    playable = [c for c in hand if c.cost <= mana]

    # 爆发：能直接打脸的法术伤害 + 冲锋随从的攻击力
    burst = 0
    for c in playable:
        if c.spell:
            burst += c.fx.damage + c.fx.missiles
        elif c.has("冲锋"):
            burst += c.attack

    sorted_playable = sorted(playable, key=lambda c: (c.cost, -(c.attack + c.health)))
    cards: List[float] = []
    for i in range(3):
        cards.extend(_hand_card_feature(sorted_playable[i]) if i < len(sorted_playable)
                     else [0.0] * 5)

    feats = [
        # 聚合
        sum(c.attack for c in hand) / 20.0,
        sum(c.health for c in hand) / 30.0,
        len(playable) / 6.0,
        sum(c.attack for c in playable) / 20.0,
        sum(c.health for c in playable) / 30.0,
        (sum(c.cost for c in hand) / len(hand)) / 10.0,
        # 下回合能拍出来的关键词
        1.0 if any(c.has("冲锋") for c in playable) else 0.0,
        1.0 if any(c.has("嘲讽") for c in playable) else 0.0,
        1.0 if any(c.has("突袭") for c in playable) else 0.0,
        # 逐卡
        *cards,
        # 法术感知
        *_spell_flags(playable),
        # 爆发
        burst / 30.0,
        1.0 if burst >= obs.hero_health else 0.0,
        # bias
        1.0,
    ]
    assert len(feats) == ORACLE_DIM, f"{len(feats)} != {ORACLE_DIM}"
    return np.asarray(feats, dtype=np.float32)


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
    is_targeted_spell = card.fx.damage > 0 or card.fx.transform
    target_is_minion = not target_is_hero and is_targeted_spell

    if target_is_minion and action.target < len(obs.enemy_board):
        defender = obs.enemy_board[action.target]
        def_atk, def_hp = defender.attack / 10.0, defender.health / 10.0
        def_taunt = 1.0 if defender.taunting else 0.0
        def_shield = 1.0 if defender.divine_shield else 0.0
        def_poison = 1.0 if defender.has("剧毒") else 0.0
        kills_def = 1.0  # 伤害法术能打死就算，变形术无条件"消灭"
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

def _board_slot_feature(m: "Minion") -> List[float]:
    """单个随从的 7 维编码：攻/血/能动/嘲讽/圣盾/剧毒/吸血。"""
    return [
        m.attack / 10.0,
        m.health / 10.0,
        1.0 if m.can_attack else 0.0,
        1.0 if m.taunting else 0.0,
        1.0 if m.divine_shield else 0.0,
        1.0 if m.has("剧毒") else 0.0,
        1.0 if m.has("吸血") else 0.0,
    ]


def _board_slots(board: List["Minion"], n: int = 7) -> List[float]:
    """场上按出场顺序的前 n 个随从的编码，不足补零。

    7 槽全覆盖（BOARD_LIMIT=7），按 uid 升序（出场顺序）——
    随从在存活期间槽位不变，模型可稳定追踪。
    """
    sorted_board = sorted(board, key=lambda m: m.uid)
    feats: List[float] = []
    for i in range(n):
        if i < len(sorted_board):
            feats.extend(_board_slot_feature(sorted_board[i]))
        else:
            feats.extend([0.0] * 7)
    return feats


def _hand_card_feature(card) -> List[float]:
    """手牌中单张可出牌的 5 维编码：费/攻/血/冲锋/突袭。"""
    return [
        card.cost / 10.0,
        card.attack / 10.0,
        card.health / 10.0,
        1.0 if card.has("冲锋") else 0.0,
        1.0 if card.has("突袭") else 0.0,
    ]


def _hand_cards(obs: Observation, n: int = 3) -> List[float]:
    """可出牌中费用最低的 n 张的编码，不足补零。"""
    playable = obs.playable()
    cards = [obs.hand[a.source] for a in playable]
    sorted_cards = sorted(cards, key=lambda c: (c.cost, -(c.attack + c.health)))
    feats: List[float] = []
    for i in range(n):
        if i < len(sorted_cards):
            feats.extend(_hand_card_feature(sorted_cards[i]))
        else:
            feats.extend([0.0] * 5)
    return feats


def _spell_awareness(obs: Observation) -> List[float]:
    """可出牌中是否有直伤/AOE/硬解法术。"""
    return _spell_flags([obs.hand[a.source] for a in obs.playable()])


def _spell_flags(cards: List) -> List[float]:
    """给定一组牌，返回是否含直伤/AOE/硬解法术。局面特征和先知特征共用。"""
    has_damage = any(c.fx.damage > 0 or c.fx.missiles > 0 for c in cards)
    has_aoe = any(
        c.fx.aoe_enemy_minions > 0 or c.fx.aoe_all_enemies > 0
        or c.fx.aoe_all > 0 or c.fx.splash > 0
        for c in cards
    )
    has_removal = any(
        c.fx.transform or c.fx.destroy_all or c.fx.brawl
        for c in cards
    )
    return [
        1.0 if has_damage else 0.0,
        1.0 if has_aoe else 0.0,
        1.0 if has_removal else 0.0,
    ]


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

    # 先后手（直接用 Observation 的显式字段）
    going_first = 1.0 if obs.going_first else 0.0

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
        # 手牌逐卡（前 3 低费可出）
        *_hand_cards(obs),
        # 法术感知
        *_spell_awareness(obs),
        # 场面大小
        len(my) / 7.0,
        len(en) / 7.0,
        sum(m.attack for m in my) / 20.0,
        sum(m.attack for m in en) / 20.0,
        sum(m.health for m in my) / 30.0,
        sum(m.health for m in en) / 30.0,
        1.0 if obs.enemy_taunts() else 0.0,
        # 双方场上逐随从（各前 3 大）
        *_board_slots(my),
        *_board_slots(en),
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
