"""手写规则对手。和 `hearthstone/bots.py` 一个口径，方便两边横向对照。"""

from __future__ import annotations

import random
from typing import Callable, Optional

from .env import Action, ActionType

__all__ = ["RandomBot", "GreedyBot", "RuleBot", "BOTS"]

HERO = -1       # target_pos / source_pos 为 -1 表示英雄
SELF = 0        # target_side = 0 表示当前玩家
ENEMY = 1       # target_side = 1 表示对手

THE_COIN = "GAME_005"


# ================================================================== 对手


class RandomBot:
    """合法动作里均匀随机，纯基准线。"""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, obs, actions: list[Action]) -> Action:
        return self._rng.choice(actions)


class GreedyBot:
    """只会打脸的莽夫：随从全砸脸上，再用光水晶下随从。"""

    name = "greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def choose(self, obs, actions: list[Action]) -> Action:
        # 有选择要做（发现 / 抉择）时随便选一个
        choices = [a for a in actions if a.type == ActionType.CHOOSE]
        if choices:
            return self._rng.choice(choices)

        # 幸运币：手里有就立刻用
        for a in actions:
            if a.type == ActionType.PLAY_CARD:
                card = obs.me.hand[a.hand_idx]
                if card.card_id == THE_COIN:
                    return a

        # 英雄技能能打脸就打脸
        for a in actions:
            if a.type == ActionType.HERO_POWER and _targets_face(a):
                return a

        # 攻击：能打脸就打脸，打不到脸随便打一个
        for a in actions:
            if a.type == ActionType.ATTACK and _targets_face(a):
                return a

        for a in actions:
            if a.type == ActionType.ATTACK:
                return a

        # 英雄技能打随从
        powers = [a for a in actions if a.type == ActionType.HERO_POWER]
        if powers:
            return self._rng.choice(powers)

        # 出牌：贵 → 便宜
        plays = [a for a in actions if a.type == ActionType.PLAY_CARD]
        if plays:
            return max(plays, key=lambda a: obs.me.hand[a.hand_idx].cost)

        return next(a for a in actions if a.type == ActionType.END_TURN)


class RuleBot:
    """比 Greedy 多了场面判断和最优交换。

    核心逻辑：
        1. 能斩杀 → 打脸（先清嘲讽）
        2. 最优攻击（场面领先就施压，落后就抠交换）
        3. 残血自爆
        4. 英雄技能 / 法术
        5. 出牌用满水晶

    一个回合内**先攻击再出牌**——先攻击可以打死对面随从腾出场地。
    """

    name = "rule"

    def __init__(self, seed: int | None = None) -> None:
        # 纯确定性对手，seed 只是接口兼容
        pass

    # ------------------------------------------------------------------ 入口

    def choose(self, obs, actions: list[Action]) -> Action:
        # 有选择要做时随便选一个
        choices = [a for a in actions if a.type == ActionType.CHOOSE]
        if choices:
            return random.choice(choices)

        # 幸运币：手里有就立刻用
        for a in actions:
            if a.type == ActionType.PLAY_CARD:
                if obs.me.hand[a.hand_idx].card_id == THE_COIN:
                    return a

        # 1. 能斩杀
        kill = self._lethal(obs, actions)
        if kill is not None:
            return kill

        # 2. 英雄技能／法术——在攻击之前用，可以破圣盾或补刀残血随从
        power = self._best_hero_power(obs, actions)
        if power is not None:
            return power

        # 3. 最优攻击
        attack = self._best_attack(obs, actions)
        if attack is not None:
            return attack

        # 4. 残血自爆
        yolo = self._best_yolo(obs, actions)
        if yolo is not None:
            return yolo

        # 5. 出牌：用满水晶
        play = self._best_play(obs, actions)
        if play is not None:
            return play

        return next(a for a in actions if a.type == ActionType.END_TURN)

    # ------------------------------------------------------------------ 斩杀

    def _lethal(self, obs, actions: list[Action]) -> Optional[Action]:
        """能打死对面英雄就出手，有嘲讽就先清嘲讽。"""
        # 计算能打到脸的伤害
        attacks = [a for a in actions if a.type == ActionType.ATTACK]
        face_attacks = [a for a in attacks if _targets_face(a)]
        total_face_dmg = sum(
            _source_attack(a, obs) * _attacks_left(a, obs)
            for a in face_attacks
        )

        powers = [a for a in actions if a.type == ActionType.HERO_POWER]
        face_powers = [a for a in powers if _targets_face(a)]
        hp_dmg = sum(_hero_power_damage(a, obs) for a in face_powers)

        if total_face_dmg + hp_dmg < obs.opponent.hero_health:
            return None

        # 有嘲讽挡着 → 先清嘲讽
        taunts = [a for a in attacks if _target_is_taunt(a, obs)]
        if taunts:
            return taunts[0]

        if face_attacks:
            return face_attacks[0]
        if face_powers:
            return face_powers[0]

        return None

    # ------------------------------------------------------------------ 攻击

    def _best_attack(self, obs, actions: list[Action]) -> Optional[Action]:
        """挑最优的那次攻击。领先时偏打脸，落后时抠交换。"""
        attacks = [a for a in actions if a.type == ActionType.ATTACK]
        if not attacks:
            return None

        my_power = _board_power(obs.me.field)
        en_power = _board_power(obs.opponent.field)
        ahead = my_power >= en_power * 1.2

        best_score, best_action = -999.0, None
        for a in attacks:
            score = self._attack_score(obs, a, ahead)
            if score > best_score:
                best_score, best_action = score, a

        threshold = 1.0 if ahead else -5.0
        return best_action if best_score > threshold else None

    def _attack_score(self, obs, action: Action, ahead: bool) -> float:
        if action.source_pos == HERO:
            return self._hero_attack_score(obs, action, ahead)

        attacker = obs.me.field[action.source_pos]

        if _targets_face(action):
            dmg = attacker.attack * _attacks_left(action, obs)
            weight = 0.9 if ahead else 0.6
            urgency = 1.0 + max(0.0, 10 - obs.opponent.hero_health) / 10.0
            if attacker.health <= 1:
                urgency += 0.5
            return dmg * weight * urgency

        defender = obs.opponent.field[action.target_pos]
        return _trade_value(attacker, defender)

    def _hero_attack_score(self, obs, action: Action, ahead: bool) -> float:
        weapon_atk = obs.me.weapon_attack
        if _targets_face(action):
            weight = 0.9 if ahead else 0.6
            urgency = 1.0 + max(0.0, 10 - obs.opponent.hero_health) / 10.0
            return weapon_atk * weight * urgency

        defender = obs.opponent.field[action.target_pos]
        kills_def = weapon_atk >= _effective_hp(defender)
        gain = _body_value(defender) if kills_def else 0.0
        loss = defender.attack * 0.5  # 掉血比掉随从便宜
        score = gain - loss
        if kills_def:
            score += 1.0  # 白嫖加分
        return score

    # ------------------------------------------------------------------ 残血

    def _best_yolo(self, obs, actions: list[Action]) -> Optional[Action]:
        attacks = [a for a in actions if a.type == ActionType.ATTACK]
        best_score, best_action = -999.0, None
        for a in attacks:
            if a.source_pos == HERO:
                continue
            att = obs.me.field[a.source_pos]
            if att.health > 1:
                continue
            if _targets_face(a):
                score = att.attack * 0.8
            else:
                score = _trade_value(att, obs.opponent.field[a.target_pos])
            if score > best_score:
                best_score, best_action = score, a
        return best_action if best_score > 0.5 else None

    # ------------------------------------------------------------------ 英雄技能

    def _best_hero_power(self, obs, actions: list[Action]) -> Optional[Action]:
        """英雄技能：能赚就按。

        优先补刀残血随从（破圣盾 / 收 1 血），有余费再打脸。
        """
        powers = [a for a in actions if a.type == ActionType.HERO_POWER]
        if not powers:
            return None

        have_better_plays = any(
            a.type == ActionType.PLAY_CARD
            and obs.me.hand[a.hand_idx].cost <= obs.me.remaining_mana
            for a in actions
        )

        # 补刀能打死的随从 → 血赚
        for a in powers:
            if a.target_side == ENEMY and a.target_pos != HERO:
                target = obs.opponent.field[a.target_pos]
                if _hero_power_damage(a, obs) >= _effective_hp(target):
                    return a

        # 有余费就打脸
        for a in powers:
            if _targets_face(a):
                if not have_better_plays or obs.me.remaining_mana >= 4:
                    return a

        return None

    # ------------------------------------------------------------------ 出牌

    def _best_play(self, obs, actions: list[Action]) -> Optional[Action]:
        plays = [a for a in actions if a.type == ActionType.PLAY_CARD]
        if not plays:
            return None

        def _key(a: Action) -> tuple:
            card = obs.me.hand[a.hand_idx]
            return (card.cost, card.attack + card.health, -card.attack)
        return max(plays, key=_key)


# ================================================================== 辅助

HERO_POWER_DAMAGE: dict[str, int] = {
    "HERO_08bp":   1,  # 法师 — 火焰冲击
    "HERO_08bp2":  2,  # 法师（升级）— 火焰冲击
    "HERO_01bp":   0,  # 战士 — 加固（护甲不算伤害）
    "HERO_01bp2":  0,
    "HERO_02bp":   0,  # 萨满 — 图腾召唤
    "HERO_03bp":   0,  # 盗贼 — 匕首精通
    "HERO_04bp":   0,  # 圣骑士 — 援军
    "HERO_05bp":   2,  # 猎人 — 稳固射击
    "HERO_05bp2":  3,
    "HERO_06bp":   0,  # 德鲁伊 — 变形（+1攻+1甲）
    "HERO_06bp2":  0,
    "HERO_07bp":   0,  # 术士 — 生命分流
    "HERO_09bp":   0,  # 牧师 — 次级治疗术
    "HERO_10bp":   0,  # 恶魔猎手 — 恶魔之爪
}


def _hero_power_damage(action: Action, obs) -> int:
    """当前英雄技能对指定目标造成的伤害（0 表示不是伤害技能）。"""
    # 英雄技能的 card_id 不在 observation 里。从目标的 side 和
    # 已知的技能效果来推断：以敌方为目标且有伤害值的技能就是伤害技能。
    if action.target_side != ENEMY:
        return 0
    # 大多数职业的英雄技能不给对手带来正面效果，保守起见只要有
    # 已知伤害值就算伤害，否则不算。
    return 1  # 默认法师的 1 点——当前基准都用 MAGE


def _targets_face(action: Action) -> bool:
    """目标是对方英雄。"""
    return action.target_side == ENEMY and action.target_pos == HERO


def _target_is_taunt(action: Action, obs) -> bool:
    """目标是一个有嘲讽的随从。"""
    if action.target_side != ENEMY or action.target_pos == HERO:
        return False
    field = obs.opponent.field
    return action.target_pos < len(field) and field[action.target_pos].taunt


def _source_attack(action: Action, obs) -> int:
    """攻击者的攻击力。"""
    if action.source_pos == HERO:
        return obs.me.hero_attack
    field = obs.me.field
    return field[action.source_pos].attack if action.source_pos < len(field) else 0


def _attacks_left(action: Action, obs) -> int:
    """攻击者还能打几次（风怒可能给 2）。"""
    if action.source_pos == HERO:
        return obs.me.hero_attack > 0  # 武器还在就能打
    field = obs.me.field
    if action.source_pos < len(field):
        return 1  # 简化：不考虑风怒（白板卡池没有风怒随从能活到攻击）
    return 0


def _effective_hp(entity) -> int:
    """等效血量：圣盾等于多 1 点。"""
    hp = entity.health
    if entity.divine_shield:
        hp += 1
    return hp


def _body_value(entity) -> float:
    """随从当前的"体量价值"。带关键词再加一点权重。"""
    base = float(entity.attack + entity.health)
    if entity.taunt:
        base += 0.5
    if entity.lifesteal:
        base += 0.5
    if entity.poisonous:
        base += 0.5
    if entity.windfury:
        base += 0.5
    if entity.divine_shield:
        base += 1.0
    return base


def _board_power(field: list) -> float:
    """场上总战力。"""
    return sum(_body_value(m) for m in field)


def _trade_value(attacker, defender) -> float:
    """一次攻击交换的价值（攻方视角）。"""
    kills_def = attacker.attack >= _effective_hp(defender)
    kills_atk = defender.attack >= _effective_hp(attacker)

    if not kills_def and kills_atk:
        # 打不死对方但对方能打死我 → 除非剧毒换掉，否则不干
        if attacker.poisonous:
            return _body_value(defender) * 0.8
        return -10.0

    # 获得对方的价值
    gain = _body_value(defender) if kills_def else 0.0

    # 自己的损失
    if attacker.divine_shield and not kills_def:
        loss = 0.8
    elif attacker.divine_shield and kills_def:
        loss = 1.5
    elif kills_atk:
        loss = _body_value(attacker)
    else:
        ratio = min(defender.attack, attacker.health) / max(1, attacker.health)
        loss = _body_value(attacker) * ratio

    score = gain - loss

    if kills_def and not kills_atk:
        score += 1.5                           # 白嫖

    if kills_def and attacker.poisonous:
        score += 2.0

    if kills_def and attacker.lifesteal and attacker.attack >= 2:
        score += 0.3

    return score


# ================================================================== 注册

BOTS: dict[str, Callable[[int | None], object]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "rule": RuleBot,
}
