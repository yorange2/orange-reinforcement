"""手写规则对手。从 `rosetta/bots.py` 平移，口径一致方便横向对照。

适配 orange-stone 结构化视图的两个差异：
- 动作是 `Action`（kind/card_index/entity_id/target_id），没有 rosetta 的
  `target_side`/`target_pos`——攻击目标不在 `opponent.field` 里就是打脸
  （英雄不占场上槽位，实体槽是全局下标）；
- `EntityView` 只暴露基础关键词（嘲讽/圣盾/潜行/风怒/冲锋），剧毒/吸血等
  还没进视图（M5 再说），所以 `_body_value`/`_trade_value` 里没有它们的加分项。
"""

from __future__ import annotations

import random
from typing import Callable, Optional

from .env import Action

__all__ = ["RandomBot", "GreedyBot", "RuleBot", "BOTS"]

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
        choices = [a for a in actions if a.kind == "choose"]
        if choices:
            return self._rng.choice(choices)

        # 幸运币：手里有就立刻用
        for a in actions:
            if a.kind == "play":
                card = obs.me.hand[a.card_index]
                if card.card_id == THE_COIN:
                    return a

        # 英雄技能能打脸就打脸（当前引擎没有英雄技能，这段是 M5 预留）
        for a in actions:
            if a.kind == "hero_power" and _targets_face(a, obs):
                return a

        # 攻击：能打脸就打脸，打不到脸随便打一个
        for a in actions:
            if a.kind == "attack" and _targets_face(a, obs):
                return a

        for a in actions:
            if a.kind == "attack":
                return a

        # 英雄技能打随从
        powers = [a for a in actions if a.kind == "hero_power"]
        if powers:
            return self._rng.choice(powers)

        # 出牌：贵 → 便宜
        plays = [a for a in actions if a.kind == "play"]
        if plays:
            return max(plays, key=lambda a: obs.me.hand[a.card_index].cost)

        return next(a for a in actions if a.kind == "end_turn")


class RuleBot:
    """比 Greedy 多了场面判断和最优交换。

    核心逻辑（抄 rosetta/bots.py，字段换成 orange-stone 视图口径）：
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
        choices = [a for a in actions if a.kind == "choose"]
        if choices:
            return random.choice(choices)

        # 幸运币：手里有就立刻用
        for a in actions:
            if a.kind == "play":
                if obs.me.hand[a.card_index].card_id == THE_COIN:
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

        return next(a for a in actions if a.kind == "end_turn")

    # ------------------------------------------------------------------ 斩杀

    def _lethal(self, obs, actions: list[Action]) -> Optional[Action]:
        """能打死对面英雄就出手，有嘲讽就先清嘲讽。"""
        # 计算能打到脸的伤害
        attacks = [a for a in actions if a.kind == "attack"]
        face_attacks = [a for a in attacks if _targets_face(a, obs)]
        total_face_dmg = sum(
            _source_attack(a, obs) * _attacks_left(a, obs)
            for a in face_attacks
        )

        powers = [a for a in actions if a.kind == "hero_power"]
        face_powers = [a for a in powers if _targets_face(a, obs)]
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
        attacks = [a for a in actions if a.kind == "attack"]
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
        if best_score > threshold:
            return best_action

        # 有嘲讽挡着时攻击是被**强制**的（不打就永远开不了脸），这时的交换
        # 全是负分——但放任嘲讽在场只会让对面白打脸。选代价最小的那次强制
        # 交换出手，这是 rule 相对只会打脸的 greedy 的优势点。
        if best_action is not None and _target_is_taunt(best_action, obs):
            return best_action
        return None

    def _attack_score(self, obs, action: Action, ahead: bool) -> float:
        if _source_is_hero(action, obs):
            return self._hero_attack_score(obs, action, ahead)

        attacker = _source_minion(action, obs)

        if _targets_face(action, obs):
            dmg = attacker.attack * _attacks_left(action, obs)
            weight = _face_weight(obs, ahead)
            urgency = 1.0 + max(0.0, 10 - obs.opponent.hero_health) / 10.0
            if attacker.health <= 1:
                urgency += 0.5
            return dmg * weight * urgency

        defender = _target_minion(action, obs)
        return _trade_value(attacker, defender)

    def _hero_attack_score(self, obs, action: Action, ahead: bool) -> float:
        weapon_atk = obs.me.weapon_attack
        if _targets_face(action, obs):
            weight = _face_weight(obs, ahead)
            urgency = 1.0 + max(0.0, 10 - obs.opponent.hero_health) / 10.0
            return weapon_atk * weight * urgency

        defender = _target_minion(action, obs)
        kills_def = weapon_atk >= _effective_hp(defender)
        gain = _body_value(defender) if kills_def else 0.0
        loss = defender.attack * 0.5  # 掉血比掉随从便宜
        score = gain - loss
        if kills_def:
            score += 1.0  # 白嫖加分
        return score

    # ------------------------------------------------------------------ 残血

    def _best_yolo(self, obs, actions: list[Action]) -> Optional[Action]:
        """1 血随从的"反正要死"攻击：不打白不打。

        1 血随从无论打不打都可能被对面白吃，攻击的边际代价≈0——所以阈值
        放宽到 -1（哪怕只蹭 1 点血也是赚的），rosetta 版的 0.5 在这个纯随从
        卡池上太保守。
        """
        attacks = [a for a in actions if a.kind == "attack"]
        best_score, best_action = -999.0, None
        for a in attacks:
            if _source_is_hero(a, obs):
                continue
            att = _source_minion(a, obs)
            if att.health > 1:
                continue
            if _targets_face(a, obs):
                score = att.attack * 0.8
            else:
                score = _trade_value(att, _target_minion(a, obs))
            if score > best_score:
                best_score, best_action = score, a
        return best_action if best_score > -1.0 else None

    # ------------------------------------------------------------------ 英雄技能

    def _best_hero_power(self, obs, actions: list[Action]) -> Optional[Action]:
        """英雄技能：能赚就按。

        当前 orange-stone 引擎的英雄没有英雄技能（2026-08 核实，结构化动作
        里没有 hero_power 出现过），这段逻辑为 M5 预留，抄 rosetta 口径：
        优先补刀残血随从（破圣盾 / 收 1 血），有余费再打脸。
        """
        powers = [a for a in actions if a.kind == "hero_power"]
        if not powers:
            return None

        have_better_plays = any(
            a.kind == "play"
            and obs.me.hand[a.card_index].cost <= obs.me.remaining_mana
            for a in actions
        )

        # 补刀能打死的随从 → 血赚
        for a in powers:
            target = _target_minion(a, obs)
            if target is not None and _hero_power_damage(a, obs) >= _effective_hp(target):
                return a

        # 有余费就打脸
        for a in powers:
            if _targets_face(a, obs):
                if not have_better_plays or obs.me.remaining_mana >= 4:
                    return a

        return None

    # ------------------------------------------------------------------ 出牌

    def _best_play(self, obs, actions: list[Action]) -> Optional[Action]:
        plays = [a for a in actions if a.kind == "play"]
        if not plays:
            return None

        def _key(a: Action) -> tuple:
            card = obs.me.hand[a.card_index]
            if card.card_type == 1:   # 法术：按费用 + 效果量级
                return (card.cost, card.bc_damage + card.bc_heal, -card.cost)
            if card.card_type == 2:   # 武器：按费用 + 攻/耐久
                return (card.cost, card.attack + card.health, -card.attack)
            return (card.cost, card.attack + card.health, -card.attack)
        return max(plays, key=_key)


# ================================================================== 辅助


def _hero_power_damage(action: Action, obs) -> int:
    """当前英雄技能对指定目标造成的伤害。

    当前引擎没有英雄技能（这段是 M5 预留），而 orange-stone 的 hero_power
    动作不带目标（目标经 choose 动作补选），所以只能按默认口径估：基准对局
    都是镜像职业，默认法师火焰冲击的 1 点。
    """
    return 1


def _face_weight(obs, ahead: bool) -> float:
    """打脸权重：领先 0.9；落后 0.6；血量赛跑按 0.9 抢脸。

    "血量赛跑"指自己血量明显比对手低——这时候抠交换是慢性死亡，
    只有抢脸对换才可能翻盘。反过来血量领先也不抠（纯随从卡池没有
    清场，交换很难扳回节奏，用血量优势对抢更赚）。
    """
    if ahead:
        return 0.9
    if abs(obs.me.hero_health - obs.opponent.hero_health) >= 5:
        return 0.9
    return 0.6


def _targets_face(action: Action, obs) -> bool:
    """目标是对方英雄（target_id 不在对方场上 = 英雄）。"""
    return action.target_id not in _minion_ids(obs.opponent.field)


def _target_is_taunt(action: Action, obs) -> bool:
    """目标是一个有嘲讽的随从。"""
    minion = _target_minion(action, obs)
    return minion is not None and minion.taunt


def _source_is_hero(action: Action, obs) -> bool:
    """攻击者是英雄（entity_id 不在我方场上）。"""
    return action.entity_id not in _minion_ids(obs.me.field)


def _source_minion(action: Action, obs):
    """攻击者随从；英雄攻击时返回 None。"""
    for minion in obs.me.field:
        if minion.entity_id == action.entity_id:
            return minion
    return None


def _target_minion(action: Action, obs):
    """目标随从；打脸时返回 None。"""
    for minion in obs.opponent.field:
        if minion.entity_id == action.target_id:
            return minion
    return None


def _minion_ids(field: list) -> set:
    return {m.entity_id for m in field}


def _source_attack(action: Action, obs) -> int:
    """攻击者的攻击力。"""
    minion = _source_minion(action, obs)
    return minion.attack if minion is not None else obs.me.hero_attack


def _attacks_left(action: Action, obs) -> int:
    """攻击者还能打几次（风怒可能给 2，这里保守按 1 估）。"""
    if _source_is_hero(action, obs):
        return obs.me.hero_attack > 0  # 武器还在就能打
    return 1


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
    if entity.windfury:
        base += 0.5
    if entity.divine_shield:
        base += 1.0
    if entity.charge:
        base += 0.3
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

    return score


# ================================================================== 注册

BOTS: dict[str, Callable[[int | None], object]] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "rule": RuleBot,
}
