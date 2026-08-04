"""人为构造的规则对手，也是衡量模型强弱的标尺。

三个难度递增的对手：

    RandomBot  合法动作里随机挑一个（纯基准线）
    GreedyBot  能打脸就打脸，打完把最贵的随从往外甩
    RuleBot    先算斩杀，再挑最赚的交换，最后用满水晶
    MakeBot    按名字或构造函数返回一个对手
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from .cards import (
    DIVINE_SHIELD,
    LIFESTEAL,
    POISONOUS,
    REBORN,
    TAUNT,
    THE_COIN,
    WINDFURY,
    CardDef,
)
from .game import (
    END_TURN,
    HERO,
    HERO_SOURCE,
    Action,
    Minion,
    Observation,
)


class Bot:
    """对手接口：给一个观测，返回一个合法动作。"""

    name = "bot"

    def choose(self, obs: Observation) -> Action:  # pragma: no cover - 接口
        raise NotImplementedError

    def bind_game(self, game, seat: int) -> None:
        """开局时拿到真实的 `Game`，搜索型选手要靠它克隆局面往下推演。

        默认什么都不做——只看观测的选手不需要这个。拿到 `Game` 就等于拿到了
        对手手牌和牌序，实现方**必须自己守住信息边界**，参考
        `search.TurnSearchAgent` 的做法（克隆后先洗掉自己的牌堆）。
        """


class RandomBot(Bot):
    """完全随机，胜率基准线。"""

    name = "random"

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def choose(self, obs: Observation) -> Action:
        return self.rng.choice(obs.legal)


class GreedyBot(Bot):
    """只会打脸的莽夫：随从全砸脸上，再把水晶花光。

    有嘲讽挡着就先打嘲讽，打死了再转脸；手上有幸运币就立刻用。
    """

    name = "greedy"

    def choose(self, obs: Observation) -> Action:
        # 幸运币：手里有就立刻用
        for action in obs.playable():
            if obs.hand[action.source].name == THE_COIN.name:
                return action

        # 伤害法术：优先打脸
        for action in obs.playable():
            card = obs.hand[action.source]
            if card.fx.damage > 0 and action.target == HERO:
                return action

        # 攻击：优先级 打脸 > 打嘲讽
        for action in obs.attacks():
            if action.target == HERO:
                return action

        for action in obs.attacks():
            return action

        # 伤害/变形法术：打随从
        for action in obs.playable():
            card = obs.hand[action.source]
            if (card.fx.damage > 0 or card.fx.transform) and action.target != HERO:
                return action

        # AoE / 清场 / 抽牌 / 飞弹：有就用
        for action in obs.playable():
            card = obs.hand[action.source]
            if (card.fx.aoe_enemy_minions > 0 or card.fx.aoe_all_enemies > 0
                    or card.fx.aoe_all > 0 or card.fx.destroy_all or card.fx.brawl
                    or card.fx.draw > 0 or card.fx.missiles > 0):
                return action

        # 出牌：贵 → 便宜
        plays = [a for a in obs.playable() if not obs.hand[a.source].spell]
        if plays:
            return max(plays, key=lambda a: (
                obs.hand[a.source].cost,
                obs.hand[a.source].stats,
            ))

        return END_TURN


class RuleBot(Bot):
    """比 Greedy 多了场面判断：不在劣势时莽脸、不在优势时浪费伤害做交换。

    核心逻辑：
        1. 能斩杀 → 打脸（先清嘲讽）
        2. 对面有高价值目标 + 自己能白嫖或剧毒换 → 交换
        3. 自己场面大于对面 → 打脸施压
        4. 否则老老实实用最优交换稳住局面
        5. 最后用满水晶

    一个回合内**先攻击再出牌**——先攻击可以打死对面随从腾出场地。
    """

    name = "rule"

    def choose(self, obs: Observation) -> Action:
        # 幸运币：手里有就立刻用
        for action in obs.playable():
            if obs.hand[action.source].name == THE_COIN.name:
                return action

        # 1. 能斩杀
        kill = self._lethal_sequence(obs)
        if kill is not None:
            return kill

        # 2. 法术：伤害打最值的目标 / 抽牌有闲费就用
        spell = self._best_spell(obs)
        if spell is not None:
            return spell

        # 3. 最优单次攻击
        attack_act = self._best_attack(obs)
        if attack_act is not None:
            return attack_act

        # 4. 残血自爆
        yolo = self._best_yolo(obs)
        if yolo is not None:
            return yolo

        # 5. 出牌：用满水晶
        play_act = self._best_play(obs)
        if play_act is not None:
            return play_act

        return END_TURN

    # ------------------------------------------------------------------ 斩杀

    def _lethal_sequence(self, obs: Observation) -> Optional[Action]:
        if not obs.has_lethal():
            return None
        # 清掉嘲讽
        taunts = obs.enemy_taunts()
        if taunts:
            for a in obs.attacks():
                if a.target in taunts:
                    return a
            return None
        face = [a for a in obs.attacks() if a.target == HERO]
        return face[0] if face else None

    # ------------------------------------------------------------------ 攻击

    def _best_attack(self, obs: Observation) -> Optional[Action]:
        attacks = obs.attacks()
        if not attacks:
            return None

        # 场面领先：偏打脸
        ahead = self._board_power(obs.board) >= self._board_power(obs.enemy_board) * 1.2

        best_score, best_action = -999.0, None
        for action in attacks:
            score = self._attack_score(obs, action, ahead)
            if score > best_score:
                best_score, best_action = score, action

        # 领先时只有很赚的交易才做，落后时保命要紧
        threshold = 1.0 if ahead else -5.0
        return best_action if best_score > threshold else None

    def _attack_score(self, obs: Observation, action: Action, ahead: bool) -> float:
        if action.source == HERO_SOURCE:
            return self._hero_attack_score(obs, action, ahead)

        attacker = obs.board[action.source]

        if action.target == HERO:
            dmg = attacker.attack * attacker.attacks_left
            weight = 0.9 if ahead else 0.6
            urgency = 1.0 + max(0.0, 10 - obs.enemy_hero_health) / 10.0
            if attacker.health <= 1:
                urgency += 0.5
            return dmg * weight * urgency

        defender = obs.enemy_board[action.target]
        return self._trade_value(attacker, defender)

    def _hero_attack_score(self, obs: Observation, action: Action, ahead: bool) -> float:
        weapon_atk = obs.hero_weapon_attack
        if action.target == HERO:
            weight = 0.9 if ahead else 0.6
            urgency = 1.0 + max(0.0, 10 - obs.enemy_hero_health) / 10.0
            return weapon_atk * weight * urgency

        defender = obs.enemy_board[action.target]
        # 英雄攻击随从：获得对方价值，损失自己的生命（反伤）
        kills_def = weapon_atk >= _effective_hp(defender)
        gain = _body_value(defender) if kills_def else 0.0
        loss = defender.attack * 0.5  # 掉血比掉随从便宜
        score = gain - loss
        if kills_def:
            score += 1.0  # 白嫖加分
        return score

    # ------------------------------------------------------------------ 交易估价

    @staticmethod
    def _trade_value(attacker: Minion, defender: Minion) -> float:
        kills_def = attacker.attack >= _effective_hp(defender)
        kills_atk = defender.attack >= _effective_hp(attacker)

        if not kills_def and kills_atk:
            # 除非能剧毒换掉对方，否则绝不白送
            if attacker.has(POISONOUS):
                return _body_value(defender) * 0.8
            return -10.0

        # 获得对方的价值
        gain = _body_value(defender) if kills_def else 0.0

        # 自己的损失
        if attacker.divine_shield and not kills_def:
            # 只掉盾不掉血 → 圣盾破掉值 0.8
            loss = 0.8
        elif attacker.divine_shield and kills_def:
            # 盾挡了反伤 → 只掉盾
            loss = 1.5
        elif kills_atk:
            loss = _body_value(attacker)
        else:
            # 没死：吃多少伤害就损失体量的多少比例
            ratio = min(defender.attack, attacker.health) / attacker.max_health
            loss = _body_value(attacker) * ratio

        score = gain - loss

        if kills_def and not kills_atk:
            score += 1.5                           # 白嫖

        if kills_def and attacker.has(POISONOUS):
            score += 2.0                           # 剧毒赚卡差

        if kills_def and attacker.has(LIFESTEAL) and attacker.attack >= 2:
            score += 0.3                           # 吸血顺便回血

        return score

    # ------------------------------------------------------------------ 残血

    def _best_yolo(self, obs: Observation) -> Optional[Action]:
        attacks = obs.attacks()
        best_score, best_action = -999.0, None
        for action in attacks:
            if action.source == HERO_SOURCE:
                continue  # 英雄不会残血
            att = obs.board[action.source]
            if att.health > 1:
                continue
            if action.target == HERO:
                score = att.attack * att.attacks_left * 0.8
            else:
                score = self._trade_value(att, obs.enemy_board[action.target])
            if score > best_score:
                best_score, best_action = score, action
        return best_action if best_score > 0.5 else None

    # ------------------------------------------------------------------ 法术

    def _best_spell(self, obs: Observation) -> Optional[Action]:
        """伤害/变形/AoE/抽牌——每种法术有自己的决策逻辑。"""
        best_score, best_action = -999.0, None
        for action in obs.playable():
            card = obs.hand[action.source]
            score = -999.0
            if card.fx.damage > 0 or card.fx.transform:
                score = self._damage_spell_score(obs, action)
            elif card.fx.aoe_enemy_minions > 0 or card.fx.aoe_all_enemies > 0 or card.fx.aoe_all > 0:
                score = self._aoe_score(obs, card)
            elif card.fx.destroy_all or card.fx.brawl:
                score = self._board_clear_score(obs, card)
            elif card.fx.draw > 0:
                if obs.mana >= card.cost + 2:
                    score = float(card.fx.draw)
            elif card.fx.missiles > 0:
                if obs.enemy_board:
                    score = 3.0
            if score > best_score:
                best_score, best_action = score, action

        return best_action if best_score > 0 else None

    def _aoe_score(self, obs: Observation, card) -> float:
        """AoE 价值：看能打到几个敌方随从，清掉多少体量。"""
        dmg = card.fx.aoe_enemy_minions or card.fx.aoe_all_enemies or card.fx.aoe_all
        en_board = obs.enemy_board
        if not en_board:
            return 0.0
        value = 0.0
        for m in en_board:
            eff_hp = m.health + (1 if m.divine_shield else 0)
            if dmg >= eff_hp:
                value += _body_value(m)  # 打死
            else:
                value += dmg * 0.3  # 打残
        # AoE 打到英雄也有价值
        if card.fx.aoe_all_enemies > 0 or card.fx.aoe_all > 0:
            value += dmg * 0.5  # 打脸
        # 打到自己的惩罚
        if card.fx.aoe_all > 0:
            for m in obs.board:
                value -= min(dmg, m.health) * 0.3
            value -= dmg * 0.5  # 打自己
        return value

    def _board_clear_score(self, obs: Observation, card) -> float:
        """清场法术：场面落后时价值高。"""
        en_board = obs.enemy_board
        if not en_board:
            return 0.0
        en_power = sum(_body_value(m) for m in en_board)
        my_power = sum(_body_value(m) for m in obs.board)
        # 敌方场面远大于我方时才清
        if en_power > my_power * 1.5:
            return en_power - my_power  # 净收益
        return 0.0

    def _damage_spell_score(self, obs: Observation, action: Action) -> float:
        card = obs.hand[action.source]
        dmg = card.fx.damage
        if action.target == HERO:
            # 打脸：伤害价值 + 斩杀权重
            urgency = 1.0 + max(0.0, 10 - obs.enemy_hero_health) / 5.0
            if dmg >= obs.enemy_hero_health:
                return 100.0  # 直接赢
            return dmg * 0.7 * urgency
        # 打随从
        defender = obs.enemy_board[action.target]
        kills = dmg >= (defender.health + (1 if defender.divine_shield else 0))
        if not kills:
            return dmg * 0.3  # 打残价值不高
        # 打死了：获得对方价值
        score = _body_value(defender)
        if defender.has("剧毒"):
            score += 1.5
        if defender.has("嘲讽"):
            score += 1.0
        return score

    # ------------------------------------------------------------------ 出牌

    def _best_play(self, obs: Observation) -> Optional[Action]:
        plays = [a for a in obs.playable() if not obs.hand[a.source].spell]
        if not plays:
            return None

        def key(action: Action) -> tuple:
            card: CardDef = obs.hand[action.source]
            return (card.cost, card.stats, -card.attack)

        return max(plays, key=key)

    # ------------------------------------------------------------------ 场面

    @staticmethod
    def _board_power(board: List[Minion]) -> float:
        """场上随从的总战力：攻+血+关键词加成。"""
        return sum(_body_value(m) for m in board)


# ---------------------------------------------------------------- 辅助

def _effective_hp(minion: Minion) -> int:
    """考虑圣盾的等效血量：圣盾在等于多加一点有效生命。"""
    hp = minion.health
    if minion.divine_shield:
        hp += 1
    return hp


def _body_value(minion: Minion) -> float:
    """随从当前的"体量价值"。带关键词再略加一点。"""
    base = float(minion.attack + minion.health)
    for word in (TAUNT, LIFESTEAL, POISONOUS, WINDFURY):
        if minion.has(word):
            base += 0.5
    if minion.has(DIVINE_SHIELD) and minion.divine_shield:
        base += 1.0
    if minion.has(REBORN) and minion.reborn:
        base += 0.5
    return base


def _chip(damage: int, minion: Minion) -> float:
    """不是致死伤害时，按掉血占比折算当前体量。"""
    if minion.max_health <= 0:
        return 0.0
    return _body_value(minion) * min(damage, minion.health) / minion.max_health


# ---------------------------------------------------------------- 构造

BOTS: Dict[str, type] = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "rule": RuleBot,
}


def make_bot(name: str, seed: Optional[int] = None) -> Bot:
    """按名字构造对手。"""
    if name not in BOTS:
        raise ValueError(f"未知对手 {name!r}，可选: {', '.join(BOTS)}")
    bot = BOTS[name]
    return bot(seed=seed) if name == "random" else bot()
