"""冻结的结算。

最容易写错的是**解冻时机**：在自己回合结束时解冻，所以被冻的一方要实打实错过
一整个回合。如果写成"回合开始解冻"，冻结就完全没有效果了。
"""

import random
import unittest

from hearthstone.cards import CARD_INDEX, POOL, CardDef
from hearthstone.game import END_TURN, HERO, PLAY, Action, Game, Minion, attack


def card(name: str) -> CardDef:
    return POOL[CARD_INDEX[name]]


class Fixture(unittest.TestCase):
    def setup(self, hand=(), mine=(), theirs=(), mana=10):
        game = Game(rng=random.Random(0))
        game.current = 0
        game.mana = [mana, mana]
        game.max_mana = [mana, mana]
        game.hands = [[card(n) for n in hand], []]
        game.boards = [
            [Minion.summon(card(n), game._take_uid()) for n in mine],
            [Minion.summon(card(n), game._take_uid()) for n in theirs],
        ]
        for board in game.boards:
            for m in board:
                m.just_played = False
        game.hero_health = [30, 30]
        return game


class TestFreezeApplies(Fixture):
    def test_targeted_freeze(self):
        game = self.setup(hand=["冰霜震击"], theirs=["冰风雪人"])
        game.step(Action(PLAY, 0, 0))
        self.assertTrue(game.boards[1][0].frozen)

    def test_frost_nova_freezes_every_enemy_minion(self):
        game = self.setup(hand=["冰霜新星"], theirs=["冰风雪人", "石拳食人魔"])
        game.step(Action(PLAY, 0))
        self.assertTrue(all(m.frozen for m in game.boards[1]))

    def test_frost_nova_leaves_my_own_minions_alone(self):
        game = self.setup(hand=["冰霜新星"], mine=["冰风雪人"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0))
        self.assertFalse(game.boards[0][0].frozen)

    def test_blizzard_damages_and_freezes(self):
        game = self.setup(hand=["暴风雪"], theirs=["石拳食人魔"])       # 6/7，挨 2 点
        game.step(Action(PLAY, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 2)
        self.assertTrue(game.boards[1][0].frozen)

    def test_battlecry_freeze(self):
        game = self.setup(hand=["冰川裂片"], theirs=["冰风雪人"])
        game.step(Action(PLAY, 0, 0))
        self.assertTrue(game.boards[1][0].frozen)

    def test_can_freeze_the_hero(self):
        game = self.setup(hand=["冰霜震击"])
        game.step(Action(PLAY, 0, HERO))
        self.assertTrue(game.hero_frozen[1])

    def test_elusive_still_blocks_targeted_freeze(self):
        game = self.setup(hand=["寒冰箭"], theirs=["精灵龙"])
        targets = [a.target for a in game.legal_actions(0) if a.kind == PLAY]
        self.assertNotIn(0, targets)

    def test_aoe_freeze_ignores_elusive(self):
        """冻结所有敌方随从不是"指定"，扰咒挡不住。"""
        game = self.setup(hand=["冰霜新星"], theirs=["精灵龙"])
        game.step(Action(PLAY, 0))
        self.assertTrue(game.boards[1][0].frozen)


class TestFrozenCannotAttack(Fixture):
    def test_frozen_minion_has_no_attack_action(self):
        game = self.setup(mine=["冰风雪人"], theirs=["石拳食人魔"])
        self.assertTrue(any(a.kind == "attack" for a in game.legal_actions(0)))
        game.boards[0][0].frozen = True
        self.assertFalse(any(a.kind == "attack" for a in game.legal_actions(0)))

    def test_frozen_hero_cannot_swing_weapon(self):
        game = self.setup(theirs=["冰风雪人"])
        game.weapons[0] = card("炽炎战斧")
        game.weapon_durability[0] = 2
        self.assertTrue(any(a.source == -2 for a in game.legal_actions(0)))
        game.hero_frozen[0] = True
        self.assertFalse(any(a.source == -2 for a in game.legal_actions(0)))


class TestThawTiming(Fixture):
    """解冻时机——这组是冻结有没有实际效果的关键。"""

    def test_victim_misses_a_whole_turn(self):
        game = self.setup(hand=["冰霜震击"], theirs=["冰风雪人"])
        game.step(Action(PLAY, 0, 0))
        self.assertTrue(game.boards[1][0].frozen)

        game.step(END_TURN)                       # 玩家 0 结束回合
        self.assertTrue(game.boards[1][0].frozen, "轮到对手时还应该冻着，这一回合白丢")
        self.assertFalse(any(a.kind == "attack" for a in game.legal_actions(1)))

        game.step(END_TURN)                       # 对手结束回合 → 这时才解冻
        self.assertFalse(game.boards[1][0].frozen)

    def test_my_own_turn_end_thaws_me(self):
        game = self.setup(mine=["冰风雪人"])
        game.boards[0][0].frozen = True
        game.step(END_TURN)
        self.assertFalse(game.boards[0][0].frozen, "自己回合结束就该化开")

    def test_hero_thaws_too(self):
        game = self.setup()
        game.hero_frozen[0] = True
        game.step(END_TURN)
        self.assertFalse(game.hero_frozen[0])


class TestFrozenIsVisibleToTheAgent(Fixture):
    def test_frozen_minion_reads_as_cannot_act(self):
        """特征里没有单独的"冻结"维度，但"能动"那一维会自动反映出来。"""
        from hearthstone.features import _board_slot_feature
        game = self.setup(mine=["冰风雪人"])
        minion = game.boards[0][0]
        before = _board_slot_feature(minion)
        minion.frozen = True
        after = _board_slot_feature(minion)
        self.assertNotEqual(before, after, "冻结必须能被特征看见")


class TestCloneCarriesFrozen(Fixture):
    def test_clone_preserves_frozen_state(self):
        game = self.setup(mine=["冰风雪人"])
        game.boards[0][0].frozen = True
        game.hero_frozen[1] = True
        twin = game.clone()
        self.assertTrue(twin.boards[0][0].frozen)
        self.assertTrue(twin.hero_frozen[1])
        twin.boards[0][0].frozen = False
        self.assertTrue(game.boards[0][0].frozen, "克隆体不能影响原局面")


if __name__ == "__main__":
    unittest.main()
