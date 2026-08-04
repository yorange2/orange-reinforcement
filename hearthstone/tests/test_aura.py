"""光环：随从在场期间持续改变别人的属性，离场立刻还原。

最容易写错的两件事，各有一组用例钉着：

1. **光环加成不能和永久增益混在一起。** 2/2 被暴风城勇士加成成 3/3，再吃一个
   +1/+1 的永久增益变 4/4；勇士死了必须回到 3/3，而不是 2/2 或者 4/4。
2. **光环消失时血量要正确回落。** 血量存的是"受了多少伤"而不是当前值，所以
   一个被加成过又受过伤的随从，光环没了不会算成负数、也不会把伤害抹掉。
"""

import random
import unittest

from hearthstone.cards import CARD_INDEX, POOL, CardDef
from hearthstone.game import PLAY, Action, Game, Minion


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
        game._refresh_auras()
        return game

    def stats(self, game, player=0):
        return [(m.attack, m.health) for m in game.boards[player]]


class TestAuraApplies(Fixture):
    def test_raid_leader_buffs_others_only(self):
        game = self.setup(mine=["团队领袖", "冰风雪人"])       # 2/2 光环, 4/5
        self.assertEqual(self.stats(game), [(2, 2), (5, 5)])

    def test_stormwind_champion_buffs_attack_and_health(self):
        game = self.setup(mine=["暴风城勇士", "冰风雪人"])     # 6/6 光环, 4/5
        self.assertEqual(self.stats(game), [(6, 6), (5, 6)])

    def test_aura_does_not_cross_sides(self):
        game = self.setup(mine=["团队领袖"], theirs=["冰风雪人"])
        self.assertEqual(self.stats(game, 1), [(4, 5)])

    def test_two_auras_stack(self):
        game = self.setup(mine=["团队领袖", "暴风城勇士", "冰风雪人"])
        # 冰风雪人 4/5 拿到 +1/+0 和 +1/+1
        self.assertEqual(self.stats(game)[2], (6, 6))

    def test_adjacent_aura_only_hits_neighbours(self):
        game = self.setup(mine=["幽灵", "炎锤先锋", "幽灵", "幽灵"])
        atk = [m.attack for m in game.boards[0]]
        self.assertEqual(atk, [2, 3, 2, 1], "只有左右紧邻的两个 +1")

    def test_aura_applies_when_source_enters(self):
        game = self.setup(hand=["团队领袖"], mine=["冰风雪人"])
        self.assertEqual(self.stats(game), [(4, 5)])
        game.step(Action(PLAY, 0))
        self.assertEqual(game.boards[0][0].attack, 5, "光环随从落地就该生效")


class TestAuraRemoval(Fixture):
    def test_aura_disappears_when_source_dies(self):
        game = self.setup(mine=["团队领袖", "冰风雪人"])
        self.assertEqual(game.boards[0][1].attack, 5)
        game.boards[0][0].to_be_destroyed = True
        game._clear_dead()
        self.assertEqual(self.stats(game), [(4, 5)], "来源没了就该还原")

    def test_permanent_buff_survives_aura_removal(self):
        """这是最容易写错的一条：光环加成和永久增益必须分开记。"""
        game = self.setup(mine=["暴风城勇士", "冰风雪人"])
        yeti = game.boards[0][1]
        self.assertEqual((yeti.attack, yeti.health), (5, 6))   # 4/5 + 光环 1/1

        yeti.buff_attack += 1                                   # 再吃一个永久 +1/+1
        yeti.buff_health += 1
        self.assertEqual((yeti.attack, yeti.health), (6, 7))

        game.boards[0][0].to_be_destroyed = True
        game._clear_dead()
        self.assertEqual((yeti.attack, yeti.health), (5, 6),
                         "光环没了要退回 4/5+永久1/1 = 5/6，不是 4/5 也不是 6/7")

    def test_damaged_minion_keeps_its_damage(self):
        """被加成过又受过伤的随从，光环消失后血量要正确回落。"""
        game = self.setup(mine=["暴风城勇士", "冰风雪人"])
        yeti = game.boards[0][1]
        game._hit(yeti, 3)                                      # 6 血挨 3 → 3
        self.assertEqual(yeti.health, 3)

        game.boards[0][0].to_be_destroyed = True
        game._clear_dead()
        self.assertEqual(yeti.max_health, 5)
        self.assertEqual(yeti.health, 2, "上限 5 减去 3 点伤害")

    def test_minion_dies_when_its_aura_health_vanishes(self):
        """靠光环血量撑着的随从，来源一死它也跟着走。"""
        game = self.setup(mine=["暴风城勇士", "幽灵"])           # 幽灵 1/1 → 2/2
        ghost = game.boards[0][1]
        game._hit(ghost, 1)
        self.assertEqual(ghost.health, 1)

        game.boards[0][0].to_be_destroyed = True
        game._clear_dead()
        self.assertEqual(len(game.boards[0]), 0, "上限回落到 1，扣掉 1 点伤害就是 0")


class TestAuraIsIdempotent(Fixture):
    def test_repeated_refresh_does_not_drift(self):
        """全量重算的关键性质：算多少次结果都一样。"""
        game = self.setup(mine=["团队领袖", "冰风雪人"])
        before = self.stats(game)
        for _ in range(20):
            game._refresh_auras()
        self.assertEqual(self.stats(game), before)

    def test_aura_is_visible_in_features(self):
        from hearthstone.features import _board_slot_feature
        game = self.setup(mine=["冰风雪人"])
        plain = _board_slot_feature(game.boards[0][0])
        game.boards[0].insert(0, Minion.summon(card("团队领袖"), 99))
        game._refresh_auras()
        buffed = _board_slot_feature(game.boards[0][1])
        self.assertNotEqual(plain, buffed, "光环加成必须能被特征看见")


class TestCloneIsolatesAuras(Fixture):
    def test_clone_recomputes_independently(self):
        game = self.setup(mine=["团队领袖", "冰风雪人"])
        twin = game.clone()
        twin.boards[0][0].to_be_destroyed = True
        twin._clear_dead()
        self.assertEqual(self.stats(twin), [(4, 5)])
        self.assertEqual(self.stats(game), [(2, 2), (5, 5)], "原局面不受影响")


if __name__ == "__main__":
    unittest.main()
