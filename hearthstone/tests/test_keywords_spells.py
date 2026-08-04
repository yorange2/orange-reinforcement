"""扰咒和法术增强的结算。

这两个关键词以前只是显示、不产生效果。边界很容易写错，所以每条规则都单独钉一个用例：

    扰咒挡的是**指定目标**——AoE、溅射、随机伤害都不算指定，照样打得到
    法术增强只加**伤害**——变形、消灭、乱斗、抽牌都不受影响
"""

import random
import unittest

from hearthstone.cards import CARD_INDEX, ELUSIVE, POOL, SPELL_DAMAGE, CardDef
from hearthstone.game import PLAY, Action, Game, Minion


def card(name: str) -> CardDef:
    return POOL[CARD_INDEX[name]]


class Fixture(unittest.TestCase):
    """手工摆一个局面：轮到玩家 0，水晶拉满，手牌和场面自己指定。"""

    def setup(self, hand=(), mine=(), theirs=(), mana=10, enemy_health=30):
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
        game.hero_health = [30, enemy_health]
        return game


class TestElusiveBlocksTargeting(Fixture):
    def test_elusive_minion_is_not_a_legal_spell_target(self):
        game = self.setup(hand=["火球术"], theirs=["精灵龙", "冰风雪人"])
        targets = [a.target for a in game.legal_actions(0) if a.kind == PLAY]
        self.assertNotIn(0, targets, "精灵龙带扰咒，不该是合法目标")
        self.assertIn(1, targets, "冰风雪人没有扰咒，应该可以指定")

    def test_polymorph_cannot_target_elusive(self):
        game = self.setup(hand=["变形术"], theirs=["精灵龙"])
        targets = [a.target for a in game.legal_actions(0) if a.kind == PLAY]
        self.assertNotIn(0, targets)

    def test_hero_is_still_targetable(self):
        """场上全是扰咒随从时，法术仍然可以打脸。"""
        game = self.setup(hand=["火球术"], theirs=["精灵龙"])
        from hearthstone.game import HERO
        targets = [a.target for a in game.legal_actions(0) if a.kind == PLAY]
        self.assertIn(HERO, targets)

    def test_casting_at_elusive_is_rejected_by_the_engine(self):
        """就算绕过 legal_actions 直接 step，引擎也要拦住。"""
        game = self.setup(hand=["火球术"], theirs=["精灵龙"])
        with self.assertRaises(ValueError):
            game.step(Action(PLAY, 0, 0))

    def test_elusive_does_not_block_attacks(self):
        """扰咒只挡法术，随从攻击照打。"""
        game = self.setup(mine=["冰风雪人"], theirs=["精灵龙"])
        attacks = [(a.source, a.target) for a in game.legal_actions(0) if a.kind == "attack"]
        self.assertIn((0, 0), attacks)


class TestElusiveDoesNotBlockUntargeted(Fixture):
    def test_aoe_still_hits_elusive(self):
        game = self.setup(hand=["烈焰风暴"], theirs=["精灵龙"])   # 敌方随从 4 点
        game.step(Action(PLAY, 0))
        self.assertEqual(game.boards[1], [], "3/2 的精灵龙应该被 4 点 AoE 打死")

    def test_consecration_still_hits_elusive(self):
        """奉献对敌方全体 2 点。精灵龙是 3/2，正好被打死。"""
        game = self.setup(hand=["奉献"], theirs=["精灵龙"])
        game.step(Action(PLAY, 0))
        self.assertEqual(game.boards[1], [])

    def test_swipe_splash_still_hits_elusive(self):
        """横扫指定另一个随从时，溅射照样打到扰咒的那个。"""
        game = self.setup(hand=["横扫"], theirs=["冰风雪人", "精灵龙"])
        game.step(Action(PLAY, 0, 0))          # 指定冰风雪人，精灵龙吃 1 点溅射
        elusive = [m for m in game.boards[1] if m.name == "精灵龙"]
        self.assertEqual(len(elusive), 1)
        self.assertEqual(elusive[0].health, 2 - 1)

    def test_missiles_can_hit_elusive(self):
        """随机伤害不是指定，扰咒挡不住——场上只有它时飞弹会落在它或脸上。"""
        game = self.setup(hand=["奥术飞弹"], theirs=["精灵龙"], enemy_health=30)
        before = game.boards[1][0].health + game.hero_health[1]
        game.step(Action(PLAY, 0))
        after = sum(m.health for m in game.boards[1]) + game.hero_health[1]
        self.assertLess(after, before, "3 点飞弹总得打在某个地方")


class TestSpellPower(Fixture):
    def test_no_bonus_without_the_keyword(self):
        game = self.setup(hand=["火球术"], theirs=["石拳食人魔"])   # 6/7
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 6)

    def test_one_spell_damage_minion_adds_one(self):
        """6 点火球打不死 6/7，7 点正好打死——这个临界点就是加成生效的证明。"""
        game = self.setup(hand=["火球术"], mine=["狗头人地卜师"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1], [])

    def test_bonus_stacks(self):
        """两个法术增强 = +2。用奉献（2 点）才看得出数值，火球加到 8 会溢出。"""
        game = self.setup(hand=["奉献"], mine=["狗头人地卜师", "达拉然法师"],
                          theirs=["石拳食人魔"])
        self.assertEqual(game.spell_power(0), 2)
        game.step(Action(PLAY, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 4)

    def test_bonus_applies_to_face(self):
        game = self.setup(hand=["火球术"], mine=["狗头人地卜师"])
        game.step(Action(PLAY, 0))
        self.assertEqual(game.hero_health[1], 30 - 7)

    def test_bonus_applies_to_aoe(self):
        game = self.setup(hand=["奉献"], mine=["狗头人地卜师"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0))                       # 敌方全体 2+1
        self.assertEqual(game.boards[1][0].health, 7 - 3)
        self.assertEqual(game.hero_health[1], 30 - 3)

    def test_bonus_applies_to_splash(self):
        game = self.setup(hand=["横扫"], mine=["狗头人地卜师"],
                          theirs=["石拳食人魔", "绿洲钳嘴龟"])
        game.step(Action(PLAY, 0, 0))                    # 主目标 4+1，其余 1+1
        self.assertEqual(game.boards[1][0].health, 7 - 5)
        self.assertEqual(game.boards[1][1].health, 7 - 2)

    def test_bonus_adds_a_missile_not_damage_per_missile(self):
        """按炉石的规矩：奥术飞弹 +1 是多一颗飞弹，总伤害同样 +1。"""
        game = self.setup(hand=["奥术飞弹"], mine=["狗头人地卜师"], enemy_health=30)
        game.step(Action(PLAY, 0))
        self.assertEqual(game.hero_health[1], 30 - 4, "场上没随从时 4 颗全打脸")

    def test_bonus_does_not_affect_transform(self):
        game = self.setup(hand=["变形术"], mine=["狗头人地卜师"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].name, "绵羊")
        self.assertEqual((game.boards[1][0].attack, game.boards[1][0].health), (1, 1))

    def test_bonus_does_not_affect_draw(self):
        game = self.setup(hand=["奥术智慧"], mine=["狗头人地卜师"])
        game.decks[0] = [card("幽灵")] * 5
        game.step(Action(PLAY, 0))
        self.assertEqual(len(game.hands[0]), 2, "抽 2 张就是 2 张，不受法术增强影响")

    def test_enemy_spell_power_does_not_help_me(self):
        game = self.setup(hand=["火球术"], theirs=["狗头人地卜师", "石拳食人魔"])
        self.assertEqual(game.spell_power(0), 0)
        self.assertEqual(game.spell_power(1), 1)
        game.step(Action(PLAY, 0, 1))
        self.assertEqual(game.boards[1][1].health, 7 - 6)

    def test_transformed_minion_loses_spell_power(self):
        """变成 1/1 绵羊之后关键词全没了——对手的法术增强也就没了。"""
        game = self.setup(hand=["变形术"], theirs=["狗头人地卜师"])
        self.assertEqual(game.spell_power(1), 1)
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].name, "绵羊")
        self.assertEqual(game.spell_power(1), 0)


class TestKeywordsAreNoLongerInert(unittest.TestCase):
    def test_inert_list_is_empty(self):
        from hearthstone.cards import INERT_KEYWORDS
        self.assertEqual(INERT_KEYWORDS, ())

    def test_pool_still_has_both_keywords(self):
        self.assertTrue([c for c in POOL if c.has(ELUSIVE)])
        self.assertTrue([c for c in POOL if c.has(SPELL_DAMAGE)])


if __name__ == "__main__":
    unittest.main()
