"""触发：随从在场期间监听事件。

事件类型和"谁的事件才算"是**正交的两个轴**——"受到伤害"要区分是本随从受伤
（苦痛侍僧）还是任意随从受伤（暴乱狂战士）。这组测试把每个组合都钉一遍，
外加三条容易出事的性质：触发顺序、遍历时场面变化、触发链深度。
"""

import random
import unittest

from hearthstone.cards import CARD_INDEX, POOL, CardDef
from hearthstone.game import END_TURN, HERO, PLAY, Action, Game, Minion, attack


def card(name: str) -> CardDef:
    return POOL[CARD_INDEX[name]]


class Fixture(unittest.TestCase):
    def setup(self, hand=(), mine=(), theirs=(), mana=10, deck=8):
        game = Game(rng=random.Random(0))
        game.current = 0
        game.mana = [mana, mana]
        game.max_mana = [mana, mana]
        game.hands = [[card(n) for n in hand], []]
        game.decks = [[card("幽灵")] * deck, [card("幽灵")] * deck]
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


class TestTakeDamage(Fixture):
    def test_self_source_only_fires_for_itself(self):
        """苦痛侍僧：本随从受伤才抽牌。"""
        game = self.setup(mine=["苦痛侍僧", "冰风雪人"])
        before = len(game.hands[0])
        game._hit(game.boards[0][1], 1)          # 打的是雪人
        self.assertEqual(len(game.hands[0]), before, "别人受伤不该触发")
        game._hit(game.boards[0][0], 1)          # 打的是侍僧本人
        self.assertEqual(len(game.hands[0]), before + 1)

    def test_all_minions_source_fires_for_anyone(self):
        """暴乱狂战士：任意随从受伤都 +1 攻击力，敌我都算。"""
        game = self.setup(mine=["暴乱狂战士"], theirs=["冰风雪人"])
        berserker = game.boards[0][0]
        self.assertEqual(berserker.attack, 2)
        game._hit(game.boards[1][0], 1)          # 敌方随从受伤
        self.assertEqual(berserker.attack, 3)

    def test_divine_shield_absorbs_so_no_trigger(self):
        """圣盾把伤害整个吃掉，不算"受到伤害"。"""
        game = self.setup(mine=["苦痛侍僧"])
        game.boards[0][0].divine_shield = True
        before = len(game.hands[0])
        game._hit(game.boards[0][0], 3)
        self.assertEqual(len(game.hands[0]), before)


class TestDealDamage(Fixture):
    def test_water_elemental_freezes_what_it_hits(self):
        game = self.setup(mine=["水元素"], theirs=["石拳食人魔"])
        game.step(attack(0, 0))
        self.assertTrue(game.boards[1][0].frozen)

    def test_water_elemental_is_not_frozen_by_the_exchange(self):
        """只冻结被它打的那个。用能让水元素活下来的对手，不然测不到。"""
        game = self.setup(mine=["水元素"], theirs=["绿洲钳嘴龟"])   # 3/6 打 2/7
        game.step(attack(0, 0))
        self.assertFalse(game.boards[0][0].frozen)
        self.assertTrue(game.boards[1][0].frozen)

    def test_dying_attacker_still_freezes(self):
        """水元素撞死在大随从身上，照样把对方冻住——炉石里伤害是同时结算的。"""
        game = self.setup(mine=["水元素"], theirs=["石拳食人魔"])   # 3/6 撞 6/7
        game.step(attack(0, 0))
        self.assertEqual(game.boards[0], [], "水元素被 6 点打死")
        self.assertTrue(game.boards[1][0].frozen, "死了也要冻上")

    def test_cobra_destroys_what_it_hits(self):
        game = self.setup(mine=["帝王眼镜蛇"], theirs=["石拳食人魔"])   # 2/3 打 6/7
        game.step(attack(0, 0))
        self.assertEqual(game.boards[1], [], "被它打到的随从直接死")

    def test_no_trigger_when_damage_is_absorbed(self):
        game = self.setup(mine=["水元素"], theirs=["石拳食人魔"])
        game.boards[1][0].divine_shield = True
        game.step(attack(0, 0))
        self.assertFalse(game.boards[1][0].frozen, "圣盾挡掉就没造成伤害")


class TestTurnTriggers(Fixture):
    def test_doomsayer_wipes_at_my_turn_start(self):
        game = self.setup(mine=["末日预言者"], theirs=["冰风雪人"])
        game.step(END_TURN)                      # 轮到对手，不该触发
        self.assertEqual(len(game.boards[0]), 1)
        game.step(END_TURN)                      # 回到自己回合开始 → 清场
        self.assertEqual(game.boards[0], [])
        self.assertEqual(game.boards[1], [])

    def test_my_turn_only_does_not_fire_on_opponent_turn(self):
        game = self.setup(mine=["末日预言者"], theirs=["冰风雪人"])
        game.step(END_TURN)
        self.assertEqual(len(game.boards[1]), 1, "对手回合开始时不该触发我的预言者")


class TestCastSpellTrigger(Fixture):
    def test_pyromancer_fires_after_a_spell(self):
        game = self.setup(hand=["奥术智慧"], mine=["狂野炎术师", "幽灵"],
                          theirs=["幽灵"])
        game.step(Action(PLAY, 0))               # 抽 2 张 → 触发全场 1 点
        self.assertEqual(game.boards[1], [], "1/1 被打死")
        names = [m.name for m in game.boards[0]]
        self.assertNotIn("幽灵", names, "自己的 1/1 也会死")
        self.assertIn("狂野炎术师", names, "3/2 挨 1 点还活着")


class TestTriggerSafety(Fixture):
    def test_trigger_order_follows_board_order(self):
        game = self.setup(mine=["苦痛侍僧", "苦痛侍僧"])
        before = len(game.hands[0])
        game._hit(game.boards[0][0], 1)
        self.assertEqual(len(game.hands[0]), before + 1, "只有被打的那个抽牌")

    def test_dead_listener_does_not_fire(self):
        """遍历过程中被打死的随从不能再触发。"""
        game = self.setup(mine=["暴乱狂战士"], theirs=["幽灵"])
        game.boards[0][0].damage = 3             # 4 血只剩 1
        game._hit(game.boards[0][0], 1)          # 自己被打死
        game._clear_dead()
        self.assertEqual(game.boards[0], [])

    def test_trigger_chain_terminates(self):
        """两个互相触发的随从不能把引擎卡死。"""
        game = self.setup(mine=["暴乱狂战士", "暴乱狂战士"],
                          theirs=["暴乱狂战士"])
        game._hit(game.boards[1][0], 1)          # 触发三个狂战士各 +1
        self.assertLessEqual(game._trigger_depth, 8)
        for m in game.boards[0]:
            self.assertGreaterEqual(m.attack, 3)

    def test_clone_resets_trigger_depth(self):
        game = self.setup(mine=["苦痛侍僧"])
        game._trigger_depth = 5
        self.assertEqual(game.clone()._trigger_depth, 0)


class TestFullGamesStillRun(unittest.TestCase):
    def test_many_games_no_crash(self):
        from hearthstone.bots import make_bot
        from hearthstone.game import play_game
        rng = random.Random(0)
        for i in range(120):
            play_game([make_bot("rule", seed=i), make_bot("rule", seed=i + 1)], rng=rng)


if __name__ == "__main__":
    unittest.main()
