"""条件效果。

关键在**求值时机**：

- `condition` 在本体结算**之前**判（"如果你装备着武器，造成 2 点伤害"）
- `then.condition` 在本体结算**之后**判（"造成 1 点伤害。如果该随从死亡，抽一张牌"）

后者还有个坑：目标死了会被清出棋盘，条件求值时已经找不到它了，所以要在结算前
抓住引用回头看。
"""

import random
import unittest

from hearthstone.cards import CARD_INDEX, POOL, CardDef
from hearthstone.game import HERO, PLAY, Action, Game, Minion


def card(name: str) -> CardDef:
    return POOL[CARD_INDEX[name]]


class Fixture(unittest.TestCase):
    def setup(self, hand=(), mine=(), theirs=(), mana=10, hp=30, deck=8):
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
        game.hero_health = [hp, 30]
        game._refresh_auras()
        return game


class TestConditionAfterResolution(Fixture):
    """then.condition —— 本体打完才有答案。"""

    def test_draw_when_target_dies(self):
        game = self.setup(hand=["死亡缠绕"], theirs=["幽灵"])       # 1 点打死 1/1
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1], [])
        self.assertEqual(len(game.hands[0]), 1, "目标死了，抽一张")

    def test_no_draw_when_target_survives(self):
        game = self.setup(hand=["死亡缠绕"], theirs=["石拳食人魔"])  # 1 点打不死 6/7
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.hands[0]), 0, "没打死就不抽")

    def test_the_opposite_condition(self):
        """猛击：打不死才抽牌。"""
        game = self.setup(hand=["猛击"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.hands[0]), 1)

    def test_the_opposite_condition_negative(self):
        game = self.setup(hand=["猛击"], theirs=["幽灵"])           # 2 点打死 1/1
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.hands[0]), 0)

    def test_target_removed_from_board_still_counts_as_dead(self):
        """死了的随从已经被清出棋盘，条件求值时要靠结算前抓的引用。"""
        game = self.setup(hand=["死亡缠绕"], theirs=["幽灵"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.boards[1]), 0)
        self.assertEqual(len(game.hands[0]), 1)


class TestConditionBeforeResolution(Fixture):
    def test_hand_empty_condition(self):
        game = self.setup(hand=["快速射击"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0, 0))               # 打完手牌就空了
        self.assertEqual(len(game.hands[0]), 1, "手牌空 → 抽一张")

    def test_hand_not_empty(self):
        game = self.setup(hand=["快速射击", "幽灵"], theirs=["石拳食人魔"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.hands[0]), 1, "还剩幽灵，不抽")

    def test_weapon_condition_met(self):
        game = self.setup(hand=["雾帆劫掠者"], theirs=["幽灵"])
        game.weapons[0] = card("炽炎战斧")
        game.weapon_durability[0] = 2
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1], [], "有武器 → 战吼造成 2 点，打死 1/1")

    def test_weapon_condition_not_met(self):
        game = self.setup(hand=["雾帆劫掠者"], theirs=["幽灵"])
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.boards[1]), 1, "没武器 → 战吼不生效")

    def test_broken_weapon_does_not_count(self):
        game = self.setup(hand=["雾帆劫掠者"], theirs=["幽灵"])
        game.weapons[0] = card("炽炎战斧")
        game.weapon_durability[0] = 0               # 耐久耗尽
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(len(game.boards[1]), 1)


class TestOtherwise(Fixture):
    """"如果…则改为…" —— 条件不成立时走另一支。"""

    def test_low_health_takes_the_stronger_branch(self):
        game = self.setup(hand=["致死打击"], theirs=["石拳食人魔"], hp=10)
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 6, "血量 ≤12 → 6 点")

    def test_high_health_takes_the_default_branch(self):
        game = self.setup(hand=["致死打击"], theirs=["石拳食人魔"], hp=30)
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 4, "血量 >12 → 4 点")

    def test_boundary_is_inclusive(self):
        game = self.setup(hand=["致死打击"], theirs=["石拳食人魔"], hp=12)
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 6, "小于等于，12 也算")


class TestSpellPowerStillApplies(Fixture):
    def test_conditional_branch_gets_spell_power(self):
        game = self.setup(hand=["致死打击"], mine=["狗头人地卜师"],
                          theirs=["石拳食人魔"], hp=30)
        game.step(Action(PLAY, 0, 0))
        self.assertEqual(game.boards[1][0].health, 7 - 5, "4 点 + 法术增强 1")


class TestFullGames(unittest.TestCase):
    def test_games_still_run(self):
        from hearthstone.bots import make_bot
        from hearthstone.game import play_game
        rng = random.Random(0)
        for i in range(120):
            play_game([make_bot("rule", seed=i), make_bot("rule", seed=i + 1)], rng=rng)


if __name__ == "__main__":
    unittest.main()
