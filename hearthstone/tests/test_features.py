import random
import unittest

import numpy as np

from hearthstone.features import (
    FEATURE_DIM,
    ORACLE_DIM,
    STATE_DIM,
    STATE_OFFSET,
    action_features,
    batch_features,
    oracle_features,
    state_features,
)
from hearthstone.game import END_TURN, Game, attack, play


class TestDimensions(unittest.TestCase):
    def test_state_offset_consistent(self):
        self.assertEqual(STATE_OFFSET + STATE_DIM, FEATURE_DIM)

    def test_batch_features_shape(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        mat = batch_features(obs)
        self.assertEqual(mat.shape, (len(obs.legal), FEATURE_DIM))

    def test_batch_features_is_float32(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        self.assertEqual(batch_features(obs).dtype, np.float32)


class TestActionFeatures(unittest.TestCase):
    def test_end_action(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = action_features(obs, END_TURN)
        self.assertEqual(len(feats), STATE_OFFSET)
        # end is one-hot: [0, 0, 1]
        self.assertEqual(feats[:3], [0.0, 0.0, 1.0])

    def test_play_action(self):
        game = Game(rng=random.Random(1))
        game.hands[0] = game.hands[0][:1]  # keep just one card we can afford
        game.mana[0] = 10
        obs = game.observe()
        play_acts = [a for a in obs.legal if a.kind == "play"]
        if play_acts:
            feats = action_features(obs, play_acts[0])
            self.assertEqual(len(feats), STATE_OFFSET)
            self.assertEqual(feats[:3], [1.0, 0.0, 0.0])

    def test_attack_action(self):
        game = Game(rng=random.Random(2))
        # give player 0 a charge minion and player 1 something to attack
        from hearthstone.cards import CardDef

        charger = CardDef("c", 1, 2, 1, ("冲锋",))
        game.hands[0] = [charger]
        game.mana[0] = 10
        game.step(play(0))
        obs = game.observe()
        atk_acts = [a for a in obs.legal if a.kind == "attack"]
        if atk_acts:
            feats = action_features(obs, atk_acts[0])
            self.assertEqual(len(feats), STATE_OFFSET)
            self.assertEqual(feats[:3], [0.0, 1.0, 0.0])


class TestStateFeatures(unittest.TestCase):
    def test_state_features_length(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = state_features(obs)
        self.assertEqual(len(feats), STATE_DIM)

    def test_state_constant_across_actions(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        mat = batch_features(obs)
        if len(obs.legal) > 1:
            for i in range(1, len(obs.legal)):
                np.testing.assert_array_equal(
                    mat[0, STATE_OFFSET:], mat[i, STATE_OFFSET:]
                )

    def test_state_bias_is_one(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = state_features(obs)
        self.assertEqual(feats[-1], 1.0)

    def test_board_slots_padding(self):
        """少于 7 个随从时后面补零。"""
        game = Game(rng=random.Random(0))
        game.boards[0] = []  # empty board
        game.boards[1] = []
        obs = game.observe()
        feats = state_features(obs)
        from hearthstone.features import S_BASE, S_WEAPON, S_HAND, S_HAND_CARDS, S_SPELLS, S_BOARD, S_BOARD_SLOTS
        slot_start = S_BASE + S_WEAPON + S_HAND + S_HAND_CARDS + S_SPELLS + S_BOARD
        board_slots = feats[slot_start:slot_start + S_BOARD_SLOTS]
        self.assertEqual(board_slots, [0.0] * S_BOARD_SLOTS)

    def test_board_slots_sorted_by_uid(self):
        """逐随从编码按出场顺序（uid）排列，7 槽 7 维。"""
        from hearthstone.cards import CardDef
        from hearthstone.game import Minion

        game = Game(rng=random.Random(0))
        # 造三个随从，乱序放入场上
        m1 = Minion.summon(CardDef("a", 1, 1, 1, ()), uid=10)
        m2 = Minion.summon(CardDef("b", 1, 5, 1, ()), uid=5)
        m3 = Minion.summon(CardDef("c", 1, 3, 1, ("剧毒",)), uid=20)
        game.boards[0] = [m1, m2, m3]  # attack: 1, 5, 3; uid: 10, 5, 20
        game.boards[1] = []
        obs = game.observe()
        feats = state_features(obs)

        from hearthstone.features import S_BASE, S_WEAPON, S_HAND, S_HAND_CARDS, S_SPELLS, S_BOARD
        slot_start = S_BASE + S_WEAPON + S_HAND + S_HAND_CARDS + S_SPELLS + S_BOARD
        STRIDE = 7
        my_slots = feats[slot_start:slot_start + STRIDE * 7]
        # uid 升序：5, 10, 20 → slot 0 = uid=5 (atk=5), slot 1 = uid=10 (atk=1), slot 2 = uid=20 (atk=3)
        self.assertAlmostEqual(my_slots[0], 5 / 10.0)              # uid=5 atk
        self.assertAlmostEqual(my_slots[STRIDE], 1 / 10.0)         # uid=10 atk
        self.assertAlmostEqual(my_slots[STRIDE * 2], 3 / 10.0)     # uid=20 atk
        # slot 3~6 补零
        for i in range(3, 7):
            self.assertEqual(my_slots[STRIDE * i:STRIDE * (i + 1)], [0.0] * STRIDE)
        # uid=20 有剧毒标志位
        self.assertEqual(my_slots[STRIDE * 2 + 5], 1.0)  # poisonous

    def test_hand_cards_sorted_by_cost(self):
        """手牌逐卡编码按费用升序排列。"""
        from hearthstone.cards import CardDef

        game = Game(rng=random.Random(0))
        # 放几张不同费用的牌到手牌
        game.hands[0] = [
            CardDef("high", 8, 8, 8),
            CardDef("low", 1, 1, 1),
            CardDef("mid", 4, 4, 4),
        ]
        game.mana[0] = 10
        obs = game.observe()
        feats = state_features(obs)

        from hearthstone.features import S_BASE, S_WEAPON, S_HAND
        card_start = S_BASE + S_WEAPON + S_HAND
        # 前 3 张可出牌（每张 5 维），第一张应为费用 1
        cards = feats[card_start:card_start + 15]
        self.assertAlmostEqual(cards[0], 1 / 10.0)   # cost of cheapest
        self.assertAlmostEqual(cards[5], 4 / 10.0)   # cost of second
        self.assertAlmostEqual(cards[10], 8 / 10.0)  # cost of third

    def test_going_first_from_obs(self):
        """先后手直接从 Observation 读取，不用推断。"""
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = state_features(obs)
        # going_first 在 bias(1.0) 之前
        self.assertIn(feats[-2], (0.0, 1.0))
        # 验证与 Observation 一致
        from hearthstone.features import S_BASE, S_WEAPON, S_HAND, S_HAND_CARDS, S_SPELLS, S_BOARD, S_BOARD_SLOTS, S_KEYWORDS, S_LETHAL
        gf_offset = S_BASE + S_WEAPON + S_HAND + S_HAND_CARDS + S_SPELLS + S_BOARD + S_BOARD_SLOTS + S_KEYWORDS + S_LETHAL + 4  # deck/en_hand/en_fatigue = 3, then going_first
        self.assertEqual(feats[gf_offset], 1.0 if obs.going_first else 0.0)


class TestOracleIsolation(unittest.TestCase):
    """先知特征绝不能泄漏进策略看到的特征里。

    这组测试守的是整个非对称 actor-critic 的正确性前提：策略只吃
    `batch_features`，价值头才吃 `oracle_features`。一旦有人把 `enemy_hand`
    接进 `_state_features`，这里会立刻红。
    """

    def _obs_pair(self):
        """同一个局面，只有对手手牌不同的两个观测。"""
        game = Game(rng=random.Random(7))
        a = game.observe()
        b = game.observe()
        self.assertTrue(a.enemy_hand, "这个局面下对手应该有手牌，测试才有意义")
        b.enemy_hand = []
        return a, b

    def test_policy_features_ignore_enemy_hand(self):
        a, b = self._obs_pair()
        np.testing.assert_array_equal(batch_features(a), batch_features(b))

    def test_state_features_ignore_enemy_hand(self):
        a, b = self._obs_pair()
        self.assertEqual(state_features(a), state_features(b))

    def test_oracle_features_do_react_to_enemy_hand(self):
        """反过来，先知特征必须真的随对手手牌变化——否则它没起作用。"""
        a, b = self._obs_pair()
        self.assertFalse(np.array_equal(oracle_features(a), oracle_features(b)))

    def test_oracle_shape_and_dtype(self):
        game = Game(rng=random.Random(0))
        vec = oracle_features(game.observe())
        self.assertEqual(vec.shape, (ORACLE_DIM,))
        self.assertEqual(vec.dtype, np.float32)

    def test_oracle_empty_hand_is_zeros(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        obs.enemy_hand = []
        np.testing.assert_array_equal(
            oracle_features(obs), np.zeros(ORACLE_DIM, dtype=np.float32)
        )

    def test_oracle_features_finite(self):
        """扫一整局，先知特征不能出现 nan/inf。"""
        from hearthstone.bots import make_bot

        game = Game(rng=random.Random(3))
        bot = make_bot("rule", seed=3)
        while not game.finished:
            obs = game.observe()
            self.assertTrue(np.all(np.isfinite(oracle_features(obs))))
            game.step(bot.choose(obs))


if __name__ == "__main__":
    unittest.main()
