import random
import unittest

import play
from paodekuai.cards import parse_card
from paodekuai.combos import classify
from paodekuai.game import Game
from paodekuai.policy import MoveScorer, PolicyAgent
from tests.test_bots import observation


def cards(*names):
    return [parse_card(name) for name in names]


class TestParseRanks(unittest.TestCase):
    def test_space_separated(self):
        self.assertEqual(play.parse_ranks("3 3 3 4"), [3, 3, 3, 4])

    def test_compact(self):
        self.assertEqual(play.parse_ranks("333 4"), [3, 3, 3, 4])
        self.assertEqual(play.parse_ranks("JQKA"), [11, 12, 13, 14])

    def test_ten_is_two_characters(self):
        self.assertEqual(play.parse_ranks("10"), [10])
        self.assertEqual(play.parse_ranks("1010"), [10, 10])
        self.assertEqual(play.parse_ranks("10 J Q K A"), [10, 11, 12, 13, 14])

    def test_case_and_commas(self):
        self.assertEqual(play.parse_ranks("j,q,k"), [11, 12, 13])

    def test_empty(self):
        self.assertEqual(play.parse_ranks("   "), [])

    def test_garbage_is_rejected(self):
        with self.assertRaises(ValueError):
            play.parse_ranks("3 x")
        with self.assertRaises(ValueError):
            play.parse_ranks("2")  # 2 不在这个变体的牌堆里


class TestFindMove(unittest.TestCase):
    def test_finds_a_legal_move_by_rank(self):
        obs = observation(cards("9", "c9", "h9", "3"))
        move = play.find_move("9 9 9 3", obs)
        self.assertIsNotNone(move)
        self.assertEqual(move.kind, "triple_one")
        self.assertIn(move, obs.legal)

    def test_suits_do_not_matter(self):
        obs = observation(cards("K", "cK"))
        self.assertEqual(play.find_move("k k", obs).kind, "pair")

    def test_rejects_cards_you_do_not_have(self):
        obs = observation(cards("3", "4", "5"))
        self.assertIsNone(play.find_move("A A", obs))

    def test_rejects_an_illegal_shape(self):
        obs = observation(cards("3", "5", "9"))
        self.assertIsNone(play.find_move("3 9", obs))  # 不成牌型

    def test_rejects_a_move_that_cannot_beat(self):
        obs = observation(cards("3", "4", "5"), required=classify(cards("A")))
        self.assertIsNone(play.find_move("3", obs))

    def test_empty_input_finds_nothing(self):
        self.assertIsNone(play.find_move("", observation(cards("3"))))


class TestExplain(unittest.TestCase):
    def test_returns_scores_sorted_by_probability(self):
        agent = PolicyAgent(MoveScorer(hidden=16))
        obs = Game(rng=random.Random(0)).observe()
        rows = play.explain(agent, obs, top=3)

        self.assertLessEqual(len(rows), 3)
        probs = [prob for _, _, prob in rows]
        self.assertEqual(probs, sorted(probs, reverse=True))
        for move, _, prob in rows:
            self.assertIn(move, obs.legal)
            self.assertTrue(0.0 <= prob <= 1.0)

    def test_probabilities_cover_every_candidate(self):
        agent = PolicyAgent(MoveScorer(hidden=16))
        obs = Game(rng=random.Random(4)).observe()
        rows = play.explain(agent, obs, top=len(obs.legal))
        self.assertAlmostEqual(sum(prob for _, _, prob in rows), 1.0, places=4)


class TestPad(unittest.TestCase):
    def test_pads_ascii_to_the_requested_width(self):
        self.assertEqual(len(play.pad("abc", 10)), 10)

    def test_counts_cjk_as_two_columns(self):
        # "玩家0" 显示宽度是 5，补到 8 需要 3 个空格
        self.assertEqual(play.pad("玩家0", 8), "玩家0   ")

    def test_never_truncates(self):
        self.assertEqual(play.pad("很长很长很长", 2), "很长很长很长")


if __name__ == "__main__":
    unittest.main()
