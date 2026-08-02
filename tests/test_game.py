import random
import unittest

from paodekuai.bots import make_bot
from paodekuai.cards import DIAMOND_THREE, parse_card
from paodekuai.combos import classify
from paodekuai.game import Game, play_game


def cards(*names):
    return [parse_card(name) for name in names]


class TestGameSetup(unittest.TestCase):
    def setUp(self):
        self.game = Game(rng=random.Random(0))

    def test_deals_three_hands_of_16(self):
        self.assertEqual([len(hand) for hand in self.game.hands], [16, 16, 16])

    def test_player_with_diamond_three_starts(self):
        self.assertIn(DIAMOND_THREE, self.game.hands[self.game.current])

    def test_first_move_must_contain_diamond_three(self):
        for move in self.game.legal_actions():
            self.assertIsNotNone(move)
            self.assertIn(DIAMOND_THREE, move.cards)

    def test_cannot_pass_when_leading(self):
        self.assertNotIn(None, self.game.legal_actions())
        with self.assertRaises(ValueError):
            self.game.step(None)

    def test_rejects_move_without_diamond_three(self):
        starter = self.game.current
        other = next(c for c in self.game.hands[starter] if c != DIAMOND_THREE)
        with self.assertRaises(ValueError):
            self.game.step(classify([other]))


class TestGameFlow(unittest.TestCase):
    """用手工摆好的牌局验证轮转规则。"""

    def setUp(self):
        self.game = Game(rng=random.Random(0))
        self.game.hands = [
            cards("3", "4", "5"),           # 玩家0 拿着 ♦3，先手
            cards("c9", "h9", "6"),
            cards("cK", "hK", "7"),
        ]
        self.game.current = 0
        self.game.required = None
        self.game.leader = None
        self.game.passes = 0
        self.game.first_move = True

    def test_play_moves_to_next_player(self):
        self.game.step(classify(cards("3")))
        self.assertEqual(self.game.current, 1)
        self.assertEqual(self.game.leader, 0)
        self.assertEqual(self.game.required.rank, 3)

    def test_cards_leave_the_hand_and_become_public(self):
        self.game.step(classify(cards("3")))
        self.assertEqual(len(self.game.hands[0]), 2)
        self.assertEqual(self.game.played_counts[3], 1)

    def test_everyone_passing_returns_the_lead(self):
        self.game.step(classify(cards("3")))   # 玩家0 出 3
        self.game.step(None)                   # 玩家1 过
        self.game.step(None)                   # 玩家2 过
        self.assertEqual(self.game.current, 0)
        self.assertIsNone(self.game.required)
        self.assertEqual(self.game.passes, 0)

    def test_lead_goes_to_whoever_played_last(self):
        self.game.step(classify(cards("3")))   # 玩家0
        self.game.step(classify(cards("6")))   # 玩家1 压
        self.game.step(None)                   # 玩家2 过
        self.game.step(None)                   # 玩家0 过
        self.assertEqual(self.game.current, 1)
        self.assertIsNone(self.game.required)

    def test_cannot_play_a_smaller_card(self):
        self.game.first_move = False
        self.game.required = classify(cards("cK"))
        self.game.leader = 2
        self.game.current = 0
        with self.assertRaises(ValueError):
            self.game.step(classify(cards("5")))

    def test_cannot_answer_with_a_different_kind(self):
        self.game.first_move = False
        self.game.required = classify(cards("c9", "h9"))  # 一对 9
        self.game.leader = 1
        self.game.current = 2
        with self.assertRaises(ValueError):
            self.game.step(classify(cards("cK")))  # 单张压不了对子

    def test_cannot_play_cards_you_do_not_have(self):
        with self.assertRaises(ValueError):
            self.game.step(classify(cards("sA")))

    def test_emptying_your_hand_wins(self):
        self.game.hands[0] = cards("3")
        self.game.step(classify(cards("3")))
        self.assertTrue(self.game.finished)
        self.assertEqual(self.game.winner, 0)
        self.assertEqual(self.game.result().remaining[0], 0)

    def test_cannot_act_after_the_game_ends(self):
        self.game.hands[0] = cards("3")
        self.game.step(classify(cards("3")))
        with self.assertRaises(RuntimeError):
            self.game.step(None)


class TestObservation(unittest.TestCase):
    def test_unseen_counts_exclude_own_hand_and_played_cards(self):
        game = Game(rng=random.Random(3))
        obs = game.observe()
        unseen = obs.unseen_counts()
        self.assertEqual(sum(unseen.values()), 32)  # 48 - 自己的 16 张
        self.assertTrue(all(count >= 0 for count in unseen.values()))

        game.step(game.legal_actions()[0])
        obs = game.observe(0)
        self.assertEqual(sum(obs.unseen_counts().values()), 48 - len(obs.hand) - sum(obs.played_counts.values()))

    def test_opponents_excludes_self(self):
        obs = Game(rng=random.Random(0)).observe()
        self.assertEqual(len(obs.opponents()), 2)
        self.assertNotIn(obs.player, obs.opponents())


class TestFullGames(unittest.TestCase):
    def test_games_always_terminate_with_one_winner(self):
        rng = random.Random(5)
        for _ in range(60):
            players = [make_bot(name) for name in ("random", "greedy", "rule")]
            result = play_game(players, rng=rng)
            self.assertIn(result.winner, (0, 1, 2))
            self.assertEqual(result.remaining[result.winner], 0)
            self.assertEqual(sum(1 for n in result.remaining if n == 0), 1)
            self.assertLess(result.turns, 400)

    def test_verbose_log_records_every_action(self):
        result = play_game([make_bot("greedy") for _ in range(3)], rng=random.Random(0), verbose=True)
        self.assertTrue(any("获胜" in line for line in result.log))
        self.assertTrue(any("起手" in line for line in result.log))


if __name__ == "__main__":
    unittest.main()
