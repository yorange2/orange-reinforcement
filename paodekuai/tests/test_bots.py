import random
import unittest

from paodekuai.arena import bot_vs_bot, evaluate, final_reward, match
from paodekuai.bots import BOTS, make_bot
from paodekuai.cards import parse_card
from paodekuai.combos import BOMB, classify
from paodekuai.game import Game, GameResult, Observation


def cards(*names):
    return [parse_card(name) for name in names]


def observation(hand, required=None, hand_sizes=None, legal=None, player=0):
    from paodekuai.combos import legal_moves

    moves = list(legal_moves(hand, required)) if legal is None else list(legal)
    if required is not None and None not in moves:
        moves.append(None)
    return Observation(
        player=player,
        hand=list(hand),
        hand_sizes=hand_sizes or [len(hand), 10, 10],
        required=required,
        leader=None if required is None else 1,
        played_counts={},
        legal=moves,
        trick=0,
    )


class TestBotsPlayLegally(unittest.TestCase):
    def test_every_bot_only_returns_legal_actions(self):
        rng = random.Random(0)
        for name in BOTS:
            for _ in range(40):
                game = Game(rng=rng)
                while not game.finished:
                    obs = game.observe()
                    action = make_bot(name, seed=1).choose(obs)
                    self.assertIn(action, obs.legal, f"{name} 给出了非法动作")
                    game.step(action)

    def test_make_bot_rejects_unknown_names(self):
        with self.assertRaises(ValueError):
            make_bot("no-such-bot")


class TestGreedyBot(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot("greedy")

    def test_plays_the_smallest_card_when_leading(self):
        move = self.bot.choose(observation(cards("3", "9", "A")))
        self.assertEqual(move.rank, 3)

    def test_plays_the_smallest_card_that_beats(self):
        move = self.bot.choose(observation(cards("5", "9", "A"), required=classify(cards("4"))))
        self.assertEqual(move.rank, 5)

    def test_passes_when_it_cannot_beat(self):
        move = self.bot.choose(observation(cards("3", "4"), required=classify(cards("A"))))
        self.assertIsNone(move)


class TestRuleBot(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot("rule")

    def test_goes_out_when_it_can_finish(self):
        move = self.bot.choose(observation(cards("9", "c9")))
        self.assertEqual(len(move.cards), 2)

    def test_saves_bombs_against_a_harmless_lead(self):
        hand = cards("9", "c9", "h9", "s9", "4", "6", "8", "10")
        move = self.bot.choose(observation(hand, required=classify(cards("K"))))
        self.assertIsNone(move, "对手没威胁时不该拆炸弹")

    def test_uses_the_bomb_when_an_opponent_is_about_to_go_out(self):
        hand = cards("9", "c9", "h9", "s9", "4")
        obs = observation(hand, required=classify(cards("K")), hand_sizes=[5, 1, 8])
        self.assertEqual(self.bot.choose(obs).kind, BOMB)

    def test_plays_the_cheapest_card_that_beats(self):
        hand = cards("A", "3", "5", "7", "9", "J")
        move = self.bot.choose(observation(hand, required=classify(cards("4"))))
        self.assertEqual(move.rank, 5, "有小牌能压就别动大牌")

    def test_does_not_waste_a_king_on_a_small_lead(self):
        # 能压 ♣4 的只剩 K 和 A，手牌还有 6 张，这时候宁可过牌留着大牌
        hand = cards("3", "c3", "h3", "4", "K", "A")
        move = self.bot.choose(observation(hand, required=classify(cards("c4"))))
        self.assertIsNone(move, "手牌还多时不该用 K/A 压小牌")

    def test_prefers_not_to_break_up_a_straight(self):
        # 3-4-5-6-7 是一手顺子，不该为了压一张 3 把它拆了
        hand = cards("3", "4", "5", "6", "7", "K")
        move = self.bot.choose(observation(hand, required=classify(cards("c3"))))
        self.assertNotIn(move, [classify(cards("4")), classify(cards("5"))])


class TestBotStrengthOrdering(unittest.TestCase):
    """规则对手要有明确的强弱梯度，模型的胜率才有参照意义。"""

    def test_rule_beats_greedy_beats_random(self):
        rule_vs_greedy = bot_vs_bot("rule", "greedy", games=150, seed=3).win_rate
        greedy_vs_random = bot_vs_bot("greedy", "random", games=150, seed=3).win_rate
        random_vs_rule = bot_vs_bot("random", "rule", games=150, seed=3).win_rate

        self.assertGreater(rule_vs_greedy, 0.45)
        self.assertGreater(greedy_vs_random, 0.45)
        self.assertLess(random_vs_rule, 0.25)

    def test_identical_bots_land_near_the_one_in_three_baseline(self):
        rate = bot_vs_bot("rule", "rule", games=300, seed=11).win_rate
        self.assertAlmostEqual(rate, 1 / 3, delta=0.09)


class TestThreeWayMatch(unittest.TestCase):
    """三方混战：每副牌打满 6 种座位排列，牌运和先手都被对消。"""

    def test_needs_exactly_three_players(self):
        with self.assertRaises(ValueError):
            match([("a", make_bot("rule")), ("b", make_bot("greedy"))], deals=1)

    def test_plays_six_games_per_deal(self):
        results = match(
            [(name, make_bot(name)) for name in ("rule", "greedy", "random")], deals=5, seed=0
        )
        self.assertEqual([stats.games for _, stats in results], [30, 30, 30])

    def test_win_rates_sum_to_one(self):
        results = match(
            [(name, make_bot(name)) for name in ("rule", "greedy", "random")], deals=20, seed=1
        )
        self.assertAlmostEqual(sum(stats.win_rate for _, stats in results), 1.0, places=6)

    def test_identical_players_all_land_near_the_baseline(self):
        results = match([(f"rule{i}", make_bot("rule")) for i in range(3)], deals=40, seed=2)
        for _, stats in results:
            self.assertAlmostEqual(stats.win_rate, 1 / 3, delta=0.1)

    def test_stronger_bot_wins_the_table(self):
        results = dict(
            match([(name, make_bot(name)) for name in ("rule", "greedy", "random")], deals=40, seed=3)
        )
        self.assertGreater(results["rule"].win_rate, results["greedy"].win_rate)
        self.assertGreater(results["greedy"].win_rate, results["random"].win_rate)


class TestArenaHelpers(unittest.TestCase):
    def test_winner_gets_one_loser_gets_negative(self):
        result = GameResult(winner=1, remaining=[4, 0, 8], turns=20)
        self.assertEqual(final_reward(result, 1), 1.0)
        self.assertAlmostEqual(final_reward(result, 0), -0.25)
        self.assertAlmostEqual(final_reward(result, 2), -0.5)

    def test_evaluate_reports_consistent_counts(self):
        stats = evaluate(make_bot("greedy"), "random", games=60, seed=1)
        self.assertEqual(stats.games, 60)
        self.assertEqual(stats.win_rate, stats.wins / 60)
        self.assertGreater(stats.avg_turns, 0)


if __name__ == "__main__":
    unittest.main()
