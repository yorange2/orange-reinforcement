import random
import unittest

from hearthstone.arena import duel, evaluate, final_reward
from hearthstone.bots import BOTS, GreedyBot, RandomBot, RuleBot, make_bot
from hearthstone.cards import (
    CHARGE,
    DIVINE_SHIELD,
    LIFESTEAL,
    POISONOUS,
    REBORN,
    RUSH,
    STEALTH,
    TAUNT,
    THE_COIN,
    WINDFURY,
    CardDef,
)
from hearthstone.game import (
    END,
    HERO,
    HERO_HEALTH,
    Game,
    GameResult,
    Minion,
    play_game,
)

WISP = CardDef("w", 0, 1, 1)
BIG = CardDef("B", 6, 6, 7)
CHARGER = CardDef("CH", 1, 2, 1, (CHARGE,))


def fresh(seed=0, **kwargs):
    return Game(rng=random.Random(seed), **kwargs)


def stack(game, player, *cards, ready=False):
    for card in cards:
        minion = Minion.summon(card, game._take_uid())
        minion.just_played = not ready
        if ready:
            minion.attacks_left = Minion.max_attacks(card)
        game.boards[player].append(minion)


# ---------------------------------------------------------------- Interface
class TestInterface(unittest.TestCase):
    def test_every_bot_only_returns_legal_actions(self):
        for name in BOTS:
            for seed in range(10):
                bot = make_bot(name, seed=seed)
                game = fresh(seed)
                while not game.finished:
                    obs = game.observe()
                    action = bot.choose(obs)
                    self.assertIn(action, obs.legal, f"{name} seed={seed} 非法 {action}")
                    game.step(action)

    def test_unknown_bot_raises(self):
        with self.assertRaises(ValueError):
            make_bot("不存在")


# ---------------------------------------------------------------- Greedy
class TestGreedy(unittest.TestCase):
    def test_plays_the_coin_immediately(self):
        game = fresh()
        game.hands[0] = [THE_COIN, WISP]
        game.mana[0] = 1
        action = GreedyBot().choose(game.observe())
        self.assertEqual(game.hands[0][action.source].name, THE_COIN.name)

    def test_goes_face_when_no_taunt(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        action = GreedyBot().choose(game.observe())
        self.assertEqual(action.target, HERO)

    def test_hits_taunt_when_blocked(self):
        game = fresh()
        stack(game, 0, BIG, ready=True)
        stack(game, 1, CardDef("T", 2, 1, 3, (TAUNT,)), ready=False)
        action = GreedyBot().choose(game.observe())
        self.assertNotEqual(action.target, HERO)

    def test_ends_turn_when_out_of_options(self):
        game = fresh()
        game.hands[0] = []
        self.assertEqual(GreedyBot().choose(game.observe()).kind, END)


# ---------------------------------------------------------------- Rule
class TestRule(unittest.TestCase):
    def test_plays_coin_immediately(self):
        game = fresh()
        game.hands[0] = [THE_COIN, WISP]
        game.mana[0] = 1
        action = RuleBot().choose(game.observe())
        self.assertEqual(game.hands[0][action.source].name, THE_COIN.name)

    def test_takes_lethal(self):
        game = fresh()
        game.hero_health[1] = 4
        stack(game, 0, CardDef("X", 0, 4, 5), ready=True)
        stack(game, 1, CardDef("D", 0, 1, 1), ready=False)
        self.assertEqual(RuleBot().choose(game.observe()).target, HERO)

    def test_lethal_blocked_by_taunt(self):
        game = fresh()
        game.hero_health[1] = 4
        stack(game, 0, CardDef("X", 0, 4, 5), ready=True)
        stack(game, 1, CardDef("T", 2, 1, 3, (TAUNT,)), ready=False)
        action = RuleBot().choose(game.observe())
        self.assertNotEqual(action.target, HERO)  # must clear taunt

    def test_attacks_before_playing(self):
        game = fresh()
        game.hands[0] = [WISP]
        game.mana[0] = 10
        stack(game, 0, BIG, ready=True)
        action = RuleBot().choose(game.observe())
        self.assertEqual(action.kind, "attack")


# ---------------------------------------------------------------- Strength gradient
class TestStrengthGradient(unittest.TestCase):
    """rule > greedy > random，同类打同类落在 50% 附近。"""

    GAMES = 200

    def rate(self, a, b):
        return evaluate(make_bot(a, seed=0), b, games=self.GAMES, seed=0).win_rate

    def test_rule_beats_greedy(self):
        # 200 局有 ±7pp 的方差，0.50 就够了（8 种子均值 58.8%）
        self.assertGreater(self.rate("rule", "greedy"), 0.50)

    def test_greedy_beats_random(self):
        self.assertGreater(self.rate("greedy", "random"), 0.8)

    def test_rule_beats_random(self):
        self.assertGreater(self.rate("rule", "random"), 0.8)

    def test_mirrors_are_even(self):
        for name in BOTS:
            with self.subTest(bot=name):
                self.assertAlmostEqual(self.rate(name, name), 0.5, delta=0.12)

    def test_first_player_advantage_is_within_hs_range(self):
        """先手有优势是炉石的常态——Coin 是补偿，不是抹平。

        真实 HS 里先手胜率通常在 52~55%，快攻 meta 更高。这个格式有冲锋和突袭、
        没有解牌，节奏更快，先手优势会比标准 HS 更明显，但不应该到碾压的程度。
        """
        thresholds = {"random": (0.40, 0.60), "greedy": (0.50, 0.72), "rule": (0.50, 0.72)}
        for name in BOTS:
            with self.subTest(bot=name):
                wins = [0, 0]
                for s in range(300):
                    table = [make_bot(name, seed=s), make_bot(name, seed=s + 9973)]
                    result = play_game(table, rng=random.Random(s), first=0)
                    if result.winner is not None:
                        wins[result.winner] += 1
                rate = wins[0] / 300
                lo, hi = thresholds[name]
                self.assertGreaterEqual(rate, lo, f"{name} 先手胜率 {rate:.1%} < {lo:.0%}")
                self.assertLessEqual(rate, hi, f"{name} 先手胜率 {rate:.1%} > {hi:.0%}")


# ---------------------------------------------------------------- Arena
class TestArena(unittest.TestCase):
    def test_duel_is_symmetric(self):
        a, b = RuleBot(), GreedyBot()
        fwd = duel(a, b, deals=40, seed=0)
        rev = duel(b, a, deals=40, seed=0)
        self.assertAlmostEqual(fwd.win_rate + rev.win_rate + fwd.draw_rate, 1.0, places=6)

    def test_stats_add_up(self):
        stat = evaluate(RuleBot(), "greedy", games=50)
        self.assertEqual(stat.games, 50)
        self.assertLessEqual(stat.wins + stat.draws, stat.games)


# ---------------------------------------------------------------- Reward
class TestReward(unittest.TestCase):
    def test_win_is_one(self):
        result = GameResult(winner=0, hero_health=[7, -2], turns=10)
        self.assertEqual(final_reward(result, 0), 1.0)

    def test_draw_is_zero(self):
        result = GameResult(winner=None, hero_health=[-1, -1], turns=10)
        self.assertEqual(final_reward(result, 0), 0.0)

    def test_close_loss_hurts_less_than_blowout(self):
        close = GameResult(winner=1, hero_health=[-3, 2], turns=10)
        blowout = GameResult(winner=1, hero_health=[-3, HERO_HEALTH], turns=10)
        self.assertGreater(final_reward(close, 0), final_reward(blowout, 0))


if __name__ == "__main__":
    unittest.main()
