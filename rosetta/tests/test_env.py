"""环境本身的守卫。重点是"枚举出来的动作，引擎一定认"。"""

from __future__ import annotations

import random
import unittest

from .. import decks
from ..bots import GreedyBot, RandomBot, RuleBot
from ..env import Action, ActionType, Env

MAX_STEPS = 3000


def make_env(**kwargs) -> Env:
    deck = decks.vanilla()
    return Env(player1_deck=deck, player2_deck=deck, **kwargs)


def play_out(env: Env, seed: int = 0, bot=None) -> int:
    """把一局打完，返回步数。"""
    bot = bot or RandomBot(seed)
    steps = 0
    while not env.done and steps < MAX_STEPS:
        actions = env.legal_actions()
        if not actions:
            break
        env.step(bot.choose(env.observe(), actions))
        steps += 1
    return steps


class TestDeck(unittest.TestCase):
    def test_deck_size(self):
        self.assertEqual(len(decks.vanilla()), 30)

    def test_card_ids_resolve(self):
        # 卡 ID 拼错的话 Env 构造时就会抛
        env = make_env()
        env.reset(seed=0)
        # 先手 3 + 第 1 回合抽 1 = 4 张，牌堆剩 26
        self.assertEqual(env.observe().me.deck_count, 30 - 4)


class TestGameFlow(unittest.TestCase):
    def test_reset_gives_a_fresh_game(self):
        env = make_env()
        obs = env.reset(seed=0)
        self.assertEqual(obs.me.hero_health, 30)
        self.assertEqual(obs.opponent.hero_health, 30)
        self.assertEqual(env.turn, 1)
        self.assertFalse(env.done)

    def test_first_player_draws_three_second_draws_four(self):
        env = make_env(start_player="PLAYER1")
        obs = env.reset(seed=0)
        # 先手 3 张，后手 4 张 + 幸运币
        self.assertEqual(obs.me.hand_count, 3 + 1)  # 本回合开始时又抽了一张
        self.assertEqual(obs.opponent.hand_count, 4 + 1)

    def test_game_reaches_an_end(self):
        env = make_env()
        env.reset(seed=1)
        steps = play_out(env, seed=1)
        self.assertTrue(env.done, f"{steps} 步都没打完")
        self.assertIn(env.winner, (0, 1, 2))

    def test_end_turn_is_always_available(self):
        env = make_env()
        env.reset(seed=0)
        for _ in range(20):
            if env.done:
                break
            actions = env.legal_actions()
            if not any(a.type == ActionType.CHOOSE for a in actions):
                self.assertTrue(
                    any(a.type == ActionType.END_TURN for a in actions),
                    "非选择状态下必须能结束回合",
                )
            env.step(next(a for a in actions if a.type == ActionType.END_TURN))

    def test_turn_alternates(self):
        env = make_env(start_player="PLAYER1")
        env.reset(seed=0)
        seats = []
        for _ in range(6):
            seats.append(env.current_player)
            actions = env.legal_actions()
            env.step(next(a for a in actions if a.type == ActionType.END_TURN))
        self.assertEqual(seats, [1, 2, 1, 2, 1, 2])

    def test_no_actions_after_game_over(self):
        env = make_env()
        env.reset(seed=1)
        play_out(env, seed=1)

        self.assertTrue(env.done)
        self.assertEqual(env.legal_actions(), [])

        end_turn = Action()
        end_turn.type = ActionType.END_TURN
        with self.assertRaises(RuntimeError):
            env.step(end_turn)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_game(self):
        def trace(seed: int) -> list[tuple[int, int, int]]:
            env = make_env()
            env.reset(seed=seed)
            bot = RandomBot(7)
            out = []
            steps = 0
            while not env.done and steps < MAX_STEPS:
                actions = env.legal_actions()
                if not actions:
                    break
                obs = env.observe()
                out.append(
                    (env.current_player, obs.me.hero_health,
                     obs.opponent.hero_health)
                )
                env.step(bot.choose(obs, actions))
                steps += 1
            return out

        self.assertEqual(trace(11), trace(11))


class TestLegalActionsAreAccepted(unittest.TestCase):
    """每个被枚举出来的动作，引擎都必须真的执行它。

    RosettaStone 的 task 在参数不合法时是静默 return，局面纹丝不动。
    绑定层为此加了指纹守卫，任何 no-op 都会抛 RuntimeError——
    也就是说这个测试跑通，就等于枚举和引擎的判定是一致的。
    """

    def test_random_playthroughs(self):
        for seed in range(30):
            env = make_env()
            env.reset(seed=seed)
            steps = play_out(env, seed=seed)  # no-op 会在这里抛出来
            self.assertLess(steps, MAX_STEPS, f"seed={seed} 打不完，疑似死循环")

    def test_greedy_playthroughs(self):
        for seed in range(30):
            env = make_env()
            env.reset(seed=seed)
            steps = play_out(env, seed=seed, bot=GreedyBot(seed))
            self.assertLess(steps, MAX_STEPS, f"seed={seed} 打不完，疑似死循环")

    def test_hero_power_is_once_per_turn(self):
        """英雄技能用过之后就不该再出现在合法动作里。

        这是当初枚举漏掉 IsExhausted() 时的具体表现：机器人在同一个回合里
        按了 5000 次法术冲击，局面一动不动。
        """
        env = make_env(player1_class="MAGE", player2_class="MAGE")
        env.reset(seed=0)
        rng = random.Random(0)

        used_in_turn: set[tuple[int, int]] = set()
        actions_total = 0
        while not env.done and actions_total < 400:
            actions = env.legal_actions()
            turn_and_player = (env.turn, env.current_player)

            powers = [a for a in actions if a.type == ActionType.HERO_POWER]
            if turn_and_player in used_in_turn:
                self.assertFalse(
                    powers,
                    f"t={env.turn} p={env.current_player} 同一个玩家 "
                    "在同一回合里英雄技能出现了两次",
                )

            chosen = None
            if powers:
                chosen = powers[0]
                used_in_turn.add(turn_and_player)
            else:
                end = [a for a in actions if a.type == ActionType.END_TURN]
                if end and rng.random() < 0.4:
                    chosen = end[0]
                else:
                    chosen = rng.choice(actions)

            env.step(chosen)
            actions_total += 1


class TestBots(unittest.TestCase):
    def test_greedy_beats_random(self):
        from .. import arena

        result = arena.duel(GreedyBot, RandomBot, episodes=60)
        self.assertGreater(result["win_rate"], 0.9)

    def test_greedy_mirror_is_balanced(self):
        from .. import arena

        result = arena.duel(GreedyBot, GreedyBot, episodes=400)
        self.assertAlmostEqual(result["win_rate"], 0.5, delta=0.08)

    def test_rule_beats_greedy(self):
        from .. import arena

        result = arena.duel(RuleBot, GreedyBot, episodes=200, seed=1)
        self.assertGreater(
            result["win_rate"], 0.55,
            "rule 有场面交换逻辑，应该比只会打脸的 greedy 强"
        )

    def test_rule_beats_random(self):
        from .. import arena

        result = arena.duel(RuleBot, RandomBot, episodes=60)
        self.assertGreater(result["win_rate"], 0.95)

    def test_rule_mirror_is_balanced(self):
        from .. import arena

        result = arena.duel(RuleBot, RuleBot, episodes=200)
        self.assertAlmostEqual(result["win_rate"], 0.5, delta=0.10)

    def test_rule_30_seeds_no_loop(self):
        """30 个 seed 下 rule 都不死循环。"""
        for seed in range(30):
            with self.subTest(seed=seed):
                env = make_env()
                env.reset(seed=seed)
                steps = play_out(env, seed=seed, bot=RuleBot())
                self.assertLess(
                    steps, MAX_STEPS,
                    f"seed={seed} rule 没打完，疑似死循环"
                )


if __name__ == "__main__":
    unittest.main()
