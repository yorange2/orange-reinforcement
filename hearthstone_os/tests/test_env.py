"""环境本身的守卫（路线图 M2，抄 `rosetta/tests/test_env.py` 三件套）。

三件套：完局、确定性、合法动作枚举一致性。重点是"枚举出来的动作，
引擎一定认"——orange-stone 的 `legal_action_infos` 用引擎校验过滤，
任何被枚举出来的动作执行后都必须推进局面（no-op 就是枚举和判定不一致）。
"""

from __future__ import annotations

import unittest

from .. import decks
from ..bots import GreedyBot, RandomBot, RuleBot
from ..env import Action, Env

MAX_STEPS = 3000


def make_env(**kwargs) -> Env:
    return Env(deck=decks.vanilla(), **kwargs)


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


def obs_sig(obs) -> tuple:
    """结构化局面的紧凑签名：任何已执行的动作都必须改变它。"""
    def entity(e):
        return (e.entity_id, e.cost, e.attack, e.health, e.can_attack)
    return (
        obs.turn, obs.my_turn, obs.done, obs.winner, obs.awaiting_choice,
        obs.me.hero_health, obs.me.hero_armor, obs.me.remaining_mana,
        obs.me.hand_count, obs.me.deck_count,
        obs.opponent.hero_health, obs.opponent.hand_count,
        tuple(entity(e) for e in obs.me.field),
        tuple(entity(e) for e in obs.opponent.field),
    )


class TestDeck(unittest.TestCase):
    def test_deck_size(self):
        self.assertEqual(len(decks.vanilla()), 30)

    def test_vanilla_ids_are_in_subset(self):
        # vanilla 是 G9 子集的一部分，对拍测试才有对应卡名
        self.assertTrue(set(decks.VANILLA_IDS) <= set(decks.SUBSET_MAP))

    def test_unknown_card_id_raises(self):
        with self.assertRaises(ValueError):
            Env(deck=["NOT_A_CARD_XYZ"])


class TestGameFlow(unittest.TestCase):
    def test_reset_gives_a_fresh_game(self):
        env = make_env()
        obs = env.reset(seed=0)
        self.assertEqual(obs.me.hero_health, 30)
        self.assertEqual(obs.opponent.hero_health, 30)
        self.assertEqual(env.turn, 1)
        self.assertEqual(env.current_player, 1)
        self.assertFalse(env.done)

    def test_observe_is_current_actor_view(self):
        """observe() 返回行动方视角：me 带手牌，opponent 不带。"""
        env = make_env()
        env.reset(seed=0)
        obs = env.observe()
        self.assertGreater(obs.me.hand_count, 0)
        self.assertEqual(obs.opponent.hand, [])       # 对手手牌隐藏
        self.assertGreater(obs.opponent.hand_count, 0)  # 但手牌数可见

    def test_opening_shape_matches_simplified_hearthstone(self):
        """先手 4 张（3 起手 + 第 1 回合抽牌）、后手 4 张 + 幸运币——官方规则。"""
        env = make_env()
        env.reset(seed=0)
        obs = env.observe()
        self.assertEqual(obs.me.hand_count, 4)
        self.assertEqual(obs.opponent.hand_count, 5)  # 4 + 硬币
        self.assertEqual(obs.me.deck_count, 26)
        self.assertEqual(obs.opponent.deck_count, 26)

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
            self.assertTrue(
                any(a.kind == "end_turn" for a in actions),
                "非选择状态下必须能结束回合",
            )
            env.step(next(a for a in actions if a.kind == "end_turn"))

    def test_turn_alternates(self):
        env = make_env()
        env.reset(seed=0)
        seats = []
        for _ in range(6):
            seats.append(env.current_player)
            actions = env.legal_actions()
            env.step(next(a for a in actions if a.kind == "end_turn"))
        self.assertEqual(seats, [1, 2, 1, 2, 1, 2])

    def test_no_actions_after_game_over(self):
        env = make_env()
        env.reset(seed=1)
        play_out(env, seed=1)

        self.assertTrue(env.done)
        self.assertEqual(env.legal_actions(), [])
        # 引擎对终局后的 step 是静默 no-op（2026-08 实测），不崩、不改局面。
        # 合约是：done 之后 legal_actions() 必须为空，由调用方据此停手。
        env.step(0)
        self.assertTrue(env.done)
        self.assertEqual(env.legal_actions(), [])


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_game(self):
        def trace(seed: int) -> list[tuple]:
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
                out.append(obs_sig(obs))
                env.step(bot.choose(obs, actions))
                steps += 1
            return out

        self.assertEqual(trace(11), trace(11))

    def test_dual_env_instances_stay_in_lockstep(self):
        """同 seed 的两个 Env（双 GameEnv 设计）走同动作序列结果一致。"""
        def trace(seed: int) -> list[tuple]:
            env = make_env()
            env.reset(seed=seed)
            bot = RandomBot(3)
            out = []
            steps = 0
            while not env.done and steps < MAX_STEPS:
                actions = env.legal_actions()
                if not actions:
                    break
                obs = env.observe()
                out.append(obs_sig(obs))
                env.step(bot.choose(obs, actions))
                steps += 1
            return out

        a, b = trace(21), trace(21)
        self.assertEqual(len(a), len(b))
        self.assertEqual(a, b)


class TestLegalActionsAreAccepted(unittest.TestCase):
    """每个被枚举出来的动作，引擎都必须真的执行它（no-op 即失败）。"""

    def test_random_playthroughs(self):
        for seed in range(30):
            env = make_env()
            env.reset(seed=seed)
            steps = play_out(env, seed=seed)
            self.assertLess(steps, MAX_STEPS, f"seed={seed} 打不完，疑似死循环")

    def test_greedy_playthroughs(self):
        for seed in range(30):
            env = make_env()
            env.reset(seed=seed)
            steps = play_out(env, seed=seed, bot=GreedyBot(seed))
            self.assertLess(steps, MAX_STEPS, f"seed={seed} 打不完，疑似死循环")

    def test_each_enumerated_action_advances_the_state(self):
        """clone() 分支：从同一局面出发逐个执行合法动作，局面必须变化。

        这是"枚举与引擎判定一致"的直接验证——如果某个动作被枚举出来但执行
        后什么也没发生，说明枚举里有引擎不会接受的幽灵动作。
        """
        env = make_env()
        env.reset(seed=5)
        checked = 0
        while not env.done and checked < 200:
            actions = env.legal_actions()
            if not actions:
                break
            obs = env.observe()
            before = obs_sig(obs)
            for action in actions:
                branch = env.clone()
                branch.step(action)
                self.assertNotEqual(
                    obs_sig(branch.observe()), before,
                    f"动作 {action} 执行后局面没动，枚举与引擎判定不一致",
                )
                checked += 1
            # 挑中间一个动作推进真局面，继续下一轮
            env.step(actions[len(actions) // 2])

        self.assertGreater(checked, 30, "校验覆盖的决策点太少")


class TestClone(unittest.TestCase):
    def test_clone_branches_without_affecting_original(self):
        env = make_env()
        env.reset(seed=3)
        before = obs_sig(env.observe())

        branch = env.clone()
        self.assertEqual(obs_sig(branch.observe()), before)
        branch.step(branch.legal_actions()[0])
        self.assertNotEqual(obs_sig(branch.observe()), before)
        self.assertEqual(obs_sig(env.observe()), before)


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

        # rosetta 那边是 >0.55，但那个卡池有潜行/扰咒/英雄技能可以白嫖，
        # rule 的优势被放大。orange-stone 的 G9 子集纯随从，实测 52~54%
        # （600 局 × 3 seed），这里只断言"比 50% 强"。
        result = arena.duel(RuleBot, GreedyBot, episodes=600, seed=1)
        self.assertGreater(
            result["win_rate"], 0.50,
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
