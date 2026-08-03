import copy
import random
import unittest

from hearthstone.bots import make_bot
from hearthstone.game import END, END_TURN, Game, play_game
from hearthstone.policy import UnifiedNet
from hearthstone.search import TurnSearchAgent


def _snapshot(game: Game):
    """局面里所有会变的东西，用来验证克隆的隔离性。"""
    return copy.deepcopy((
        game.decks, game.hands, game.boards, game.hero_health, game.mana,
        game.max_mana, game.fatigue, game.burned, game.weapons,
        game.weapon_durability, game.hero_attacked, game.turns,
        game.finished, game.winner, game.current, game._next_uid,
    ))


def _advance(game: Game, n: int, seed: int = 0) -> None:
    bot = make_bot("rule", seed=seed)
    for _ in range(n):
        if game.finished:
            return
        game.step(bot.choose(game.observe()))


class TestClone(unittest.TestCase):
    def test_clone_is_isolated(self):
        """在克隆体上一路打到底，原局面必须一动不动。"""
        for t in range(40):
            game = Game(rng=random.Random(t))
            _advance(game, t % 20, seed=t)
            if game.finished:
                continue
            before = _snapshot(game)
            twin = game.clone()
            rng = random.Random(t)
            while not twin.finished:
                twin.step(rng.choice(twin.legal_actions()))
            self.assertEqual(_snapshot(game), before, f"第 {t} 次克隆污染了原局面")

    def test_clone_does_not_consume_original_rng(self):
        """搜索消耗的随机数不能来自真实对局，否则真实牌序会被搅乱。"""
        game = Game(rng=random.Random(0))
        _advance(game, 6)
        state = game.rng.getstate()
        twin = game.clone()
        rng = random.Random(1)
        while not twin.finished:
            twin.step(rng.choice(twin.legal_actions()))
        self.assertEqual(game.rng.getstate(), state)

    def test_clone_preserves_state(self):
        game = Game(rng=random.Random(3))
        _advance(game, 9)
        self.assertEqual(_snapshot(game.clone()), _snapshot(game))


class TestPlanIsolation(unittest.TestCase):
    """回归测试：计划绝不能跨回合残留。

    曾经的 bug——`choose` 在"只有一个合法动作"时提前返回，没有消费掉计划，于是上个
    回合没走完的尾巴漏到下一个回合，开局就打出一个陈旧动作再立刻结束回合。胜率从
    50% 崩到 2%，但全程没有任何异常抛出，只能靠这条测试守住。
    """

    def _agent(self, game, seat=0):
        agent = TurnSearchAgent(UnifiedNet(), beam=4, seed=0)
        agent.bind_game(game, seat)
        return agent

    def test_stale_plan_is_discarded_on_new_turn(self):
        game = Game(rng=random.Random(0))
        agent = self._agent(game)
        obs = game.observe()
        self.assertGreater(len(obs.legal), 1, "这个局面要有多个选择，测试才有意义")

        bogus = END_TURN
        agent._plan = [bogus, bogus]
        agent._plan_turn = obs.turn - 1        # 属于上一个回合的计划

        action = agent.choose(obs)
        self.assertEqual(agent._plan_turn, obs.turn, "计划应该为当前回合重新搜过")
        self.assertNotIn(bogus, [action], "陈旧计划不该被执行")

    def test_plan_is_reused_within_a_turn(self):
        game = Game(rng=random.Random(0))
        agent = self._agent(game)
        obs = game.observe()
        agent.choose(obs)
        turn_after_first = agent._plan_turn
        self.assertEqual(turn_after_first, obs.turn)

    def test_every_action_belongs_to_the_current_turn(self):
        """打完整局，每次出手时计划都必须属于当前回合。"""
        game = Game(rng=random.Random(5))
        agent = self._agent(game, seat=0)
        bot = make_bot("rule", seed=5)
        while not game.finished:
            obs = game.observe()
            if obs.player == 0:
                action = agent.choose(obs)
                if agent._plan or action.kind != END:
                    self.assertEqual(agent._plan_turn, obs.turn)
            else:
                action = bot.choose(obs)
            game.step(action)


class TestSearchAgent(unittest.TestCase):
    def test_plays_a_full_game_legally(self):
        """`play_game` 自己会校验动作合法性，跑通就说明搜索没给出非法动作。"""
        agent = TurnSearchAgent(UnifiedNet(), beam=6, seed=0)
        result = play_game([agent, make_bot("rule", seed=1)], rng=random.Random(1))
        self.assertIn(result.winner, (0, 1, None))

    def test_bind_game_is_called_by_play_game(self):
        agent = TurnSearchAgent(UnifiedNet(), beam=4, seed=0)
        self.assertIsNone(agent._game)
        play_game([agent, make_bot("rule", seed=2)], rng=random.Random(2))
        self.assertIsNotNone(agent._game)

    def test_requires_bind_game(self):
        agent = TurnSearchAgent(UnifiedNet(), beam=4, seed=0)
        game = Game(rng=random.Random(0))
        with self.assertRaises(RuntimeError):
            agent.choose(game.observe())

    def test_root_shuffles_own_deck(self):
        """搜索不许预知自己的牌序——回合内的抽牌法术会用到牌堆顶。"""
        game = Game(rng=random.Random(0))
        agent = TurnSearchAgent(UnifiedNet(), beam=4, seed=0)
        agent.bind_game(game, 0)
        orders = {tuple(c.name for c in agent._root(game, 0).decks[0]) for _ in range(8)}
        self.assertGreater(len(orders), 1, "每次搜索都该重新洗自己的牌堆")
        # 洗牌只改顺序，不改牌堆构成
        real = sorted(c.name for c in game.decks[0])
        for order in orders:
            self.assertEqual(sorted(order), real)

    def test_opponent_bots_are_unaffected_by_bind_game(self):
        """普通规则对手也会被 play_game 调用 bind_game，必须是空操作。"""
        bot = make_bot("rule", seed=0)
        game = Game(rng=random.Random(0))
        bot.bind_game(game, 0)
        self.assertIn(bot.choose(game.observe()), game.observe().legal)


if __name__ == "__main__":
    unittest.main()
