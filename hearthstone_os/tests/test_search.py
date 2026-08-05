"""整回合搜索的守卫：clone 恢复、计划合法性、完整对局。"""

from __future__ import annotations

import unittest

from .. import decks
from ..arena import play_game
from ..bots import RuleBot
from ..env import Env
from ..policy import UnifiedNet
from ..search import MAX_TURN_ACTIONS, TurnSearchAgent


def make_net() -> UnifiedNet:
    return UnifiedNet()


class TestCloneSearch(unittest.TestCase):
    def test_search_does_not_touch_the_real_game(self):
        """克隆推演必须不消耗真实对局的 RNG/状态。"""
        env = Env(deck=decks.vanilla(), seed=5)
        env.reset(seed=5)
        before = env.observe()
        agent = TurnSearchAgent(make_net())
        agent.bind_env(env, seat=1)

        # 搜一个回合（内部全是克隆）
        plan = agent._search(env, 1)
        self.assertTrue(plan)

        after = env.observe()
        # 回合数、双方血量、手牌数都不该动
        self.assertEqual(before.turn, after.turn)
        self.assertEqual(before.me.hero_health, after.me.hero_health)
        self.assertEqual(before.me.hand_count, after.me.hand_count)

    def test_plan_is_legal_sequence(self):
        """搜索出的计划：逐步执行时每个动作都必须是当时的合法动作，且回合走完。"""
        for seed in range(5):
            env = Env(deck=decks.vanilla(), seed=seed)
            env.reset(seed=seed)
            agent = TurnSearchAgent(make_net())
            agent.bind_env(env, seat=1)
            plan = agent._search(env, 1)

            twin = env.clone()
            for action in plan:
                actions = twin.legal_actions()
                self.assertIn(action, actions, f"seed={seed}: 计划动作不合法")
                twin.step(action)
                if twin.done:
                    break
            # 计划必须以 end_turn 收尾（或已经打完）
            self.assertTrue(twin.done or plan[-1].kind == "end_turn",
                            f"seed={seed}: 计划没结束回合")

    def test_search_completes_full_games(self):
        """搜索智能体 + rule 对手打完一局不报错。"""
        agent = TurnSearchAgent(make_net(), seed=0)
        for seed in range(3):
            result = play_game([agent, RuleBot(seed=1)], seed=seed)
            self.assertIn(result.winner, (0, 1, 2), f"seed={seed} 没打完")

    def test_stale_plan_discarded_on_new_turn(self):
        """计划只对搜它的回合有效，跨回合必须作废重搜。"""
        env = Env(deck=decks.vanilla(), seed=2)
        env.reset(seed=2)
        agent = TurnSearchAgent(make_net())
        agent.bind_env(env, seat=1)

        obs = env.observe()
        agent._plan = [next(a for a in env.legal_actions() if a.kind == "end_turn")]
        agent._plan_turn = obs.turn
        actions = env.legal_actions()
        chosen = agent.choose(obs, actions)
        # 同回合：直接复用计划
        self.assertEqual(chosen.kind, "end_turn")

        # 跨回合：计划作废，重新搜索
        env.step(chosen)
        env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
        obs2 = env.observe()
        self.assertNotEqual(obs2.turn, obs.turn)
        actions2 = env.legal_actions()
        chosen2 = agent.choose(obs2, actions2)
        self.assertIn(chosen2, actions2)


if __name__ == "__main__":
    unittest.main()
