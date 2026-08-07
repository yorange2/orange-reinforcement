"""M5 卡池扩展与保真：全经典构筑池、潜行/扰咒行为、压力测试。"""

from __future__ import annotations

import random
import unittest

import orange_stone as os

from .. import decks
from ..bots import GreedyBot, RandomBot
from ..env import Env


class TestFullPool(unittest.TestCase):
    def test_pool_is_large_enough(self):
        pool = decks.full_pool()
        self.assertGreater(len(pool), 300)
        # 池里没有硬币和衍生物
        self.assertNotIn("GAME_005", pool)
        self.assertFalse(any(cid.endswith("t") for cid in pool))

    def test_pool_open_flag_excludes_exactly_the_registry(self):
        """D1 开关：include_pool_open=False 恰好剔除注册表里的开放池卡。"""
        pool = decks.full_pool()
        closed = decks.full_pool(include_pool_open=False)
        open_ids = set(os.GameEnv.pool_open_card_ids())
        self.assertEqual(len(pool) - len(closed), len(open_ids))
        self.assertTrue(set(closed) <= set(pool))
        self.assertFalse(open_ids & set(closed))

    def test_random_deck_is_30(self):
        deck = decks.random_deck(random.Random(0))
        self.assertEqual(len(deck), 30)
        self.assertEqual(len(set(deck)) <= 30, True)

    def test_stress_random_games_finish(self):
        """随机套牌（全池）对局全部正常完局。"""
        bad = []
        for s in range(20):
            deck = decks.random_deck(random.Random(s))
            env = Env(deck=deck, seed=s)
            env.reset(seed=s)
            seats = {1: RandomBot(s), 2: GreedyBot(s)}
            steps = 0
            while not env.done and steps < 5000:
                acts = env.legal_actions()
                if not acts:
                    bad.append(("stuck", s))
                    break
                env.step(seats[env.current_player].choose(env.observe(), acts))
                steps += 1
            if not env.done and not bad:
                bad.append(("deadlock", s))
        self.assertEqual(bad, [], f"压力测试异常: {bad[:3]}")


class TestKeywords(unittest.TestCase):
    def test_stealth_minion_cannot_be_attacked(self):
        """丛林豹（潜行）在对手攻击枚举里不可见。"""
        env = Env(deck=["NEUTRAL_C10"] * 10, seed=1, bot="none")
        env.reset(seed=1)
        for _ in range(6):
            acts = env.legal_actions()
            play = next((a for a in acts if a.kind == "play"), None)
            if play is not None and env.current_player == 1:
                env.step(play)
                break
            env.step(next(a for a in acts if a.kind == "end_turn"))
        env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
        env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
        obs = env.observe()
        self.assertTrue(obs.me.field[0].stealth)
        env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
        obs2 = env.observe()
        stealth_ids = {m.entity_id for m in obs2.opponent.field if m.stealth}
        for a in env.legal_actions():
            if a.kind == "attack":
                self.assertNotIn(a.target_id, stealth_ids,
                                 "潜行随从不能被攻击")

    def test_elusive_card_exposes_the_flag(self):
        """精灵龙（扰咒）视图暴露 elusive。"""
        import orange_stone as os

        env = os.GameEnv(seed=2, deck=["CLASSIC_019"] * 10, bot="none")
        card = next(x for x in env.structured_observation().me.hand
                    if x.card_id == "CLASSIC_019")
        self.assertTrue(card.elusive)


if __name__ == "__main__":
    unittest.main()
