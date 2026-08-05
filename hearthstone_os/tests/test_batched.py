"""批量对局（M4）的守卫：确定性、批量视图、并行训练冒烟。"""

from __future__ import annotations

import unittest

from .. import decks
from ..arena import play_game
from ..batched import BatchedEnv
from ..bots import RandomBot
from ..env import Env

N = 10


class TestBatchedDeterminism(unittest.TestCase):
    def test_batched_matches_single_per_seed(self):
        """确定性策略下，批量与单局逐 seed 结果一致。"""
        b = BatchedEnv(N, decks.vanilla(), seeds=list(range(N)), bot="none")
        for _ in range(3000):
            legal = b.legal_actions()
            if not any(legal):
                break
            b.step([len(l) - 1 if l else 0 for l in legal])
        bw = b.winners()

        def single(seed: int):
            env = Env(deck=decks.vanilla(), seed=seed)
            env.reset(seed=seed)
            while not env.done:
                legal = env.legal_actions()
                if not legal:
                    break
                env.step(legal[-1] if len(legal) > 1 else legal[0])
            return env.winner - 1 if env.winner else None

        self.assertEqual([single(s) for s in range(N)], bw)

    def test_actor_view_observations(self):
        """批量观测是当前行动方视角（me 带手牌、opponent 不带）。"""
        b = BatchedEnv(2, decks.vanilla(), seeds=[1, 2], bot="none")
        obs = b.observe()
        for o in obs:
            self.assertGreater(o.me.hand_count, 0)
            self.assertEqual(o.opponent.hand, [])
            self.assertTrue(o.my_turn)

    def test_reset_one_does_not_touch_others(self):
        """只重开一局不影响其他局。"""
        b = BatchedEnv(3, decks.vanilla(), seeds=[1, 2, 3], bot="none")
        obs_before = [o.me.hero_health for o in b.observe()]
        b.reset_one(1, 99)
        obs_after = [o.me.hero_health for o in b.observe()]
        self.assertEqual(obs_before[0], obs_after[0])
        self.assertEqual(obs_before[2], obs_after[2])
        # 第 1 局重开后是新局面（回合 1）
        self.assertEqual(b.observe()[1].turn, 1)

    def test_active_players(self):
        """开局所有局都在 P1 行动。"""
        b = BatchedEnv(3, decks.vanilla(), seeds=[1, 2, 3], bot="none")
        self.assertEqual(b.active_players(), [0, 0, 0])


class TestBatchedBattle(unittest.TestCase):
    def test_battle_batch_deterministic(self):
        import orange_stone as os

        deck = decks.vanilla()
        single = os.battle_batch([7], deck, "greedy")[0]
        batch = os.battle_batch(list(range(N)), deck, "greedy")
        self.assertEqual(batch[7], single)

    def test_battle_batch_unknown_deck_raises(self):
        import orange_stone as os

        with self.assertRaises(ValueError):
            os.battle_batch([1], ["NOT_A_CARD"], "greedy")


class TestParallelTrainSmoke(unittest.TestCase):
    def test_parallel_train_runs_and_learns(self):
        """--parallel 8 短训能跑完且明显强于随机。"""
        from ..train import parse_args, train_parallel

        args = parse_args(["--episodes", "800", "--parallel", "8", "--seed", "0",
                           "--quiet"])
        agent = train_parallel(args)
        wins = 0
        for seed in range(30):
            result = play_game([agent.eval_agent(), RandomBot(seed)], seed=seed)
            if result.winner == 1:
                wins += 1
        # 短训 800 局实测 ~93%（M4 冒烟），30 局 σ≈8pp，阈值留 ~2σ
        self.assertGreater(wins / 30, 0.75, f"并行训练后打 random 才 {wins}/30")


if __name__ == "__main__":
    unittest.main()
