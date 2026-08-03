import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hearthstone.features import FEATURE_DIM, STATE_DIM, batch_features
from hearthstone.game import Game, play_game
from hearthstone.policy import (
    MoveScorer,
    PolicyAgent,
    ValueNet,
    discounted_returns,
    evaluate_batch,
    load_agent,
    make_batch,
    save_agent,
)


class TestMoveScorer(unittest.TestCase):
    def test_output_is_flat(self):
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        x = torch.randn(5, FEATURE_DIM)
        out = scorer(x)
        self.assertEqual(out.shape, (5,))

    def test_param_count(self):
        scorer = MoveScorer(hidden=32, layers=2)
        self.assertGreater(scorer.n_params, 100)

    def test_candidates_are_independent(self):
        """一起算和拆开算结果完全一致——钉住"不能用 BatchNorm"这条。"""
        scorer = MoveScorer(hidden=16, layers=1, norm="layer")
        scorer.eval()
        x = torch.randn(8, FEATURE_DIM)

        with torch.no_grad():
            batch = scorer(x)
            solo = torch.cat([scorer(x[i:i + 1]) for i in range(len(x))])

        self.assertTrue(torch.allclose(batch, solo, atol=1e-5))

    def test_padding_does_not_change_real_scores(self):
        """后面补零行不影响前面真实候选的分数。"""
        scorer = MoveScorer(hidden=16, layers=1, norm="layer")
        scorer.eval()
        real = torch.randn(3, FEATURE_DIM)

        with torch.no_grad():
            alone = scorer(real)
            padded = torch.cat([real, torch.zeros(5, FEATURE_DIM)])
            together = scorer(padded)
        self.assertTrue(torch.allclose(alone, together[:3], atol=1e-5))


class TestValueNet(unittest.TestCase):
    def test_output_is_scalar_per_sample(self):
        v = ValueNet(hidden=16, norm="none")
        x = torch.randn(4, STATE_DIM)
        out = v(x)
        self.assertEqual(out.shape, (4,))


class TestPolicyAgent(unittest.TestCase):
    def test_choose_returns_legal_action(self):
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(scorer, training=False)
        game = Game(rng=random.Random(0))
        while not game.finished:
            obs = game.observe()
            action = agent.choose(obs)
            self.assertIn(action, obs.legal, f"非法动作 {action}")
            game.step(action)

    def test_only_choice_no_gradient(self):
        """只有一个合法动作时不记录 step——这种决策点学不到东西。"""
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(scorer, training=True)
        game = Game(rng=random.Random(0))
        game.hands[0] = []
        obs = game.observe()
        # 只有结束回合一个选项
        self.assertEqual(len(obs.legal), 1)
        agent.choose(obs)
        self.assertEqual(len(agent.trajectory.steps), 0)

    def test_eval_agent_is_deterministic(self):
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        scorer.eval()
        agent = PolicyAgent(scorer, training=False)
        eval_agent = agent.eval_agent()
        self.assertFalse(eval_agent.training)

        game = Game(rng=random.Random(0))
        obs = game.observe()
        a1 = agent.choose(obs)
        a2 = eval_agent.choose(obs)
        self.assertEqual(a1, a2)


class TestBatch(unittest.TestCase):
    def test_make_and_evaluate(self):
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        scorer.eval()
        value = ValueNet(hidden=16, norm="none")
        value.eval()

        game = Game(rng=random.Random(0))
        agent = PolicyAgent(scorer, value, training=True)

        for _ in range(10):
            if game.finished:
                break
            obs = game.observe()
            agent.choose(obs)
            game.step(obs.legal[0])

        steps = agent.trajectory.steps
        if not steps:
            self.skipTest("没有产生决策点")

        batch = make_batch(steps)
        self.assertEqual(len(batch), len(steps))
        self.assertTrue(batch.mask.all(dim=1).any())

        with torch.no_grad():
            log_prob, entropy, values = evaluate_batch(scorer, value, batch)
        self.assertEqual(log_prob.shape, (len(steps),))
        self.assertEqual(entropy.shape, (len(steps),))


class TestDiscountedReturns(unittest.TestCase):
    def test_positive_reward(self):
        ret = discounted_returns(1.0, 5, 0.9)
        self.assertAlmostEqual(float(ret[0]), 1.0 * 0.9 ** 4, places=5)
        self.assertAlmostEqual(float(ret[-1]), 1.0)

    def test_negative_reward(self):
        ret = discounted_returns(-1.0, 3, 1.0)
        np.testing.assert_array_equal(ret, [-1.0, -1.0, -1.0])


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        scorer = MoveScorer(hidden=32, layers=2, norm="layer")
        value = ValueNet(hidden=32, norm="layer")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            save_agent(str(path), scorer, value, meta={"lr": 1e-3})
            agent = load_agent(str(path))
            self.assertEqual(agent.scorer.hidden, 32)
            self.assertEqual(agent.scorer.layers, 2)
            self.assertFalse(agent.training)


class TestEndToEnd(unittest.TestCase):
    def test_agent_can_complete_a_game(self):
        """智能体 + 规则对手能正常打完一局。"""
        scorer = MoveScorer(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(scorer, training=True)

        from hearthstone.bots import make_bot

        opponent = make_bot("random", seed=1)
        result = play_game([agent, opponent], rng=random.Random(2))
        self.assertIn(result.winner, (0, 1, None))
        self.assertTrue(result.turns > 0)


if __name__ == "__main__":
    unittest.main()
