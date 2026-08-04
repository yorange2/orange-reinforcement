import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hearthstone.features import FEATURE_DIM, STATE_DIM, STATE_OFFSET, batch_features
from hearthstone.game import Game, play_game
from hearthstone.policy import (
    PolicyAgent,
    UnifiedNet,
    ValueNet,
    discounted_returns,
    evaluate_batch,
    gae_advantages,
    load_agent,
    make_batch,
    save_agent,
)


class TestUnifiedNet(unittest.TestCase):
    def test_forward_single_output_shape(self):
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        net.eval()
        x = torch.randn(5, FEATURE_DIM)
        with torch.no_grad():
            logits, value = net.forward_single(x)
        self.assertEqual(logits.shape, (5,))
        self.assertEqual(value.shape, ())

    def test_param_count(self):
        net = UnifiedNet(hidden=32, layers=2)
        self.assertGreater(net.n_params, 100)

    def test_candidates_are_independent(self):
        """一起算和拆开算结果完全一致——钉住"不能用 BatchNorm"这条。

        UnifiedNet 用第一行的 state tail 为所有候选编码局面，
        所以要求测试数据 state tail 全行一致（和真实 game features 一样）。
        """
        net = UnifiedNet(hidden=16, layers=1, norm="layer")
        net.eval()
        # 构造共享 state tail 的特征矩阵
        state = torch.randn(STATE_DIM)
        N = 8
        x = torch.randn(N, FEATURE_DIM)
        x[:, STATE_OFFSET:] = state  # 所有行共享同一局面

        with torch.no_grad():
            batch_logits, _ = net.forward_single(x)
            solo_logits = torch.cat([
                net.forward_single(x[i:i + 1])[0] for i in range(len(x))
            ])
        self.assertTrue(torch.allclose(batch_logits, solo_logits, atol=1e-5))

    def test_padding_does_not_change_real_scores(self):
        """后面补零行不影响前面真实候选的分数。"""
        net = UnifiedNet(hidden=16, layers=1, norm="layer")
        net.eval()
        state = torch.randn(STATE_DIM)
        real = torch.randn(3, FEATURE_DIM)
        real[:, STATE_OFFSET:] = state
        padding = torch.zeros(5, FEATURE_DIM)
        padding[:, STATE_OFFSET:] = state  # 补零行的 state tail 也保持一致

        with torch.no_grad():
            alone, _ = net.forward_single(real)
            padded = torch.cat([real, padding])
            together, _ = net.forward_single(padded)
        self.assertTrue(torch.allclose(alone, together[:3], atol=1e-5))


class TestValueNet(unittest.TestCase):
    def test_output_is_scalar_per_sample(self):
        v = ValueNet(hidden=16, norm="none")
        x = torch.randn(4, STATE_DIM)
        out = v(x)
        self.assertEqual(out.shape, (4,))


class TestPolicyAgent(unittest.TestCase):
    def test_choose_returns_legal_action(self):
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(net, training=False)
        game = Game(rng=random.Random(0))
        while not game.finished:
            obs = game.observe()
            action = agent.choose(obs)
            self.assertIn(action, obs.legal, f"非法动作 {action}")
            game.step(action)

    def test_only_choice_no_gradient(self):
        """只有一个合法动作时不记录 step——这种决策点学不到东西。"""
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(net, training=True)
        game = Game(rng=random.Random(0))
        game.hands[0] = []
        obs = game.observe()
        # 只有结束回合一个选项
        self.assertEqual(len(obs.legal), 1)
        agent.choose(obs)
        self.assertEqual(len(agent.trajectory.steps), 0)

    def test_eval_agent_is_deterministic(self):
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        net.eval()
        agent = PolicyAgent(net, training=False)
        eval_agent = agent.eval_agent()
        self.assertFalse(eval_agent.training)

        game = Game(rng=random.Random(0))
        obs = game.observe()
        a1 = agent.choose(obs)
        a2 = eval_agent.choose(obs)
        self.assertEqual(a1, a2)


class TestBatch(unittest.TestCase):
    def test_make_and_evaluate(self):
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        net.eval()

        game = Game(rng=random.Random(0))
        agent = PolicyAgent(net, training=True)

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
            log_prob, entropy, values = evaluate_batch(net, batch)
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


class TestGAE(unittest.TestCase):
    """GAE(λ) 的行为约束。最关键的是 λ=1 必须复现引入 GAE 之前的实现。"""

    def test_lambda_one_equals_monte_carlo(self):
        """λ=1 ⇒ 优势 = 折扣回报 − V，价值目标 = 折扣回报。

        这是安全绳：只要这条过，把 --gae-lambda 设成 1.0 就等价于旧行为。
        """
        rng = np.random.default_rng(0)
        for n in (1, 2, 5, 37):
            for reward in (1.0, -1.0, 0.3):
                for gamma in (0.99, 0.9, 1.0):
                    values = rng.normal(size=n).astype(np.float32)
                    adv, target = gae_advantages(values, reward, gamma, lam=1.0)
                    expected_target = discounted_returns(reward, n, gamma)
                    np.testing.assert_allclose(target, expected_target, atol=1e-5)
                    np.testing.assert_allclose(adv, expected_target - values, atol=1e-5)

    def test_lambda_zero_is_one_step_td(self):
        """λ=0 ⇒ 优势就是单步 TD 残差 δ_t。"""
        values = np.array([0.2, -0.5, 0.4], dtype=np.float32)
        gamma, reward = 0.9, 1.0
        adv, _ = gae_advantages(values, reward, gamma, lam=0.0)
        expected = [
            gamma * values[1] - values[0],
            gamma * values[2] - values[1],
            reward - values[2],          # 终局，V(s_T) = 0
        ]
        np.testing.assert_allclose(adv, expected, atol=1e-6)

    def test_target_is_adv_plus_values(self):
        values = np.array([0.1, 0.2, -0.3, 0.5], dtype=np.float32)
        for lam in (0.0, 0.5, 0.95, 1.0):
            adv, target = gae_advantages(values, 1.0, 0.99, lam)
            np.testing.assert_allclose(target, adv + values, atol=1e-6)

    def test_perfect_value_gives_zero_advantage(self):
        """如果 V 恰好等于真实折扣回报，优势应该处处为零——任意 λ 都成立。"""
        n, gamma, reward = 8, 0.99, 1.0
        values = discounted_returns(reward, n, gamma)
        for lam in (0.0, 0.5, 0.95, 1.0):
            adv, _ = gae_advantages(values, reward, gamma, lam)
            np.testing.assert_allclose(adv, np.zeros(n), atol=1e-5)

    def test_shrinking_lambda_shrinks_dependence_on_final_reward(self):
        """λ 越小，早期步骤的优势对终局奖励越不敏感——这正是降方差的机制。"""
        n, gamma = 30, 0.99
        values = np.zeros(n, dtype=np.float32)
        sens = {}
        for lam in (1.0, 0.95, 0.5):
            hi, _ = gae_advantages(values, 1.0, gamma, lam)
            lo, _ = gae_advantages(values, -1.0, gamma, lam)
            sens[lam] = abs(hi[0] - lo[0])       # 第 0 步对终局奖励的敏感度
        self.assertGreater(sens[1.0], sens[0.95])
        self.assertGreater(sens[0.95], sens[0.5])


class TestSaveLoad(unittest.TestCase):
    def test_roundtrip(self):
        net = UnifiedNet(hidden=32, layers=2, norm="layer")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.pt"
            save_agent(str(path), net, meta={"lr": 1e-3})
            agent = load_agent(str(path))
            self.assertEqual(agent.net.hidden, 32)
            self.assertEqual(agent.net.layers, 2)
            self.assertFalse(agent.training)


class TestEndToEnd(unittest.TestCase):
    def test_agent_can_complete_a_game(self):
        """智能体 + 规则对手能正常打完一局。"""
        net = UnifiedNet(hidden=16, layers=1, norm="none")
        agent = PolicyAgent(net, training=True)

        from hearthstone.bots import make_bot

        opponent = make_bot("random", seed=1)
        result = play_game([agent, opponent], rng=random.Random(2))
        self.assertIn(result.winner, (0, 1, None))
        self.assertTrue(result.turns > 0)


if __name__ == "__main__":
    unittest.main()
