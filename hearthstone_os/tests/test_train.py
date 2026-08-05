"""训练管线的守卫：GAE、PPO 更新、权重存取、短训冒烟。"""

from __future__ import annotations

import os
import random
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from .. import decks
from ..arena import play_game
from ..bots import RandomBot
from ..env import Env
from ..policy import (
    PolicyAgent,
    UnifiedNet,
    evaluate_batch,
    gae_advantages,
    make_batch,
    load_agent,
    save_agent,
)
from ..train import _update


def make_args(**overrides):
    import argparse
    args = argparse.Namespace(
        algo="ppo", ppo_epochs=4, clip_ratio=0.2, lr=1e-3, gamma=0.99,
        gae_lambda=0.5, entropy_coef=0.01, value_coef=0.5, clip=5.0,
        device="cpu",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestGae(unittest.TestCase):
    def test_lambda_1_equals_discounted_returns(self):
        """λ=1 时 GAE 退化成折扣回报（v6 测试的同一个性质）。

        λ=1 时优势 telescoping 成 `γ^(T-1-t)·R − V(s_t)`，价值目标
        `adv + V` 正好等于折扣回报本身。
        """
        values = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        adv, tgt = gae_advantages(values, final_reward=1.0, gamma=0.99, lam=1.0)
        expected = 1.0 * (0.99 ** np.array([3, 2, 1, 0]))
        np.testing.assert_allclose(tgt, expected, atol=1e-5)
        np.testing.assert_allclose(adv, expected - values, atol=1e-5)

    def test_terminal_reward_only_at_last_step(self):
        values = np.zeros(4, dtype=np.float32)
        adv, _ = gae_advantages(values, final_reward=-0.5, gamma=0.99, lam=0.5)
        self.assertAlmostEqual(adv[-1], -0.5)
        self.assertAlmostEqual(adv[0], -0.5 * (0.99 * 0.5) ** 3)


class TestModelIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        net = UnifiedNet()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            save_agent(path, net, meta={"note": "test"})
            agent = load_agent(path)
        self.assertEqual(agent.net.n_params, net.n_params)
        # 加载后推理正常
        env = Env(deck=decks.vanilla(), seed=0)
        agent.bind_env(env, 1)
        obs = env.observe()
        actions = env.legal_actions()
        chosen = agent.choose(obs, actions)
        self.assertIn(chosen, actions)


class TestUpdate(unittest.TestCase):
    def _collect_steps(self, seed: int = 3, n: int = 16):
        env = Env(deck=decks.vanilla(), seed=seed)
        env.reset(seed=seed)
        agent = PolicyAgent(UnifiedNet(), training=True, seed=seed)
        agent.bind_env(env)
        agent.trajectory.clear()
        rng = random.Random(seed)
        while len(agent.trajectory) < n and not env.done:
            actions = env.legal_actions()
            if not actions:
                break
            obs = env.observe()
            agent.choose(obs, actions)          # 记录决策点进轨迹
            env.step(rng.choice(actions))       # 用随机动作推进
        return agent, env

    def test_update_moves_value_head_toward_targets(self):
        """一次 PPO 更新让价值头朝 GAE 目标靠近。"""
        torch.manual_seed(0)
        net = UnifiedNet()
        args = make_args()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        agent, _ = self._collect_steps()
        steps = agent.trajectory.steps
        self.assertGreaterEqual(len(steps), 8, "决策点太少，测试无效")
        batch = make_batch(steps, args.device)
        episodes = [(len(steps), 1.0, 1)]

        with torch.no_grad():
            _, _, old_values = evaluate_batch(net, batch)
            # 与 _update 同口径的 GAE 目标（λ=0.5）
            _, target_np = gae_advantages(
                old_values.cpu().numpy(), 1.0, args.gamma, args.gae_lambda
            )
            target = torch.from_numpy(target_np).to(args.device)
            mse_before = F.mse_loss(old_values, target).item()

        _update(optimizer, net, steps, episodes, args, args.device)

        with torch.no_grad():
            _, _, new_values = evaluate_batch(net, batch)
            mse_after = F.mse_loss(new_values, target).item()
        self.assertLess(mse_after, mse_before,
                        f"价值头没朝目标靠近：{mse_before:.4f} → {mse_after:.4f}")

    def test_short_train_beats_random(self):
        """短训 200 局后应该明显强于随机（v6 的同一个冒烟口径）。"""
        from ..train import train, parse_args

        args = parse_args(["--episodes", "200", "--quiet", "--seed", "1",
                           "--opponent", "random"])
        agent = train(args)
        wins = 0
        for seed in range(20):
            result = play_game([agent.eval_agent(), RandomBot(seed)], seed=seed)
            if result.winner == 1:
                wins += 1
        self.assertGreater(wins / 20, 0.55, f"短训后打 random 才 {wins}/20")


if __name__ == "__main__":
    unittest.main()
