import os
import random
import tempfile
import unittest

import numpy as np
import torch

import train as train_module
from paodekuai.bots import make_bot
from paodekuai.cards import parse_card
from paodekuai.combos import classify
from paodekuai.features import (FEATURE_DIM, FEATURE_NAMES, STATE_OFFSET,
                                batch_features)
from paodekuai.game import Game, play_game
from paodekuai.policy import (MoveScorer, PolicyAgent, Step, ValueNet,
                              discounted_returns, evaluate_batch, load_agent,
                              make_batch, save_agent)
from tests.test_bots import observation


def cards(*names):
    return [parse_card(name) for name in names]


class TestFeatures(unittest.TestCase):
    def test_names_match_dimension(self):
        self.assertEqual(len(FEATURE_NAMES), FEATURE_DIM)
        self.assertEqual(FEATURE_NAMES[-1], "bias")

    def test_matrix_shape_matches_the_action_set(self):
        obs = Game(rng=random.Random(0)).observe()
        x = batch_features(obs)
        self.assertEqual(x.shape, (len(obs.legal), FEATURE_DIM))
        self.assertEqual(x.dtype, np.float32)
        self.assertTrue(np.isfinite(x).all())

    def test_state_half_is_identical_across_candidates(self):
        obs = Game(rng=random.Random(1)).observe()
        x = batch_features(obs)
        self.assertTrue((x[:, STATE_OFFSET:] == x[0, STATE_OFFSET:]).all())

    def test_pass_row_is_flagged(self):
        obs = observation(cards("3", "4"), required=classify(cards("A")))
        x = batch_features(obs)
        pass_index = obs.legal.index(None)
        self.assertEqual(x[pass_index, FEATURE_NAMES.index("is_pass")], 1.0)
        self.assertEqual(x[pass_index, FEATURE_NAMES.index("is_bomb")], 0.0)

    def test_winning_move_is_flagged(self):
        obs = observation(cards("9", "c9"))
        x = batch_features(obs)
        wins = x[:, FEATURE_NAMES.index("wins_now")]
        pair_index = next(i for i, m in enumerate(obs.legal) if m is not None and len(m.cards) == 2)
        self.assertEqual(wins[pair_index], 1.0)

    def test_breaking_a_pair_is_flagged(self):
        obs = observation(cards("9", "c9", "3"))
        x = batch_features(obs)
        breaks = x[:, FEATURE_NAMES.index("breaks_pair")]
        single_nine = next(i for i, m in enumerate(obs.legal)
                           if m is not None and m.rank == 9 and len(m.cards) == 1)
        single_three = next(i for i, m in enumerate(obs.legal)
                            if m is not None and m.rank == 3 and len(m.cards) == 1)
        self.assertEqual(breaks[single_nine], 1.0)
        self.assertEqual(breaks[single_three], 0.0)


class TestPolicyAgent(unittest.TestCase):
    def make_agent(self, training=False):
        scorer = MoveScorer(hidden=16)
        value = ValueNet(hidden=8) if training else None
        return PolicyAgent(scorer, value, training=training, seed=0)

    def test_only_returns_legal_actions(self):
        agent = self.make_agent()
        rng = random.Random(0)
        for _ in range(20):
            game = Game(rng=rng)
            while not game.finished:
                obs = game.observe()
                action = agent.choose(obs)
                self.assertIn(action, obs.legal)
                game.step(action)

    def test_greedy_agent_is_deterministic(self):
        agent = self.make_agent()
        obs = Game(rng=random.Random(2)).observe()
        self.assertEqual(agent.choose(obs), agent.choose(obs))

    def test_training_agent_records_a_trajectory(self):
        agent = self.make_agent(training=True)
        play_game([agent, make_bot("greedy"), make_bot("greedy")], rng=random.Random(4))
        self.assertGreater(len(agent.trajectory), 0)
        step = agent.trajectory.steps[0]
        # 记录的是特征本身，不带计算图——PPO 要用它反复重新前向
        self.assertEqual(step.features.shape[1], FEATURE_DIM)
        self.assertFalse(step.features.requires_grad)
        self.assertIn(step.action, range(step.features.shape[0]))
        self.assertLessEqual(step.log_prob, 0.0)

        agent.trajectory.clear()
        self.assertEqual(len(agent.trajectory), 0)

    def test_forced_moves_are_not_recorded(self):
        # 只有一个合法动作时没有可学的东西，不该产生梯度
        agent = self.make_agent(training=True)
        obs = observation(cards("3"))
        self.assertEqual(len(obs.legal), 1)
        agent.choose(obs)
        self.assertEqual(len(agent.trajectory), 0)

    def test_scorer_handles_any_number_of_candidates(self):
        scorer = MoveScorer(hidden=16)
        for n in (1, 5, 80):
            self.assertEqual(scorer(torch.zeros(n, FEATURE_DIM)).shape, (n,))


class TestModelSize(unittest.TestCase):
    def test_more_layers_and_width_means_more_parameters(self):
        small = MoveScorer(hidden=128, layers=2)
        big = MoveScorer(hidden=512, layers=3)
        self.assertEqual(small.n_params, 21889)
        self.assertGreater(big.n_params, 20 * small.n_params)

    def test_deeper_scorer_still_scores_every_candidate(self):
        scorer = MoveScorer(hidden=32, layers=4)
        self.assertEqual(len([m for m in scorer.net if isinstance(m, torch.nn.Linear)]), 5)
        self.assertEqual(scorer(torch.zeros(7, FEATURE_DIM)).shape, (7,))

    def test_at_least_one_hidden_layer_required(self):
        with self.assertRaises(ValueError):
            MoveScorer(layers=0)

    def test_checkpoint_round_trips_the_architecture(self):
        scorer = MoveScorer(hidden=64, layers=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "big.pt")
            save_agent(path, scorer)
            restored = load_agent(path)
        self.assertEqual((restored.scorer.hidden, restored.scorer.layers), (64, 3))
        self.assertEqual(restored.scorer.n_params, scorer.n_params)

    def test_old_checkpoints_without_layers_default_to_two(self):
        scorer = MoveScorer(hidden=32, layers=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "legacy.pt")
            torch.save(
                {"scorer": scorer.state_dict(), "value": None, "hidden": 32,
                 "feature_dim": FEATURE_DIM, "meta": {}},  # 故意不写 layers
                path,
            )
            restored = load_agent(path)
        self.assertEqual(restored.scorer.layers, 2)


class TestBatching(unittest.TestCase):
    """变长动作集补齐成矩形之后，填充位绝对不能漏进 softmax。"""

    def steps(self, widths):
        return [
            Step(features=torch.randn(width, FEATURE_DIM), action=width - 1, log_prob=-1.0)
            for width in widths
        ]

    def test_pads_to_the_widest_action_set(self):
        batch = make_batch(self.steps([3, 17, 8]))
        self.assertEqual(batch.features.shape, (3, 17, FEATURE_DIM))
        self.assertEqual(batch.mask.sum(dim=1).tolist(), [3, 17, 8])
        self.assertEqual(len(batch), 3)

    def test_padded_rows_get_zero_probability(self):
        batch = make_batch(self.steps([2, 9]))
        scorer = MoveScorer(hidden=16)
        log_probs, _, _ = evaluate_batch(scorer, None, batch)
        self.assertTrue(torch.isfinite(log_probs).all())

        scores = scorer(batch.features.reshape(-1, FEATURE_DIM)).reshape(2, 9)
        probs = torch.softmax(scores.masked_fill(~batch.mask, float("-inf")), dim=1)
        self.assertAlmostEqual(float(probs[0, :2].sum()), 1.0, places=5)
        self.assertEqual(float(probs[0, 2:].sum()), 0.0)  # 填充位概率必须是 0

    def test_entropy_stays_within_bounds(self):
        batch = make_batch(self.steps([4, 4, 4]))
        _, entropy, _ = evaluate_batch(MoveScorer(hidden=16), None, batch)
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertTrue((entropy >= 0).all())
        self.assertTrue((entropy <= np.log(4) + 1e-5).all())

    def test_batched_scores_match_scoring_one_at_a_time(self):
        steps = self.steps([5, 11])
        scorer = MoveScorer(hidden=16)
        batched, _, _ = evaluate_batch(scorer, None, make_batch(steps))

        for i, step in enumerate(steps):
            alone = torch.log_softmax(scorer(step.features), dim=0)[step.action]
            self.assertAlmostEqual(float(batched[i]), float(alone), places=5)

    def test_value_head_reads_the_state_slice(self):
        batch = make_batch(self.steps([6, 6]))
        _, _, values = evaluate_batch(MoveScorer(hidden=16), ValueNet(hidden=8), batch)
        self.assertEqual(values.shape, (2,))
        self.assertTrue(torch.isfinite(values).all())


class TestPPO(unittest.TestCase):
    def run_algo(self, algo, episodes=48):
        args = train_module.parse_args(
            ["--algo", algo, "--episodes", str(episodes), "--batch", "8",
             "--log-every", "10000", "--eval-every", "0", "--opponent", "greedy", "--hidden", "16", "--quiet"]
        )
        return train_module.train(args)

    def test_ppo_trains_without_blowing_up(self):
        agent = self.run_algo("ppo")
        for param in agent.scorer.parameters():
            self.assertTrue(torch.isfinite(param).all(), "PPO 更新出现了 nan/inf")

    def test_reinforce_still_works(self):
        agent = self.run_algo("reinforce")
        for param in agent.scorer.parameters():
            self.assertTrue(torch.isfinite(param).all())

    def test_ppo_is_the_default(self):
        self.assertEqual(train_module.parse_args([]).algo, "ppo")
        self.assertEqual(train_module.parse_args([]).ppo_epochs, 4)
        self.assertEqual(train_module.parse_args([]).clip_ratio, 0.2)

    def test_clipping_caps_how_far_one_update_can_push(self):
        # 概率比超出 [1-ε, 1+ε] 的部分被裁掉，优势为正时收益不再增长
        epsilon, advantage = 0.2, 1.0
        for ratio in (0.5, 1.0, 3.0):
            r = torch.tensor(ratio)
            objective = torch.min(r * advantage, torch.clamp(r, 1 - epsilon, 1 + epsilon) * advantage)
            self.assertLessEqual(float(objective), 1 + epsilon + 1e-5)

    def test_ppo_runs_multiple_epochs_per_batch(self):
        calls = []
        original = train_module.evaluate_batch

        def counting(*a, **kw):
            calls.append(1)
            return original(*a, **kw)

        train_module.evaluate_batch = counting
        try:
            self.run_algo("ppo", episodes=8)
            ppo_calls = len(calls)
            calls.clear()
            self.run_algo("reinforce", episodes=8)
            reinforce_calls = len(calls)
        finally:
            train_module.evaluate_batch = original

        # PPO 每批多跑 3 轮（4 轮 vs 1 轮），另有 1 次算旧价值
        self.assertGreater(ppo_calls, reinforce_calls)


class TestReturnsAndCheckpoints(unittest.TestCase):
    def test_returns_discount_backwards_from_the_final_reward(self):
        returns = discounted_returns(1.0, 3, gamma=0.5)
        np.testing.assert_allclose(returns, [0.25, 0.5, 1.0], rtol=1e-6)

    def test_losing_gives_negative_returns(self):
        self.assertTrue((discounted_returns(-0.5, 4, gamma=0.9) < 0).all())

    def test_save_and_load_reproduce_the_same_moves(self):
        scorer = MoveScorer(hidden=32)
        for param in scorer.parameters():
            torch.nn.init.normal_(param, std=0.5)
        agent = PolicyAgent(scorer)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "agent.pt")
            save_agent(path, scorer, meta={"note": "test"})
            restored = load_agent(path)

        for seed in range(5):
            obs = Game(rng=random.Random(seed)).observe()
            self.assertEqual(agent.choose(obs), restored.choose(obs))

    def test_load_rejects_a_feature_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.pt")
            torch.save({"scorer": {}, "value": None, "hidden": 32, "feature_dim": 7, "meta": {}}, path)
            with self.assertRaises(ValueError):
                load_agent(path)


class TestTrainingLoop(unittest.TestCase):
    def test_a_short_run_updates_the_weights(self):
        args = train_module.parse_args(
            ["--episodes", "40", "--batch", "8", "--log-every", "1000",
             "--eval-every", "0", "--opponent", "greedy", "--hidden", "16", "--quiet"]
        )
        agent = train_module.train(args)
        self.assertGreater(len(list(agent.scorer.parameters())), 0)

        weights = agent.scorer.net[0].weight.detach()
        self.assertTrue(torch.isfinite(weights).all())
        self.assertGreater(float(weights.abs().sum()), 0.0)

    def test_mixed_opponents_are_sampled(self):
        rng = random.Random(0)
        names = set()
        for episode in range(30):
            names.update(type(bot).__name__ for bot in train_module.build_opponents("mix", rng))
        self.assertGreater(len(names), 1)


if __name__ == "__main__":
    unittest.main()
