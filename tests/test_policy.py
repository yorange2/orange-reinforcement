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
from paodekuai.policy import (MoveScorer, PolicyAgent, ValueNet,
                              discounted_returns, load_agent, save_agent)
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
        self.assertTrue(step.log_prob.requires_grad)
        self.assertLessEqual(float(step.log_prob.detach()), 0.0)

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
             "--eval-every", "0", "--opponent", "greedy", "--hidden", "16"]
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
            names.update(type(bot).__name__ for bot in train_module.build_opponents("mix", rng, episode))
        self.assertGreater(len(names), 1)


if __name__ == "__main__":
    unittest.main()
