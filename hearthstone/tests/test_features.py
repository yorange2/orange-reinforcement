import random
import unittest

import numpy as np

from hearthstone.features import (
    FEATURE_DIM,
    STATE_DIM,
    STATE_OFFSET,
    action_features,
    batch_features,
    state_features,
)
from hearthstone.game import END_TURN, Game, attack, play


class TestDimensions(unittest.TestCase):
    def test_state_offset_consistent(self):
        self.assertEqual(STATE_OFFSET + STATE_DIM, FEATURE_DIM)

    def test_batch_features_shape(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        mat = batch_features(obs)
        self.assertEqual(mat.shape, (len(obs.legal), FEATURE_DIM))

    def test_batch_features_is_float32(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        self.assertEqual(batch_features(obs).dtype, np.float32)


class TestActionFeatures(unittest.TestCase):
    def test_end_action(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = action_features(obs, END_TURN)
        self.assertEqual(len(feats), STATE_OFFSET)
        # end is one-hot: [0, 0, 1]
        self.assertEqual(feats[:3], [0.0, 0.0, 1.0])

    def test_play_action(self):
        game = Game(rng=random.Random(1))
        game.hands[0] = game.hands[0][:1]  # keep just one card we can afford
        game.mana[0] = 10
        obs = game.observe()
        play_acts = [a for a in obs.legal if a.kind == "play"]
        if play_acts:
            feats = action_features(obs, play_acts[0])
            self.assertEqual(len(feats), STATE_OFFSET)
            self.assertEqual(feats[:3], [1.0, 0.0, 0.0])

    def test_attack_action(self):
        game = Game(rng=random.Random(2))
        # give player 0 a charge minion and player 1 something to attack
        from hearthstone.cards import CardDef

        charger = CardDef("c", 1, 2, 1, ("冲锋",))
        game.hands[0] = [charger]
        game.mana[0] = 10
        game.step(play(0))
        obs = game.observe()
        atk_acts = [a for a in obs.legal if a.kind == "attack"]
        if atk_acts:
            feats = action_features(obs, atk_acts[0])
            self.assertEqual(len(feats), STATE_OFFSET)
            self.assertEqual(feats[:3], [0.0, 1.0, 0.0])


class TestStateFeatures(unittest.TestCase):
    def test_state_features_length(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = state_features(obs)
        self.assertEqual(len(feats), STATE_DIM)

    def test_state_constant_across_actions(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        mat = batch_features(obs)
        if len(obs.legal) > 1:
            for i in range(1, len(obs.legal)):
                np.testing.assert_array_equal(
                    mat[0, STATE_OFFSET:], mat[i, STATE_OFFSET:]
                )

    def test_state_bias_is_one(self):
        game = Game(rng=random.Random(0))
        obs = game.observe()
        feats = state_features(obs)
        self.assertEqual(feats[-1], 1.0)


if __name__ == "__main__":
    unittest.main()
