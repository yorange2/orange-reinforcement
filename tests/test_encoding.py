import random
import unittest

import numpy as np
import torch

from paodekuai.cards import RANKS, parse_card
from paodekuai.combos import classify
from paodekuai.encoding import (ENCODERS, GRID_CHANNEL_NAMES, GRID_CHANNELS,
                                GRID_SIZE, N_RANKS, RawEncoder, make_encoder)
from paodekuai.game import Game
from paodekuai.policy import MoveScorer, PolicyAgent
from tests.test_bots import observation


def cards(*names):
    return [parse_card(name) for name in names]


def grid_of(encoder, obs, row):
    """把某个候选动作的网格还原成 (通道, 点数)。"""
    start, end = encoder.grid_slice
    return encoder.build(obs)[row, start:end].reshape(GRID_CHANNELS, N_RANKS)


class TestEncoderContract(unittest.TestCase):
    def test_every_encoder_reports_a_consistent_layout(self):
        for name in ENCODERS:
            encoder = make_encoder(name)
            self.assertEqual(encoder.name, name)
            self.assertEqual(encoder.state_offset, encoder.dim - encoder.state_dim)
            self.assertGreater(encoder.state_dim, 0)

    def test_output_shape_matches_the_action_set(self):
        obs = Game(rng=random.Random(0)).observe()
        for name in ENCODERS:
            encoder = make_encoder(name)
            x = encoder.build(obs)
            self.assertEqual(x.shape, (len(obs.legal), encoder.dim), name)
            self.assertEqual(x.dtype, np.float32, name)
            self.assertTrue(np.isfinite(x).all(), name)

    def test_state_block_is_shared_by_every_candidate(self):
        obs = Game(rng=random.Random(3)).observe()
        for name in ENCODERS:
            encoder = make_encoder(name)
            x = encoder.build(obs)
            self.assertTrue((x[:, encoder.state_offset :] == x[0, encoder.state_offset :]).all(), name)

    def test_unknown_encoder_is_rejected(self):
        with self.assertRaises(ValueError):
            make_encoder("nope")


class TestRawGrid(unittest.TestCase):
    """原始编码只给事实：牌在哪、还剩多少，不给任何"好不好"的判断。"""

    def setUp(self):
        self.encoder = RawEncoder()

    def test_grid_is_ranks_by_channels(self):
        self.assertEqual(GRID_SIZE, GRID_CHANNELS * N_RANKS)
        self.assertEqual(N_RANKS, len(RANKS))
        self.assertEqual(len(GRID_CHANNEL_NAMES), GRID_CHANNELS)

    def test_hand_channel_counts_the_cards_in_hand(self):
        obs = observation(cards("3", "c3", "h3", "9", "K"))
        grid = grid_of(self.encoder, obs, 0)
        self.assertAlmostEqual(float(grid[0, 0]), 3 / 4)    # 三张 3
        self.assertAlmostEqual(float(grid[0, 6]), 1 / 4)    # 一张 9
        self.assertAlmostEqual(float(grid[0, 1]), 0.0)      # 没有 4

    def test_move_channel_marks_the_cards_being_played(self):
        obs = observation(cards("3", "c3", "h3", "9"))
        played_index = next(i for i, m in enumerate(obs.legal)
                            if m is not None and m.kind == "triple")
        grid = grid_of(self.encoder, obs, played_index)
        self.assertAlmostEqual(float(grid[1, 0]), 3 / 4)
        self.assertAlmostEqual(float(grid[1, 6]), 0.0)

    def test_hand_after_channel_subtracts_the_move(self):
        obs = observation(cards("3", "c3", "h3", "9"))
        for i, move in enumerate(obs.legal):
            grid = grid_of(self.encoder, obs, i)
            np.testing.assert_allclose(grid[2], grid[0] - grid[1], atol=1e-6)

    def test_pass_leaves_the_hand_untouched(self):
        obs = observation(cards("3", "4"), required=classify(cards("A")))
        pass_row = obs.legal.index(None)
        grid = grid_of(self.encoder, obs, pass_row)
        np.testing.assert_allclose(grid[2], grid[0], atol=1e-6)
        self.assertAlmostEqual(float(grid[1].sum()), 0.0)

    def test_required_channel_marks_the_rank_to_beat(self):
        obs = observation(cards("K", "cK"), required=classify(cards("9", "c9")))
        grid = grid_of(self.encoder, obs, 0)
        self.assertAlmostEqual(float(grid[5, 6]), 1.0)   # 9 是第 7 个点数
        self.assertAlmostEqual(float(grid[5].sum()), 1.0)

    def test_unseen_and_played_come_from_public_information(self):
        game = Game(rng=random.Random(5))
        game.step(game.legal_actions()[0])
        obs = game.observe()
        grid = grid_of(self.encoder, obs, 0)
        self.assertAlmostEqual(float(grid[4].sum()) * 4, sum(obs.played_counts.values()))
        self.assertTrue((grid[3] >= 0).all())

    def test_no_heuristics_leak_in(self):
        # raw 的卖点就是不含手工判断，维度对不上说明混进了别的东西
        from paodekuai.encoding import RAW_MOVE_SCALARS, RAW_STATE_SCALARS

        self.assertEqual(self.encoder.dim, GRID_SIZE + len(RAW_MOVE_SCALARS) + len(RAW_STATE_SCALARS))
        for name in ("turns_after", "turns_gain", "breaks_pair", "higher_unseen"):
            self.assertNotIn(name, RAW_MOVE_SCALARS + list(RAW_STATE_SCALARS))


class TestBothEncoder(unittest.TestCase):
    def test_carries_the_handcrafted_and_the_raw_block(self):
        from paodekuai.encoding import (HandcraftedEncoder, RAW_MOVE_SCALARS,
                                        HANDCRAFTED_STATE_OFFSET)

        obs = Game(rng=random.Random(7)).observe()
        both, hand, raw = make_encoder("both"), HandcraftedEncoder(), RawEncoder()
        x, h, r = both.build(obs), hand.build(obs), raw.build(obs)

        cut = HANDCRAFTED_STATE_OFFSET
        np.testing.assert_allclose(x[:, :cut], h[:, :cut])                     # 手工动作块
        np.testing.assert_allclose(x[:, cut : cut + GRID_SIZE], r[:, :GRID_SIZE])  # 网格
        np.testing.assert_allclose(x[:, both.state_offset :], h[:, cut:])      # 手工局面块


class TestScorerWithGrid(unittest.TestCase):
    def test_conv_front_end_only_when_there_is_a_grid(self):
        self.assertIsNone(MoveScorer(dim=42, grid=None).conv)
        self.assertIsNotNone(MoveScorer(dim=96, grid=(0, GRID_SIZE)).conv)

    def test_rejects_a_grid_that_does_not_divide_evenly(self):
        with self.assertRaises(ValueError):
            MoveScorer(dim=50, grid=(0, 7), grid_channels=GRID_CHANNELS)

    def test_scores_every_candidate_for_each_encoder(self):
        for name in ENCODERS:
            encoder = make_encoder(name)
            scorer = MoveScorer(dim=encoder.dim, grid=encoder.grid_slice)
            for n in (1, 5, 80):
                self.assertEqual(scorer(torch.zeros(n, encoder.dim)).shape, (n,), name)

    def test_grid_scoring_stays_independent_per_candidate(self):
        # 和无网格时一样：候选之间不能互相影响，补齐的假零行也不能影响真实行
        torch.manual_seed(0)
        encoder = RawEncoder()
        scorer = MoveScorer(dim=encoder.dim, grid=encoder.grid_slice, hidden=16).eval()
        x = torch.randn(5, encoder.dim)

        with torch.no_grad():
            together = scorer(x)
            alone = torch.cat([scorer(x[i : i + 1]) for i in range(5)])
            padded = scorer(torch.cat([x, torch.zeros(11, encoder.dim)]))[:5]

        torch.testing.assert_close(together, alone)
        torch.testing.assert_close(together, padded)

    def test_agents_play_legally_with_every_encoder(self):
        for name in ENCODERS:
            encoder = make_encoder(name)
            agent = PolicyAgent(MoveScorer(dim=encoder.dim, hidden=16, grid=encoder.grid_slice),
                                encoder=encoder)
            game = Game(rng=random.Random(1))
            while not game.finished:
                obs = game.observe()
                action = agent.choose(obs)
                self.assertIn(action, obs.legal, name)
                game.step(action)


if __name__ == "__main__":
    unittest.main()
