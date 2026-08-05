"""v7 特征的守卫：维度布局、共享局面尾、动作块结构。"""

from __future__ import annotations

import unittest

from .. import decks
from ..env import Env
from ..features import ACTION_DIM, FEATURE_DIM, STATE_DIM, STATE_OFFSET, batch_features


def make_state(seed: int = 0):
    env = Env(deck=decks.vanilla(), seed=seed)
    env.reset(seed=seed)
    # 先手出个 0 费随从，让局面有随从可打
    actions = env.legal_actions()
    plays = [a for a in actions if a.kind == "play" and a.card_index >= 0]
    if plays:
        env.step(plays[0])
    return env


class TestDims(unittest.TestCase):
    def test_dim_layout(self):
        self.assertEqual(FEATURE_DIM, ACTION_DIM + STATE_DIM)
        self.assertEqual(STATE_OFFSET, ACTION_DIM)
        # v7+ 定版 223 维（47 动作 + 176 局面，M5 卡面文本块），见 features.py 头注释
        self.assertEqual(FEATURE_DIM, 223)

    def test_batch_features_shape(self):
        env = make_state()
        obs = env.observe()
        actions = env.legal_actions()
        rows = batch_features(obs, actions, going_first=1.0)
        self.assertEqual(rows.shape, (len(actions), FEATURE_DIM))
        self.assertEqual(rows.dtype, "float32")

    def test_state_tail_is_shared(self):
        env = make_state()
        obs = env.observe()
        actions = env.legal_actions()
        rows = batch_features(obs, actions, going_first=1.0)
        # 局面尾是共享的（所有候选同一份），动作块不同
        self.assertTrue((rows[0, STATE_OFFSET:] == rows[:, STATE_OFFSET:]).all())

    def test_action_blocks_differ(self):
        env = make_state()
        obs = env.observe()
        actions = env.legal_actions()
        rows = batch_features(obs, actions, going_first=1.0)
        if len(actions) > 2:
            self.assertFalse((rows[0, :STATE_OFFSET] == rows[1, :STATE_OFFSET]).all())

    def test_going_first_is_encoded(self):
        env = make_state()
        obs = env.observe()
        actions = env.legal_actions()
        first = batch_features(obs, actions, going_first=1.0)
        second = batch_features(obs, actions, going_first=0.0)
        # 先后手在 S_OTHER 的倒数第二维（bias 前一位）
        idx = STATE_OFFSET + STATE_DIM - 2
        self.assertEqual(first[0, idx], 1.0)
        self.assertEqual(second[0, idx], 0.0)

    def test_keyword_onehot_has_5_dims(self):
        env = make_state()
        obs = env.observe()
        plays = [a for a in env.legal_actions() if a.kind == "play"]
        if not plays:
            self.skipTest("这局没有可出的牌")
        a = plays[0]
        feats = batch_features(obs, [a], going_first=1.0)[0]
        # A_CARD = 费/攻/血 + 5 关键词：位置 3..7 是关键词块
        kw = feats[3 + 3: 3 + 3 + 5]
        self.assertEqual(len(kw), 5)


if __name__ == "__main__":
    unittest.main()
