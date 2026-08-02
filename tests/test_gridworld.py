import unittest

from orange_rl import GridWorld
from orange_rl.gridworld import ACTIONS


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        self.env = GridWorld(["S..", ".#X", "..G"], max_steps=10)

    def test_layout_parsing(self):
        self.assertEqual(self.env.start, (0, 0))
        self.assertEqual(self.env.goal, (2, 2))
        self.assertEqual(self.env.walls, {(1, 1)})
        self.assertEqual(self.env.pits, {(1, 2)})
        self.assertEqual(self.env.n_actions, len(ACTIONS))

    def test_reset_returns_start(self):
        self.env.step(1)
        self.assertEqual(self.env.reset(), (0, 0))
        self.assertEqual(self.env.steps, 0)

    def test_border_blocks_movement(self):
        self.env.reset()
        state, reward, done, _ = self.env.step(0)  # up, into the border
        self.assertEqual(state, (0, 0))
        self.assertEqual(reward, self.env.step_penalty)
        self.assertFalse(done)

    def test_wall_blocks_movement(self):
        self.env.reset()
        self.env.step(1)  # down to (1, 0)
        state, _, _, _ = self.env.step(3)  # right, into the wall at (1, 1)
        self.assertEqual(state, (1, 0))

    def test_reaching_goal_ends_episode(self):
        self.env.reset()
        for action in (1, 1, 3, 3):  # down, down, right, right
            state, reward, done, info = self.env.step(action)
        self.assertEqual(state, (2, 2))
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "goal")
        self.assertEqual(reward, self.env.step_penalty + self.env.goal_reward)

    def test_pit_ends_episode(self):
        self.env.reset()
        self.env.step(3)  # right to (0, 1)
        self.env.step(3)  # right to (0, 2)
        state, reward, done, info = self.env.step(1)  # down into the pit
        self.assertEqual(state, (1, 2))
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "pit")
        self.assertEqual(reward, self.env.step_penalty + self.env.pit_penalty)

    def test_timeout_ends_episode(self):
        env = GridWorld(["S..", ".#X", "..G"], max_steps=3)
        env.reset()
        for _ in range(2):
            _, _, done, _ = env.step(0)  # bump the border, going nowhere
            self.assertFalse(done)
        _, _, done, info = env.step(0)
        self.assertTrue(done)
        self.assertEqual(info["outcome"], "timeout")

    def test_states_exclude_walls(self):
        states = self.env.states()
        self.assertEqual(len(states), 8)
        self.assertNotIn((1, 1), states)

    def test_invalid_action_rejected(self):
        self.env.reset()
        with self.assertRaises(ValueError):
            self.env.step(9)

    def test_invalid_layouts_rejected(self):
        with self.assertRaises(ValueError):
            GridWorld(["S.", "..G"])  # ragged rows
        with self.assertRaises(ValueError):
            GridWorld(["S.S", "..G"])  # two starts
        with self.assertRaises(ValueError):
            GridWorld(["S..", "..."])  # no goal

    def test_render_marks_agent(self):
        self.env.reset()
        self.assertEqual(self.env.render().splitlines()[0], "A . .")


if __name__ == "__main__":
    unittest.main()
