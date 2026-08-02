import os
import tempfile
import unittest

from orange_rl import GridWorld, QLearningAgent
from train import evaluate, train


class TestQLearningAgent(unittest.TestCase):
    def test_q_values_start_at_zero(self):
        agent = QLearningAgent(n_actions=4, seed=0)
        self.assertEqual(agent.q_values("s"), [0.0, 0.0, 0.0, 0.0])

    def test_learn_moves_value_towards_target(self):
        agent = QLearningAgent(n_actions=2, alpha=0.5, gamma=0.9, seed=0)
        td = agent.learn("a", 0, reward=10.0, next_state="b", done=True)
        self.assertEqual(td, 10.0)
        self.assertEqual(agent.q_values("a")[0], 5.0)  # 0 + 0.5 * (10 - 0)

    def test_learn_bootstraps_from_next_state(self):
        agent = QLearningAgent(n_actions=2, alpha=1.0, gamma=0.5, seed=0)
        agent.q["b"] = [4.0, 8.0]
        agent.learn("a", 1, reward=1.0, next_state="b", done=False)
        self.assertEqual(agent.q_values("a")[1], 5.0)  # 1 + 0.5 * 8

    def test_greedy_action_picks_best(self):
        agent = QLearningAgent(n_actions=3, epsilon=0.0, seed=0)
        agent.q["s"] = [1.0, 7.0, 3.0]
        self.assertEqual(agent.act("s"), 1)
        self.assertEqual(agent.act("s", explore=False), 1)

    def test_epsilon_one_explores(self):
        agent = QLearningAgent(n_actions=4, epsilon=1.0, seed=1)
        agent.q["s"] = [0.0, 0.0, 0.0, 99.0]
        chosen = {agent.act("s") for _ in range(50)}
        self.assertGreater(len(chosen), 1)

    def test_epsilon_decays_to_floor(self):
        agent = QLearningAgent(n_actions=2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5, seed=0)
        self.assertAlmostEqual(agent.decay_epsilon(), 0.5)
        for _ in range(20):
            agent.decay_epsilon()
        self.assertEqual(agent.epsilon, 0.1)

    def test_policy_covers_only_visited_states(self):
        agent = QLearningAgent(n_actions=2, seed=0)
        agent.q["seen"] = [0.0, 1.0]
        policy = agent.policy(["seen", "unseen"])
        self.assertEqual(policy, {"seen": 1})

    def test_save_and_load_roundtrip(self):
        agent = QLearningAgent(n_actions=4, seed=0)
        agent.learn((0, 0), 2, reward=5.0, next_state=(0, 1), done=True)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "qtable.json")
            agent.save(path)
            restored = QLearningAgent.load(path)

        self.assertEqual(restored.n_actions, agent.n_actions)
        self.assertEqual(restored.q_values((0, 0)), agent.q_values((0, 0)))

    def test_rejects_zero_actions(self):
        with self.assertRaises(ValueError):
            QLearningAgent(n_actions=0)


class TestTrainingLoop(unittest.TestCase):
    def test_agent_learns_to_reach_the_goal(self):
        env = GridWorld(max_steps=100)
        agent = QLearningAgent(n_actions=env.n_actions, epsilon_decay=0.99, seed=0)

        rewards = train(env, agent, episodes=500, verbose=False)
        total, path, outcome = evaluate(env, agent)

        self.assertEqual(len(rewards), 500)
        self.assertEqual(outcome, "goal")
        self.assertEqual(path[0], env.start)
        self.assertEqual(path[-1], env.goal)
        self.assertGreater(total, 0)

        early = sum(rewards[:50]) / 50
        late = sum(rewards[-50:]) / 50
        self.assertGreater(late, early)


if __name__ == "__main__":
    unittest.main()
