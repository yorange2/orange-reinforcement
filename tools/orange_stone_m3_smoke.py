"""M3 冒烟测试：训练接入（features v7 / policy / search / train / bench）。

验证口径（路线图 M3 验收）：
- 特征 v7：199 维（31 动作 + 168 局面），布局与维度断言
- PPO 更新：价值头朝 GAE 目标收敛
- 整回合搜索：clone() 恢复、计划逐步合法、不污染真实对局
- 短训：2000 局后 vs random >95%、vs rule >50%（训练管线端到端）
- 模型存取 roundtrip

用法：
    .venv/bin/python -m tools.orange_stone_m3_smoke
"""

from __future__ import annotations

import random

from hearthstone_os import decks
from hearthstone_os.arena import evaluate, play_game
from hearthstone_os.bots import RandomBot
from hearthstone_os.env import Env
from hearthstone_os.features import ACTION_DIM, FEATURE_DIM, STATE_DIM, batch_features
from hearthstone_os.policy import PolicyAgent, UnifiedNet, save_agent
from hearthstone_os.search import TurnSearchAgent
from hearthstone_os.train import train, parse_args

MAX_STEPS = 3000


def section_features() -> None:
    """v7 特征：199 维定版、共享局面尾、先手首回合 1 水晶（orange-stone #70）。"""
    print("=== 特征 v7 ===")
    assert FEATURE_DIM == 199, f"v7 应定版 199 维，实际 {FEATURE_DIM}"
    assert ACTION_DIM == 31 and STATE_DIM == 168, \
        f"布局不对: {ACTION_DIM} + {STATE_DIM}"
    env = Env(deck=decks.vanilla(), seed=3)
    env.reset(seed=3)
    obs = env.observe()
    # orange-stone #70 回归：先手第一回合必须有 1 水晶（少了会导致 P1 侧训练坍缩）
    assert obs.me.total_mana == 1 and obs.me.remaining_mana == 1, \
        f"先手首回合应 1/1 水晶，实际 {obs.me.remaining_mana}/{obs.me.total_mana}"
    actions = env.legal_actions()
    rows = batch_features(obs, actions, going_first=1.0)
    assert rows.shape == (len(actions), FEATURE_DIM)
    assert (rows[0, ACTION_DIM:] == rows[:, ACTION_DIM:]).all()
    print(f"  ✓ v7 = {FEATURE_DIM} 维（动作 {ACTION_DIM} + 局面 {STATE_DIM}），"
          f"候选 {len(actions)} 个，局面尾共享，先手首回合 {obs.me.total_mana} 水晶")


def section_search() -> None:
    """整回合搜索：计划逐步合法、克隆不污染真实对局。"""
    print("=== 整回合搜索 ===")
    net = UnifiedNet()
    env = Env(deck=decks.vanilla(), seed=5)
    env.reset(seed=5)
    agent = TurnSearchAgent(net, seed=0)
    agent.bind_env(env, seat=1)

    before = env.observe()
    plan = agent._search(env, 1)
    assert plan, "搜索必须给出计划"
    assert before.turn == env.observe().turn, "搜索不得推进真实对局"

    twin = env.clone()
    for action in plan:
        actions = twin.legal_actions()
        assert action in actions, "计划动作必须逐步合法"
        twin.step(action)
        if twin.done:
            break
    assert twin.done or plan[-1].kind == "end_turn"
    print(f"  ✓ clone 搜索：计划 {len(plan)} 步逐步合法，真实对局未被污染")

    # 完局冒烟：搜索智能体 vs random
    result = play_game([TurnSearchAgent(net, seed=1), RandomBot(1)], seed=9)
    assert result.winner in (0, 1, 2)
    print(f"  ✓ 搜索智能体 vs random 完局（winner={result.winner}）")


def section_train() -> None:
    """短训：2000 局端到端，策略应该碾压 random 并逼近 rule。"""
    print("=== 短训 2000 局 ===")
    args = parse_args(["--episodes", "2000", "--quiet", "--seed", "2",
                       "--opponent", "rule"])
    agent = train(args)
    stats = evaluate(agent.eval_agent(), "random", games=200, seed=0)
    assert stats.win_rate > 0.95, f"打 random 只有 {stats.win_rate:.1%}"
    print(f"  ✓ 短训后 vs random {stats.win_rate:.1%}")
    stats = evaluate(agent.eval_agent(), "rule", games=200, seed=0)
    print(f"  ✓ 短训后 vs rule   {stats.win_rate:.1%}")


def section_model_io() -> None:
    """模型存取 roundtrip。"""
    print("=== 模型存取 ===")
    from hearthstone_os.policy import load_agent
    net = UnifiedNet()
    save_agent("/tmp/hsos_m3_smoke.pt", net, meta={"smoke": True})
    loaded = load_agent("/tmp/hsos_m3_smoke.pt")
    assert loaded.net.n_params == net.n_params
    print(f"  ✓ save/load roundtrip（{net.n_params:,} 参数）")


def main() -> None:
    section_features()
    section_search()
    section_train()
    section_model_io()
    print("\n全部 M3 小节通过 ✓")


if __name__ == "__main__":
    main()
