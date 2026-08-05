"""M1 冒烟测试：逐个验证 orange_stone 绑定层的新增接口（G2~G7）。

每个 PR 在此文件的对应小节追加断言；跑完所有小节才退出 0。

用法：
    .venv/bin/python -m tools.orange_stone_m1_smoke
"""

from __future__ import annotations

import random

import orange_stone as os

# 经典卡池里几张可用的卡（ID 见 orange-stone/src/cards/classic_*.rs）
DECK_10 = [
    "CLASSIC_001",   # Bloodfen Raptor 2/3/2
    "CLASSIC_018",   # Amani Berserker 2/2/3
    "CLASSIC_019",   # Faerie Dragon 2/3/2
    "NEUTRAL_025",   # Core Hound 7/9/5
    "NEUTRAL_B02",   # Murloc Raider 1/2/1
    "NEUTRAL_B09",   # Magma Rager 3/5/1
    "NEUTRAL_B13",   # Oasis Snapjaw 4/2/7
    "NEUTRAL_B19",   # Gurubashi Berserker 5/2/8
    "CLASSIC_006t",  # Murloc Scout 1/1/1
    "NEUTRAL_020t",  # Squire 1/2/2
]


def section_g2() -> None:
    """G2 — 自定义卡组：固定卡组可打完整局、确定性、未知 ID 报错。"""
    print("=== G2 自定义卡组 ===")
    env = os.GameEnv(seed=42, deck=DECK_10)
    env.reset(seed=42)
    obs = env.observation()
    deck_remaining = round(obs[8] * 30), round(obs[9] * 30)
    assert deck_remaining == (7, 7), f"固定 10 张卡组起手 3 张后应剩 7 张，实际 {deck_remaining}"
    print(f"  ✓ 固定卡组起手后双方牌库各剩 {deck_remaining[0]} 张")

    # 打完整局（随机策略 vs Greedy）
    rng = random.Random(42)
    steps = 0
    done = False
    while not done and steps < 3000:
        legal = env.legal_actions()
        if not legal:
            break
        _, _, done, winner = env.step(rng.randrange(len(legal)))
        steps += 1
    assert done or steps >= 3000
    print(f"  ✓ 固定卡组完局：{steps} 步，winner={winner}")

    # 确定性：同 seed 两次完局逐位一致
    def play(s: int):
        e = os.GameEnv(seed=s, deck=DECK_10)
        e.reset(seed=s)
        r = random.Random(s)
        acts = []
        for _ in range(3000):
            legal = e.legal_actions()
            if not legal:
                break
            i = r.randrange(len(legal))
            acts.append(i)
            _, _, d, w = e.step(i)
            if d:
                return acts, w
        return acts, None

    a1, w1 = play(7)
    a2, w2 = play(7)
    assert a1 == a2 and w1 == w2, "同 seed 固定卡组两次完局必须逐位一致"
    print(f"  ✓ 确定性：seed 7 两次完局逐位一致（{len(a1)} 步, winner={w1}）")

    # 未知 ID 报错
    try:
        os.GameEnv(seed=1, deck=["NOT_A_CARD_XYZ"])
        raise AssertionError("未知卡牌 ID 应抛 ValueError")
    except ValueError as e:
        assert "NOT_A_CARD_XYZ" in str(e)
        print(f"  ✓ 未知卡牌 ID 抛 ValueError：{e}")

    # 随机模式仍然可用（deck=None 默认）
    r = os.GameEnv(seed=1).observation()
    assert len(r) == 168
    print("  ✓ 随机卡组模式保留")


def main() -> None:
    print(f"orange_stone {os.__version__} | obs_len={os.GameEnv.obs_len()}")
    section_g2()
    print("\n全部 M1 小节通过 ✓")


if __name__ == "__main__":
    main()
