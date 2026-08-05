"""M0 冒烟测试：验证 orange_stone 绑定能完整打完一局，且同 seed 可逐位复现。

路线图（docs/roadmap.md）M0 的三件事：
1. 从 Python 跑通一局：GameEnv → 循环 legal_actions() + step() vs Greedy bot
2. 确定性：同 seed 两次完局，动作序列与终局 winner 逐位一致
3. 基线吞吐：局/s 与步/s，留作 M4 批量化的对照

用法：
    .venv/bin/python -m tools.orange_stone_smoke              # 默认：单局演示 + 确定性 + 200 局吞吐
    .venv/bin/python -m tools.orange_stone_smoke --games 1000
    .venv/bin/python -m tools.orange_stone_smoke --no-bench   # 只跑演示与确定性
"""

from __future__ import annotations

import argparse
import random
import time

import orange_stone as os

MAX_STEPS = 5000  # 与 EnvConfig::max_steps 一致


def play_game(seed: int, *, perspective: int = 0, rng: random.Random | None = None):
    """打一局：随机策略（Python 侧独立随机源，保证复现）对内置 Greedy bot。

    返回 (动作序列, 终局 winner, 步数)。winner: None=未结束/平局, 0=perspective 玩家, 1=对手。
    """
    env = os.GameEnv(seed=seed, perspective=perspective)
    env.reset(seed=seed)
    rng = rng or random.Random(seed)
    actions: list[tuple[int, str]] = []

    for _ in range(MAX_STEPS):
        legal = env.legal_actions()
        if not legal:
            break
        idx = rng.randrange(len(legal))
        actions.append((idx, legal[idx][1]))
        _, _, done, winner = env.step(idx)
        if done:
            return actions, winner, len(actions)
    # 达到步数上限：按平局处理
    return actions, None, len(actions)


def check_determinism(seed: int, *, games: int = 8) -> bool:
    """同 seed 两次完局，动作序列与终局逐位一致。"""
    ok = True
    for g in range(games):
        s = seed + g
        a1, w1, n1 = play_game(s)
        a2, w2, n2 = play_game(s)
        same = w1 == w2 and n1 == n2 and a1 == a2
        if not same:
            ok = False
            print(f"  ✗ seed {s}: winner {w1} vs {w2}, steps {n1} vs {n2}, actions 一致={a1 == a2}")
        else:
            print(f"  ✓ seed {s}: {n1} 步, winner={w1}, 两次完局逐位一致")
    return ok


def bench(games: int, seed: int) -> dict[str, float]:
    """打 `games` 局，报局/s 与步/s（单进程、随机策略 vs Greedy）。"""
    wins = draws = losses = steps = 0
    t0 = time.perf_counter()
    for g in range(games):
        _, winner, n = play_game(seed + g)
        steps += n
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
    elapsed = time.perf_counter() - t0
    return {
        "games": games,
        "elapsed_s": elapsed,
        "games_per_s": games / elapsed,
        "steps_per_s": steps / elapsed,
        "avg_steps": steps / games,
        "win_rate": wins / games,
        "loss_rate": losses / games,
        "draw_rate": draws / games,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M0 冒烟测试：完局 + 确定性 + 吞吐基线")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--games", type=int, default=200, help="吞吐基准局数")
    ap.add_argument("--no-bench", action="store_true", help="跳过吞吐基准")
    args = ap.parse_args()

    print("=== orange_stone 绑定 ===")
    print(f"version: {os.__version__}, obs_len: {os.GameEnv.obs_len()}")

    print("\n=== 单局演示 (seed=42) ===")
    actions, winner, n = play_game(args.seed)
    print(f"打完一局：{n} 步，winner={winner}（None=平局/未分胜负, 0=视角玩家, 1=对手）")
    print(f"首 5 个动作: {[a[1] for a in actions[:5]]}")

    print("\n=== 确定性检查（同 seed 两次完局逐位对比）===")
    ok = check_determinism(args.seed, games=8)
    if not ok:
        raise SystemExit("✗ 确定性检查失败")
    print("✓ 全部通过")

    if not args.no_bench:
        print(f"\n=== 吞吐基线（{args.games} 局，单进程，随机策略 vs Greedy）===")
        r = bench(args.games, args.seed)
        print(f"耗时 {r['elapsed_s']:.1f}s | {r['games_per_s']:.1f} 局/s | {r['steps_per_s']:.0f} 步/s")
        print(f"平均 {r['avg_steps']:.1f} 步/局 | 胜率 {r['win_rate']:.1%} | 负 {r['loss_rate']:.1%} | 平 {r['draw_rate']:.1%}")
        print("（此数值为 M4 批量化的对照基线，机器/策略不同会有差异）")


if __name__ == "__main__":
    main()
