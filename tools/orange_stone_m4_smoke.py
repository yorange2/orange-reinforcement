"""M4 冒烟测试：批量与性能（BatchedEnv / battle_batch / 并行训练）。

验证口径（路线图 M4 验收）：
- 确定性：批量与单局逐 seed 结果一致（不因并行丢失）
- 吞吐基准表：battle_batch（引擎 rayon 批量）与 BatchedEnv（Python 批量 +
  线程池）的局/s，对照 M0 基线
- 并行训练：train --parallel N 能跑通且学习（vs random > 90%）

用法：
    .venv/bin/python -m tools.orange_stone_m4_smoke
"""

from __future__ import annotations

import random
import time

from hearthstone_os import decks
from hearthstone_os.batched import BatchedEnv, bench_battle_batch, bench_bots
from hearthstone_os.env import Env

N_DETERMINISM = 12


def section_determinism() -> None:
    """批量与单局逐 seed 一致（确定性不因并行丢失）。"""
    print("=== 确定性 ===")
    # battle_batch：单局结果 == 批量里的同 seed 结果
    import orange_stone as os
    deck = decks.vanilla()
    single = os.battle_batch([7], deck, "greedy")[0]
    batch = os.battle_batch(list(range(N_DETERMINISM)), deck, "greedy")
    assert batch[7] == single, f"battle_batch 逐 seed 不一致: {batch[7]} vs {single}"
    print("  ✓ battle_batch 单局与批量逐 seed 一致")

    # BatchedEnv vs 单局 Env（确定性策略：永远选最后一个动作）
    b = BatchedEnv(N_DETERMINISM, deck, seeds=list(range(N_DETERMINISM)), bot="none")
    for _ in range(3000):
        legal = b.legal_actions()
        if not any(legal):
            break
        b.step([len(l) - 1 if l else 0 for l in legal])
    bw = b.winners()

    def single(seed: int):
        env = Env(deck=deck, seed=seed)
        env.reset(seed=seed)
        while not env.done:
            legal = env.legal_actions()
            if not legal:
                break
            env.step(legal[-1] if len(legal) > 1 else legal[0])
        return env.winner - 1 if env.winner else None

    sw = [single(s) for s in range(N_DETERMINISM)]
    assert sw == bw, f"BatchedEnv 与单局 Env 逐 seed 不一致: {sw} vs {bw}"
    print(f"  ✓ BatchedEnv 与单局 Env 逐 seed 一致（{N_DETERMINISM} 局）")


def section_throughput() -> None:
    """吞吐基准表（M0 基线 → M4）。"""
    print("=== 吞吐（对照 M0 基线 ~970 局/s）===")
    deck = decks.vanilla()
    eps = bench_battle_batch(5000, deck)
    print(f"  引擎批量 battle_batch（rayon 全 Rust）: {eps:.0f} 局/s"
          f"（≈{eps / 970:.1f}× M0 基线 / {eps / 460:.1f}× rosetta）")
    bps = bench_bots(2000, deck, workers=4)
    print(f"  Python 批量 BatchedEnv（4 线程）: {bps:.0f} 局/s"
          f"（≈{bps / 970:.1f}× M0 基线）")


def section_parallel_train() -> None:
    """并行训练：--parallel 8 能跑通且学习。"""
    print("=== 并行训练（--parallel 8，800 局）===")
    from hearthstone_os.train import parse_args, train_parallel
    from hearthstone_os.arena import evaluate

    t0 = time.time()
    args = parse_args(["--episodes", "800", "--parallel", "8", "--seed", "0",
                       "--quiet"])
    agent = train_parallel(args)
    dt = time.time() - t0
    stats = evaluate(agent.eval_agent(), "random", games=100, seed=0)
    print(f"  800 局耗时 {dt:.1f}s（{800 / dt:.0f} 局/s）")
    print(f"  vs random: {stats.win_rate:.1%}")
    assert stats.win_rate > 0.9, f"并行训练没学会打 random（{stats.win_rate:.1%}）"


def main() -> None:
    section_determinism()
    section_throughput()
    section_parallel_train()
    print("\n全部 M4 小节通过 ✓")


if __name__ == "__main__":
    main()
