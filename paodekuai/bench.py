#!/usr/bin/env python3
"""统一口径的胜率基准：每个选手都坐一家，对面两家是同一种规则对手。

三家游戏的随机基准是 33.3%；每局轮转座位，抵消 ♦3 的先手优势；同一 seed 下
所有选手打的是同一批牌局，所以横向可比。

用法：
    python -m paodekuai.bench                              # 只比规则对手
    python -m paodekuai.bench --model paodekuai/models/agent.pt      # 把模型也放进来
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .arena import evaluate
from .bots import make_bot

OPPONENTS = ["random", "greedy", "rule"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], metavar="PATH",
                        help="模型权重路径，可重复给多个，都会参与评测")
    parser.add_argument("--games", type=int, default=1500, help="每个组合打多少局（默认 1500）")
    parser.add_argument("--seed", type=int, default=999, help="随机种子")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    contenders = []
    for path in args.model:
        from .policy import load_agent

        agent = load_agent(path, device=args.device)
        label = f"{path.split('/')[-1]}({agent.scorer.n_params / 1000:.0f}k)"
        contenders.append((label, agent))
    contenders.extend((name, make_bot(name, seed=args.seed)) for name in reversed(OPPONENTS))

    width = max(len(name) for name, _ in contenders) + 2
    header = "".join(f"vs 2x{opponent:<10}" for opponent in OPPONENTS)
    print(f"每格 {args.games} 局，轮转座位，随机基准 33.3%\n")
    print(f"{'选手':<{width}}{header}")
    print("-" * (width + 15 * len(OPPONENTS)))

    for name, player in contenders:
        cells = []
        for opponent in OPPONENTS:
            stats = evaluate(player, opponent, games=args.games, seed=args.seed)
            cells.append(f"{stats.win_rate * 100:5.1f}%".ljust(15))
        print(f"{name:<{width}}{''.join(cells)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
