#!/usr/bin/env python3
"""统一口径的胜率基准：先后手轮换，同一 seed 下所有选手打同一批牌局，横向可比。

双人游戏的随机基准是 50%。

用法：
    python -m hearthstone.bench                              # 只比规则对手
    python -m hearthstone.bench --model hearthstone/models/agent.pt   # 加上模型
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .arena import evaluate
from .bots import make_bot

OPPONENTS = ["random", "greedy", "rule"]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", default=[], metavar="PATH",
                        help="模型权重路径，可重复给多个")
    parser.add_argument("--games", type=int, default=600, help="每个组合打多少局（默认 600）")
    parser.add_argument("--seed", type=int, default=999, help="随机种子")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    contenders = []
    for path in args.model:
        from .policy import load_agent

        agent = load_agent(path, device=args.device)
        label = f"{path.split('/')[-1]}({agent.net.n_params / 1000:.0f}k)"
        contenders.append((label, agent))
    contenders.extend((name, make_bot(name, seed=args.seed)) for name in reversed(OPPONENTS))

    width = max(len(name) for name, _ in contenders) + 2
    header = "".join(f"vs {opponent:<10}" for opponent in OPPONENTS)
    print(f"每格 {args.games} 局，先后手轮换，随机基准 50%\n")
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
