#!/usr/bin/env python3
"""三方混战：让任意三个选手同桌打，看谁真的强。

每副牌都按 6 种座位排列各打一遍，牌运和 ♦3 先手完全对消，所以三个人的
基准胜率都是 33.3%。

选手写法：random / greedy / rule / model:权重路径

用法：
    python duel.py model:models/agent_big.pt model:models/agent.pt rule
    python duel.py model:models/agent_big.pt model:models/agent.pt   # 一大 vs 两小
"""

from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

from paodekuai.arena import match
from paodekuai.bots import make_bot


def make_player(spec: str, device: str) -> Tuple[str, object]:
    """把 'rule' 或 'model:路径' 变成 (显示名, 选手)。"""
    if spec.startswith("model:"):
        from paodekuai.policy import load_agent

        path = spec.split(":", 1)[1]
        agent = load_agent(path, device=device)
        label = f"{path.split('/')[-1]}({agent.scorer.n_params / 1000:.0f}k)"
        return label, agent
    return spec, make_bot(spec, seed=0)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("players", nargs="+", help="1-3 个选手；不足 3 个时用最后一个补齐")
    parser.add_argument("--deals", type=int, default=300, help="打多少副牌（每副 6 局，默认 300）")
    parser.add_argument("--seed", type=int, default=2024, help="随机种子")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    specs = list(args.players)
    if len(specs) > 3:
        raise SystemExit("最多 3 个选手")
    while len(specs) < 3:
        specs.append(specs[-1])

    players = [make_player(spec, args.device) for spec in specs]
    print("同桌：" + "  vs  ".join(label for label, _ in players))
    print(f"{args.deals} 副牌 x 6 种座位排列 = {args.deals * 6} 局，每人基准胜率 33.3%\n")

    results = match(players, deals=args.deals, seed=args.seed)
    width = max(len(label) for label, _ in results) + 2

    for label, stats in results:
        delta = (stats.win_rate - 1 / 3) * 100
        print(f"  {label:<{width}} 胜率 {stats.win_rate * 100:5.1f}%  ({delta:+5.1f} 个百分点)"
              f"  负局平均剩牌 {stats.avg_remaining:4.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
