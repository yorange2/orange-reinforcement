#!/usr/bin/env python3
"""导出真实对局中的局面，供 tools/parity_check.mjs 核对 JS 移植是否走样。

每个局面记下 Python 算出的合法动作、特征矩阵和网络打分，JS 那边从同样的局面
出发重算一遍，三者必须逐一对上。牌统一编码成整数 rank * 4 + suit。
"""

from __future__ import annotations

import argparse
import json
import random
from typing import List, Optional

import torch

from paodekuai.bots import make_bot
from paodekuai.features import batch_features
from paodekuai.game import Game
from paodekuai.policy import load_agent

card_id = lambda card: card.rank * 4 + card.suit  # noqa: E731


def dump_move(move) -> Optional[List[int]]:
    return None if move is None else [card_id(c) for c in move.cards]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=60, help="跑多少局取样")
    parser.add_argument("--model", default="models/agent.pt")
    parser.add_argument("--out", default="tools/parity_cases.json")
    args = parser.parse_args(argv)

    agent = load_agent(args.model)
    rng = random.Random(20260803)
    cases = []

    for game_index in range(args.games):
        game = Game(rng=rng)
        # 混着不同风格的对手，好覆盖到各种局面
        bots = [make_bot(name) for name in ("rule", "greedy", "random")]
        rng.shuffle(bots)

        while not game.finished:
            obs = game.observe()
            x = batch_features(obs)
            with torch.no_grad():
                scores = agent.scorer(torch.from_numpy(x))

            cases.append({
                "player": obs.player,
                "hand": sorted(card_id(c) for c in obs.hand),
                "hand_sizes": list(obs.hand_sizes),
                "required": dump_move(obs.required),
                "required_kind": None if obs.required is None else obs.required.kind,
                "required_length": None if obs.required is None else obs.required.length,
                "required_rank": None if obs.required is None else obs.required.rank,
                "leader": obs.leader,
                "played_counts": sorted(obs.played_counts.items()),
                "trick": obs.trick,
                "first_move": game.first_move,
                "legal": [dump_move(m) for m in obs.legal],
                "features": [[round(float(v), 6) for v in row] for row in x],
                "scores": [round(float(v), 4) for v in scores],
            })
            game.step(bots[obs.player].choose(obs))

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(cases, fh, separators=(",", ":"))

    import os

    print(f"导出 {len(cases)} 个局面 -> {args.out} "
          f"({os.path.getsize(args.out) / 1024 / 1024:.1f} MB)")
    print(f"合法动作总数 {sum(len(c['legal']) for c in cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
