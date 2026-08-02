#!/usr/bin/env python3
"""人机对战：你坐一家，另外两家是规则机器人或训练好的模型。

用法：
    python play.py                          # 对两个 rule 机器人
    python play.py --opponent greedy        # 换个对手
    python play.py --model models/agent.pt  # 跟训练好的模型打
    python play.py --watch --model models/agent.pt   # 只看机器互打，不用自己出牌
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional

from paodekuai.bots import make_bot
from paodekuai.cards import hand_to_str
from paodekuai.game import Action, Observation, play_game


class HumanPlayer:
    """从命令行读取出牌。"""

    name = "human"

    def choose(self, obs: Observation) -> Action:
        print("\n" + "=" * 66)
        for i in obs.opponents():
            print(f"  玩家{i}: 剩 {obs.hand_sizes[i]} 张")
        if obs.required is None:
            print("  当前：轮到你自由出牌")
        else:
            print(f"  当前要压：{obs.required}")

        print(f"\n  你的手牌: {hand_to_str(obs.hand)}")
        options = self._menu(obs)

        while True:
            raw = input("  出牌（输入编号，或 ? 看全部选项）: ").strip()
            if raw == "?":
                self._print_all(obs)
                continue
            if not raw.isdigit() or not 0 <= int(raw) < len(options):
                print("  请输入列表里的编号")
                continue
            return options[int(raw)]

    def _menu(self, obs: Observation, limit: int = 12) -> List[Action]:
        """合法动作可能有几十个，先展示一批有代表性的。"""
        plays = [m for m in obs.legal if m is not None]
        plays.sort(key=lambda m: (len(m.cards), m.rank))
        shown: List[Action] = plays[:limit]
        if None in obs.legal:
            shown.append(None)

        for i, move in enumerate(shown):
            print(f"    [{i}] {'过' if move is None else move}")
        if len(plays) > limit:
            print(f"    ... 还有 {len(plays) - limit} 种，输入 ? 查看全部")
        return shown

    def _print_all(self, obs: Observation) -> None:
        for i, move in enumerate(obs.legal):
            print(f"    [{i}] {'过' if move is None else move}")


def make_player(name: str, model: Optional[str], device: str):
    if name == "model":
        if not model:
            raise SystemExit("用 --model 指定权重文件才能让模型上场")
        from paodekuai.policy import load_agent

        return load_agent(model, device=device)
    return make_bot(name)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opponent", default="rule", choices=["random", "greedy", "rule", "model"],
                        help="两个对手的类型（默认 rule）")
    parser.add_argument("--model", help="模型权重路径；--opponent model 或 --watch 时用得上")
    parser.add_argument("--watch", action="store_true", help="不参与，只观战一局")
    parser.add_argument("--seat", type=int, default=0, help="你坐第几家（0-2）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    players = [make_player(args.opponent, args.model, args.device) for _ in range(3)]

    if args.watch:
        if args.model:
            from paodekuai.policy import load_agent

            players[0] = load_agent(args.model, device=args.device)
            print("玩家0 = 模型，其余为规则对手\n")
        result = play_game(players, rng=rng, verbose=True)
        print("\n".join(result.log))
        return 0

    seat = args.seat % 3
    players[seat] = HumanPlayer()
    print(f"你是玩家{seat}，对手是 2 个 {args.opponent} 机器人。持 ♦3 者先出，首手必须带 ♦3。")

    result = play_game(players, rng=rng, verbose=True)
    print("\n" + "=" * 66)
    print("\n".join(result.log[-12:]))
    print("\n你赢了！" if result.winner == seat else f"\n玩家{result.winner} 赢了，你还剩 {result.remaining[seat]} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
