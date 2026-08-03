"""人机对战 / 观战。

    python -m hearthstone.play                  # 你 vs rule
    python -m hearthstone.play --opponent greedy
    python -m hearthstone.play --watch          # 观战两个机器人，回车逐步推进
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional

from .bots import BOTS, make_bot
from .game import (
    ATTACK,
    END_TURN,
    HERO,
    PLAY,
    Action,
    Game,
    Minion,
    Observation,
    attack,
    describe,
    play,
)

WIDTH = 62


# ---------------------------------------------------------------- 画面

def render(obs: Observation, hide_hand: bool = False) -> str:
    lines: List[str] = []
    lines.append("=" * WIDTH)
    lines.append(
        f"第 {obs.turn + 1} 个回合    玩家{obs.player} 行动    "
        f"水晶 {obs.mana}/{obs.max_mana}"
    )
    lines.append("-" * WIDTH)

    lines.append(
        f"  对手   英雄 {obs.enemy_hero_health:>3}    "
        f"手牌 {obs.enemy_hand_size:>2}    牌堆 {obs.enemy_deck_size:>2}"
        + (f"    疲劳 {obs.enemy_fatigue}" if obs.enemy_fatigue else "")
    )
    lines.append("    场上 " + _board_str(obs.enemy_board))
    lines.append("  " + "· " * ((WIDTH - 4) // 2))
    lines.append("    场上 " + _board_str(obs.board))
    lines.append(
        f"  自己   英雄 {obs.hero_health:>3}    "
        f"手牌 {len(obs.hand):>2}    牌堆 {obs.deck_size:>2}"
        + (f"    疲劳 {obs.fatigue}" if obs.fatigue else "")
    )

    if not hide_hand:
        lines.append("-" * WIDTH)
        lines.append("  手牌 " + _hand_str(obs))
    return "\n".join(lines)


def _board_str(board: List[Minion]) -> str:
    if not board:
        return "（空）"
    return "   ".join(f"[{i}] {m}" for i, m in enumerate(board))


def _hand_str(obs: Observation) -> str:
    if not obs.hand:
        return "（空）"
    parts = []
    for i, card in enumerate(obs.hand):
        mark = " " if card.cost <= obs.mana else "×"     # × = 现在出不起
        parts.append(f"[{i}]{mark}{card}")
    return "   ".join(parts)


def render_actions(obs: Observation) -> str:
    """把合法动作按类型分组列出来，每个都带一个编号。"""
    groups = {"出牌": [], "攻击": [], "": []}
    for i, action in enumerate(obs.legal):
        label = f"[{i}] {describe(obs, action)}"
        if action.kind == PLAY:
            groups["出牌"].append(label)
        elif action.kind == ATTACK:
            groups["攻击"].append(label)
        else:
            groups[""].append(label)

    lines = []
    for title, items in groups.items():
        if not items:
            continue
        head = f"    {title:<6}" if title else " " * 10
        lines.append(head + _wrap(items, indent=10))
    lines.append(f"    —— 共 {len(obs.legal)} 个候选")
    return "\n".join(lines)


def _wrap(items: List[str], indent: int, per_line: int = 3) -> str:
    rows = [items[i : i + per_line] for i in range(0, len(items), per_line)]
    pad = "\n" + " " * indent
    return pad.join("   ".join(row) for row in rows)


# ---------------------------------------------------------------- 输入

HELP = """
可以输入：
  3           执行 3 号候选动作
  p 2         出手牌 2 号
  a 0 1       用自己 0 号随从攻击对方 1 号随从
  a 0 f       用自己 0 号随从打脸
  e           结束回合
  ?           重新列一遍候选
  q           退出
""".strip()


def parse_input(text: str, obs: Observation) -> Optional[Action]:
    """把一行输入解析成动作。看不懂就抛 ValueError。"""
    parts = text.strip().lower().split()
    if not parts:
        raise ValueError("空输入")

    head = parts[0]
    if head in ("e", "end"):
        return END_TURN

    if head in ("p", "play"):
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError("出牌写成 `p <手牌编号>`")
        return play(int(parts[1]))

    if head in ("a", "atk", "attack"):
        if len(parts) != 3:
            raise ValueError("攻击写成 `a <自己随从> <对方随从|f>`")
        if not parts[1].isdigit():
            raise ValueError(f"看不懂攻击者 {parts[1]!r}")
        target = HERO if parts[2] in ("f", "face", "h", "hero") else None
        if target is None:
            if not parts[2].isdigit():
                raise ValueError(f"看不懂目标 {parts[2]!r}")
            target = int(parts[2])
        return attack(int(parts[1]), target)

    if head.isdigit():
        index = int(head)
        if not 0 <= index < len(obs.legal):
            raise ValueError(f"候选只有 {len(obs.legal)} 个，没有 {index} 号")
        return obs.legal[index]

    raise ValueError(f"看不懂 {text.strip()!r}")


class Human:
    """从命令行读输入的玩家。"""

    name = "你"

    def choose(self, obs: Observation) -> Action:
        print()
        print(render(obs))
        print(render_actions(obs))

        while True:
            try:
                text = input("  > ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\n再见")

            if text in ("q", "quit", "exit"):
                raise SystemExit("再见")
            if text in ("?", "h", "help"):
                print(render_actions(obs))
                print(HELP)
                continue

            try:
                action = parse_input(text, obs)
            except ValueError as err:
                print(f"    {err}")
                continue

            if action not in obs.legal:
                print(f"    {action} 现在不能做（? 看候选）")
                continue
            return action


# ---------------------------------------------------------------- 主流程

def run(opponent: str, seed: Optional[int], first: int, watch: bool) -> None:
    rng = random.Random(seed)
    game = Game(rng=rng, first=first)

    if watch:
        players = [make_bot(opponent, seed=seed), make_bot(opponent, seed=(seed or 0) + 1)]
        names = [f"玩家0({opponent})", f"玩家1({opponent})"]
    else:
        players = [Human(), make_bot(opponent, seed=seed)]
        names = ["你", f"对手({opponent})"]

    print(f"简化版炉石：{names[0]} vs {names[1]}，玩家{first} 先手。")

    last_turn = -1
    while not game.finished:
        obs = game.observe()
        is_human = isinstance(players[obs.player], Human)

        if not is_human and obs.turn != last_turn:
            last_turn = obs.turn
            print()
            print(render(obs, hide_hand=True))
            if watch:
                input("  （回车继续）")

        action = players[obs.player].choose(obs)
        if not is_human:
            print(f"  → {names[obs.player]} {describe(obs, action)}")
        game.step(action)

    result = game.result()
    print()
    print("=" * WIDTH)
    if result.winner is None:
        print(f"平局！血量 {result.hero_health}，共 {result.turns} 个回合")
    else:
        print(f"{names[result.winner]} 获胜！血量 {result.hero_health}，共 {result.turns} 个回合")


def main() -> None:
    parser = argparse.ArgumentParser(description="简化版炉石：人机对战 / 观战")
    parser.add_argument("--opponent", default="rule", choices=sorted(BOTS), help="对手是谁")
    parser.add_argument("--seed", type=int, default=None, help="固定洗牌种子")
    parser.add_argument("--first", type=int, default=0, choices=(0, 1), help="谁先手")
    parser.add_argument("--watch", action="store_true", help="观战两个机器人互打")
    args = parser.parse_args()
    run(args.opponent, args.seed, args.first, args.watch)


if __name__ == "__main__":
    main()
