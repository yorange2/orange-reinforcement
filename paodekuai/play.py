#!/usr/bin/env python3
"""人机对战：你坐一家，另外两家由模型或规则机器人出牌。

默认对手就是训练好的模型，而且每次它出牌都会把**内心戏**摊开——候选牌的打分和概率，
这样你能直观看到它凭什么这么打。

用法：
    python -m paodekuai.play                        # 你 vs 2 个模型（默认）
    python -m paodekuai.play --opponent rule        # 换成规则机器人
    python -m paodekuai.play --hint                 # 每步先看模型建议，再自己决定
    python -m paodekuai.play --watch                # 观战，回车逐步推进
    python -m paodekuai.play --no-explain           # 关掉打分展示，只看牌

轮到你时可以输入：
    编号      出对应的牌
    3 3 3 4   直接报牌（不用管花色，程序会挑一手合法的）
    p         过
    ?         重新列一遍候选
    m         问模型这一手它会怎么打
    q         退出
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from typing import List, Optional, Sequence, Tuple

from .bots import make_bot
from .cards import RANK_NAMES, hand_to_str
from .combos import KIND_NAMES_CN, KINDS, Combo
from .features import attachment_ranks
from .game import Action, Game, Observation

WIDTH = 72
QUIT = object()

HELP = """
  编号      出列表里对应的牌
  3 3 3 4   直接报牌，不用管花色（也可以写成 333 4）
  p         过
  ?         列出全部候选
  m         问模型这一手它会怎么打
  h         看这份帮助
  q         退出
"""


def pad(text: str, width: int) -> str:
    """按显示宽度补空格。中文占两列，直接用 ljust 会对不齐。"""
    shown = sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)
    return text + " " * max(width - shown, 0)


# --------------------------------------------------------------------- 输入


def parse_ranks(text: str) -> List[int]:
    """把 '3 3 3 4' / '333 4' / 'JQKA' 这类输入解析成点数列表。"""
    names = {name.upper(): rank for rank, name in RANK_NAMES.items()}
    ranks: List[int] = []

    for token in text.replace(",", " ").upper().split():
        while token:
            if token.startswith("10"):
                ranks.append(10)
                token = token[2:]
            elif token[0] in names:
                ranks.append(names[token[0]])
                token = token[1:]
            else:
                raise ValueError(f"看不懂的牌: {token[0]}")
    return ranks


def find_move(text: str, obs: Observation) -> Optional[Combo]:
    """按报出的点数找一手合法牌。花色不用管，找不到就返回 None。"""
    want = Counter(parse_ranks(text))
    if not want:
        return None
    for move in obs.legal:
        if move is not None and Counter(card.rank for card in move.cards) == want:
            return move
    return None


# --------------------------------------------------------------------- 展示


def explain(agent, obs: Observation, top: int = 5) -> List[Tuple[Action, float, float]]:
    """模型对每个候选的打分和概率，按概率从高到低。"""
    import torch

    from .features import batch_features

    x = torch.from_numpy(batch_features(obs))
    with torch.no_grad():
        scores = agent.scorer(x)
        probs = torch.softmax(scores, dim=0)

    order = torch.argsort(probs, descending=True)[:top]
    return [(obs.legal[i], float(scores[i]), float(probs[i])) for i in order.tolist()]


def show_scores(rows: Sequence[Tuple[Action, float, float]], title: str) -> None:
    print(f"  {title}")
    for move, score, prob in rows:
        bar = "█" * round(prob * 20)
        text = "过" if move is None else str(move)
        print(f"    {pad(text, 34)}{score:>8.2f}  {prob * 100:>5.1f}% {bar}")


def show_table(game: Game, seat: int, last: dict) -> None:
    print("\n" + "═" * WIDTH)
    for i in range(3):
        who = f"玩家{i}" + ("（你）" if i == seat else "")
        lead = "◀ 当前牌面" if game.leader == i and game.required is not None else ""
        played = last.get(i, "—")
        print(f"  {pad(who, 12)}剩 {len(game.hands[i]):>2} 张   "
              f"上一手: {pad(played, 30)}{lead}")
    print("─" * WIDTH)
    if game.required is None:
        print("  轮到你自由出牌（不能过）")
    else:
        print(f"  要压的牌: {game.required}")


def show_hand(hand) -> None:
    print(f"\n  你的手牌 ({len(hand)} 张): {hand_to_str(hand)}")


def move_label(move: Combo) -> str:
    """只显示牌面，牌型由分组标题给出。带牌的牌型把主体和带牌分开写。"""
    attached = attachment_ranks(move)
    if attached:
        body = [c for c in move.cards if c.rank not in attached]
        kicker = [c for c in move.cards if c.rank in attached]
        return f"{hand_to_str(body)} 带 {hand_to_str(kicker)}"
    return hand_to_str(move.cards)


def show_menu(obs: Observation) -> List[Action]:
    """列出**全部**候选，按牌型分组、逐行排布。返回的顺序就是编号顺序。"""
    plays = [m for m in obs.legal if m is not None]
    shown: List[Action] = []

    for kind in KINDS:
        group = sorted((m for m in plays if m.kind == kind), key=lambda m: (m.length, m.rank))
        if not group:
            continue

        entries = []
        for move in group:
            entries.append((len(shown), move_label(move)))
            shown.append(move)

        head = pad(f"    {KIND_NAMES_CN[kind]}", 14)
        line, used = head, 14
        for index, label in entries:
            cell = f"[{index}] {label}"
            width = sum(2 if ord(ch) > 0x2E7F else 1 for ch in cell) + 3
            if used + width > WIDTH and used > 14:
                print(line)
                line, used = pad("", 14), 14
            line += pad(cell, width)
            used += width
        print(line.rstrip())

    if None in obs.legal:
        print(f"{pad('    过', 14)}[{len(shown)}] 不要")
        shown.append(None)

    print(f"    —— 共 {len(shown)} 个候选")
    return shown


# --------------------------------------------------------------------- 玩家


class HumanPlayer:
    name = "human"

    def __init__(self, hint_agent=None) -> None:
        self.hint_agent = hint_agent

    def choose(self, obs: Observation) -> Action:
        options = show_menu(obs)
        if self.hint_agent is not None:
            show_scores(explain(self.hint_agent, obs, top=3), "模型建议：")

        while True:
            raw = input("  出牌 > ").strip()
            if not raw:
                continue
            if raw in ("q", "quit"):
                return QUIT
            if raw == "?":
                options = show_menu(obs)
                continue
            if raw in ("h", "help"):
                print(HELP)
                continue
            if raw == "m":
                if self.hint_agent is None:
                    print("  没有加载模型，用 --model 指定权重")
                else:
                    show_scores(explain(self.hint_agent, obs, top=5), "模型会这么打：")
                continue
            if raw in ("p", "pass", "过"):
                if None in obs.legal:
                    return None
                print("  自由出牌时不能过")
                continue
            if raw.isdigit() and int(raw) < len(options):
                return options[int(raw)]

            try:
                move = find_move(raw, obs)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            if move is None:
                print("  这手牌出不了（不是合法牌型，或压不过当前的牌）")
                continue
            return move


# --------------------------------------------------------------------- 主循环


def make_player(kind: str, model_path: Optional[str], device: str):
    if kind == "model":
        from .policy import load_agent

        return load_agent(model_path, device=device)
    return make_bot(kind)


def run(args) -> int:
    rng = random.Random(args.seed)
    game = Game(rng=rng)

    agent = None
    if args.opponent == "model" or args.hint or args.watch:
        try:
            from .policy import load_agent

            agent = load_agent(args.model, device=args.device)
        except FileNotFoundError:
            raise SystemExit(f"找不到权重 {args.model}，先训练一个：python -m paodekuai.train --save {args.model}")

    seat = -1 if args.watch else args.seat % 3
    players = []
    for i in range(3):
        if i == seat:
            players.append(HumanPlayer(agent if args.hint else None))
        elif args.opponent == "model":
            players.append(agent)
        else:
            players.append(make_bot(args.opponent))

    label = "模型" if args.opponent == "model" else f"{args.opponent} 机器人"
    if args.watch:
        print(f"观战模式：三家都是{label}。回车推进一步，q 退出。")
    else:
        print(f"你是玩家{seat}，对手是 2 个{label}。持 ♦3 者先出，首手必须带 ♦3。")
        print("输入 h 随时看帮助。")

    last: dict = {}
    while not game.finished:
        obs = game.observe()
        is_human = obs.player == seat

        if is_human:
            show_table(game, seat, last)
            show_hand(obs.hand)
            action = players[obs.player].choose(obs)
            if action is QUIT:
                print("\n退出。")
                return 0
        else:
            player = players[obs.player]
            explained = args.explain and hasattr(player, "scorer") and len(obs.legal) > 1
            if args.watch:
                show_table(game, seat, last)
                print(f"  轮到玩家{obs.player}，手牌 {len(obs.hand)} 张")
            if explained:
                if not args.watch:
                    print()
                show_scores(explain(player, obs), f"玩家{obs.player} 在想：")
            action = player.choose(obs)
            text = "过" if action is None else str(action)
            print(f"  → 玩家{obs.player} {text}")
            if args.watch and input("  [回车继续 / q 退出] ").strip() in ("q", "quit"):
                return 0

        last[obs.player] = "过" if action is None else str(action)
        game.step(action)

    result = game.result()
    print("\n" + "═" * WIDTH)
    if seat < 0:
        print(f"  玩家{result.winner} 获胜。剩余张数: {result.remaining}")
    elif result.winner == seat:
        print(f"  你赢了！对手还剩 {[result.remaining[i] for i in range(3) if i != seat]} 张。")
    else:
        print(f"  玩家{result.winner} 赢了，你还剩 {result.remaining[seat]} 张。")
    print(f"  全局共 {result.turns} 手。")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--opponent", default="model",
                        choices=["random", "greedy", "rule", "model"], help="对手类型（默认 model）")
    parser.add_argument("--model", default="paodekuai/models/agent.pt", help="模型权重路径")
    parser.add_argument("--hint", action="store_true", help="轮到你时先显示模型建议")
    parser.add_argument("--watch", action="store_true", help="观战模式，回车逐步推进")
    parser.add_argument("--no-explain", dest="explain", action="store_false",
                        help="不显示模型出牌时的打分")
    parser.add_argument("--seat", type=int, default=0, help="你坐第几家（0-2）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
