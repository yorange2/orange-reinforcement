"""M2 冒烟测试：hearthstone_os 新模块（env/bots/arena/对拍）。

验证口径（路线图 M2 验收）：
- env 门面：完局、确定性、clone、双视角（P2 也能看到自己的手牌）
- bots 强弱序：random < greedy < rule
- arena 胜率矩阵：对角线 50% ± 2pp（同镜像卡组没有优势）
- 对拍：简版引擎 vs orange-stone 统计口径对齐（动作类型包含 + 结局分布）

用法：
    .venv/bin/python -m tools.orange_stone_m2_smoke
"""

from __future__ import annotations

import random
import time

from hearthstone import cards as hs_cards
from hearthstone import game as hs_game

from hearthstone_os import arena, decks
from hearthstone_os.bots import BOTS, GreedyBot, RandomBot, RuleBot
from hearthstone_os.env import Env

MAX_STEPS = 3000


def section_env() -> None:
    """env 门面：完局、确定性、clone、双视角。"""
    print("=== env 门面 ===")
    env = Env(deck=decks.vanilla(), seed=42)
    env.reset(seed=42)

    # 双视角：P1 开局能看到自己的手牌，P2 回合能看到 P2 的手牌
    obs = env.observe()
    assert obs.me.hand_count == 3 and obs.opponent.hand_count == 5
    env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
    obs2 = env.observe()
    # P2 首回合：4 起手 + 硬币 + 回合首抽 = 6（与简版引擎口径一致）
    assert obs2.me.hand_count == 6, "P2 行动时必须能看到 P2 自己的手牌"
    print(f"  ✓ 双视角：P1 起手 {obs.me.hand_count} 张，P2 首回合 {obs2.me.hand_count} 张（4+硬币+首抽）")

    # 完局 + 确定性
    def play(seed: int):
        e = Env(deck=decks.vanilla(), seed=seed)
        e.reset(seed=seed)
        rng = random.Random(seed)
        acts = []
        for _ in range(MAX_STEPS):
            actions = e.legal_actions()
            if not actions:
                break
            a = rng.choice(actions)
            acts.append(a.index)
            e.step(a)
            if e.done:
                return acts, e.winner
        return acts, None

    a1, w1 = play(7)
    a2, w2 = play(7)
    assert a1 == a2 and w1 == w2, "同 seed 两次完局必须逐位一致"
    assert w1 in (1, 2)
    print(f"  ✓ 确定性：seed 7 两次完局逐位一致（{len(a1)} 步, winner={w1}）")

    # clone 分支不影响原局
    env2 = Env(deck=decks.vanilla(), seed=3)
    before = env2.observe().turn
    branch = env2.clone()
    branch.step(next(a for a in branch.legal_actions() if a.kind == "end_turn"))
    assert env2.observe().turn == before, "推进 clone 不得影响原局"
    print("  ✓ clone 分支不污染原局")


def section_bots() -> None:
    """bots 强弱序：random < greedy < rule。"""
    print("=== bots 强弱序 ===")
    gr = arena.duel(GreedyBot, RandomBot, episodes=200, seed=0)
    rg = arena.duel(RuleBot, GreedyBot, episodes=600, seed=1)
    rr = arena.duel(RuleBot, RandomBot, episodes=200, seed=0)
    print(f"  greedy vs random : {gr['win_rate']:.1%}")
    print(f"  rule   vs greedy : {rg['win_rate']:.1%}")
    print(f"  rule   vs random : {rr['win_rate']:.1%}")
    assert gr["win_rate"] > 0.9
    assert rg["win_rate"] > 0.50
    assert rr["win_rate"] > 0.95


def section_matrix() -> None:
    """胜率矩阵：对角线 50% ± 2pp（M2 验收口径）。"""
    print("=== 胜率矩阵（每对 1000 局，对角线应 ≈ 50%）===")
    names = sorted(BOTS)
    start = time.time()
    result = arena.matrix(names, episodes=1000, seed=0)
    header = "        " + "".join(f"{n:>8}" for n in names)
    print(header)
    for row in names:
        cells = "".join(f"{result[row][col]:>8.1%}" for col in names)
        print(f"{row:>8}{cells}")
    for name in names:
        diag = result[name][name]
        assert abs(diag - 0.5) <= 0.02, f"对角线 {name} 自打 {diag:.1%} 偏离 50%"
    print(f"  ✓ 对角线全部落在 50%±2pp（耗时 {time.time() - start:.0f}s）")


def section_parity() -> None:
    """对拍：简版引擎 vs orange-stone 统计口径。"""
    print("=== 对拍（G9 子集卡池，40 seed）===")
    hs_ids = [decks.SUBSET_MAP[cid] for cid in decks.VANILLA_IDS]
    hs_deck = [hs_cards.POOL[hs_cards.CARD_INDEX[n]] for n in hs_ids for _ in range(2)]

    def play_hs(seed: int):
        g = hs_game.Game(rng=random.Random(seed), first=0,
                         decklists=[hs_deck, hs_deck], mirror=False)
        rng = random.Random(seed)
        while not g.finished:
            obs = g.observe()
            legal = [a for a in obs.legal if a.kind in ("play", "attack", "end")]
            g.step(rng.choice(legal))
        return g.result().winner

    def play_os(seed: int):
        env = Env(deck=decks.vanilla(), seed=seed)
        rng = random.Random(seed)
        while not env.done:
            legal = [a for a in env.legal_actions()
                     if a.kind in ("play", "attack", "end_turn")]
            env.step(rng.choice(legal))
        return None if env.winner == 0 else env.winner - 1

    n = 40
    hs_w = [play_hs(s) for s in range(n)]
    os_w = [play_os(s) for s in range(n)]
    agree = sum(1 for a, b in zip(hs_w, os_w) if a == b)
    hs_rate = sum(1 for w in hs_w if w == 0) / n
    os_rate = sum(1 for w in os_w if w == 0) / n
    print(f"  同 seed 胜者一致率：{agree / n:.0%}")
    print(f"  P1 胜率：简版 {hs_rate:.0%}  os {os_rate:.0%}")
    assert agree / n > 0.45, "同 seed 一致率太低"
    assert 0.05 < hs_rate < 0.55 and 0.05 < os_rate < 0.55
    assert abs(hs_rate - os_rate) < 0.35
    print("  ✓ 结局分布同量级、无死循环")


def main() -> None:
    section_env()
    section_bots()
    section_matrix()
    section_parity()
    print("\n全部 M2 小节通过 ✓")


if __name__ == "__main__":
    main()
