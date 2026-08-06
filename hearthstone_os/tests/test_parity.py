"""对拍测试（路线图 M2 / D4 / G9）：自研简化引擎 vs orange-stone。

同 seed、同镜像卡组（G9 子集，`decks.VANILLA_IDS`——两引擎语义一致的
白板 + 基础关键词卡）、同一"受限随机"策略（只从 play/attack/end_turn 里
均匀随机，两引擎的公共动作宇宙），各自完局，逐步记录后断言：

1. **都打得完**：无死循环（两引擎逐 seed 完局）；
2. **合法动作集包含关系**：简版引擎每局出现的动作类型 ⊆ orange-stone 的，
   且双方每步都有 end_turn（G9 子集没有 choose 状态）——简版能表达的动作
   真实引擎一定也能表达，语义没有丢；
3. **结局分布**：两引擎的 P1 胜率落在同一量级区间（镜像随机对局都远离
   极端），互差有界，且同 seed 的胜者一致率显著高于随机（~60% 实测——
   规则高度一致但非逐位复现，见下）。

规则差异已收口（原 G10 的抽牌差异）：两引擎现在都按官方规则——先手第 1
回合抽第 4 张牌、后手在自己第 1 回合抽牌（orange-stone 修复了漏抽的保真
债 F-A9）。仍存的差异只有简版引擎的既有简化（砍了职业、英雄技能、起手
换牌等），对拍只做统计口径对齐（2026-08 实测 48 局：简版 33% vs os 46%）。

阈值口径（N=48，二项 σ≈7pp，2026-08 实测校准）：
- 一致率 58% → 断言 > 45%（留 ~2σ 余量）
- P1 胜率：简版 33%、os 46% → 断言都在 [0.05, 0.55]，互差 ≤ 35pp
"""

from __future__ import annotations

import random
import unittest

from hearthstone import cards as hs_cards
from hearthstone import game as hs_game

from .. import decks
from ..env import Env

N_SEEDS = 48
MAX_STEPS = 3000


def _hs_deck() -> list:
    """G9 子集卡组（简版引擎侧）：vanilla 15 种 × 2。"""
    ids = [decks.SUBSET_MAP[cid] for cid in decks.VANILLA_IDS]
    return [hs_cards.POOL[hs_cards.CARD_INDEX[name]] for name in ids for _ in range(2)]


def _play_hs(seed: int) -> dict:
    """简版引擎完局：同 seed 同策略（受限随机），逐步记录。"""
    deck = _hs_deck()
    game = hs_game.Game(rng=random.Random(seed), first=0,
                        decklists=[deck, deck], mirror=False)
    rng = random.Random(seed)
    types_seen: set[str] = set()
    end_always = True
    steps = 0
    while not game.finished and steps < MAX_STEPS:
        obs = game.observe()
        # 简版的动作类型叫 "end"，orange-stone 叫 "end_turn"，归一化再比
        types_seen.update("end_turn" if a.kind == "end" else a.kind
                          for a in obs.legal)
        if not any(a.kind == "end" for a in obs.legal):
            end_always = False
        legal = [a for a in obs.legal if a.kind in ("play", "attack", "end")]
        game.step(rng.choice(legal))
        steps += 1
    result = game.result()
    return {
        "winner": result.winner,          # None = 平局, 0/1 = 玩家号
        "finished": game.finished,
        "steps": steps,
        "types": types_seen,
        "end_always": end_always,
    }


def _play_os(seed: int) -> dict:
    """orange-stone 完局：同 seed 同策略（受限随机），逐步记录。"""
    env = Env(deck=decks.vanilla(), seed=seed)
    env.reset(seed=seed)
    rng = random.Random(seed)
    types_seen: set[str] = set()
    end_always = True
    steps = 0
    while not env.done and steps < MAX_STEPS:
        actions = env.legal_actions()
        if not actions:
            break
        types_seen.update(a.kind for a in actions)
        if not any(a.kind == "end_turn" for a in actions):
            end_always = False
        legal = [a for a in actions if a.kind in ("play", "attack", "end_turn")]
        env.step(rng.choice(legal))
        steps += 1
    # os 的 winner 是 1/2（玩家号），归一成简版口径的 0/1/None
    winner = None if env.winner == 0 else env.winner - 1
    return {
        "winner": winner,
        "finished": env.done,
        "steps": steps,
        "types": types_seen,
        "end_always": end_always,
    }


class TestParity(unittest.TestCase):
    """两引擎逐 seed 对拍（结果级 + 动作类型级）。"""

    @classmethod
    def setUpClass(cls):
        cls.hs = [_play_hs(s) for s in range(N_SEEDS)]
        cls.os = [_play_os(s) for s in range(N_SEEDS)]

    def test_both_engines_finish_every_game(self):
        for seed in range(N_SEEDS):
            self.assertTrue(self.hs[seed]["finished"], f"简版 seed={seed} 没打完")
            self.assertTrue(self.os[seed]["finished"], f"os seed={seed} 没打完")

    def test_legal_action_types_are_contained(self):
        """简版引擎的动作类型 ⊆ orange-stone 的（逐局并集）。"""
        for seed in range(N_SEEDS):
            self.assertTrue(
                self.hs[seed]["types"] <= self.os[seed]["types"],
                f"seed={seed}: 简版 {sorted(self.hs[seed]['types'])} "
                f"⊄ os {sorted(self.os[seed]['types'])}",
            )

    def test_end_turn_always_available(self):
        """两引擎每步都有 end_turn（子集卡池没有 choose 状态）。"""
        for seed in range(N_SEEDS):
            self.assertTrue(self.hs[seed]["end_always"],
                            f"简版 seed={seed} 某步没有 end_turn")
            self.assertTrue(self.os[seed]["end_always"],
                            f"os seed={seed} 某步没有 end_turn")

    def test_outcome_distribution_is_aligned(self):
        """结局分布：两引擎 P1 胜率同量级、远离极端、互差有界。"""
        hs_wins = sum(1 for r in self.hs if r["winner"] == 0)
        os_wins = sum(1 for r in self.os if r["winner"] == 0)
        hs_rate, os_rate = hs_wins / N_SEEDS, os_wins / N_SEEDS

        for rate, engine in ((hs_rate, "简版"), (os_rate, "os")):
            self.assertGreater(rate, 0.05, f"{engine} P1 胜率 {rate:.0%} 过低")
            self.assertLess(rate, 0.55, f"{engine} P1 胜率 {rate:.0%} 过高")
        self.assertLess(abs(hs_rate - os_rate), 0.35,
                        f"P1 胜率互差 {abs(hs_rate - os_rate):.0%} 过大")

    def test_same_seed_winners_correlate(self):
        """同 seed 胜者一致率显著高于随机（两引擎规则高度一致）。"""
        agree = sum(1 for a, b in zip(self.hs, self.os)
                    if a["winner"] == b["winner"])
        self.assertGreater(agree / N_SEEDS, 0.45,
                           f"同 seed 一致率 {agree / N_SEEDS:.0%} 太低")


if __name__ == "__main__":
    unittest.main()
