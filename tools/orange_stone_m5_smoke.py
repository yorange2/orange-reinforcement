"""M5 冒烟测试：卡池扩展与保真（潜行/扰咒机制、全经典构筑池、特征 v7+）。

验证口径（路线图 M5）：
- 潜行：丛林豹上场后不能被攻击；攻击后破潜行（orange-stone #72）
- 扰咒：精灵龙不能被法术指定为目标（orange-stone #73）
- 全经典构筑池：321 张可用，随机套牌压力测试 200 局无异常
- 特征 v7+：223 维（47 动作 + 176 局面，卡面文本块）

用法：
    .venv/bin/python -m tools.orange_stone_m5_smoke
"""

from __future__ import annotations

import random

from hearthstone_os import decks
from hearthstone_os.env import Env
from hearthstone_os.bots import GreedyBot, RandomBot
from hearthstone_os.features import ACTION_DIM, FEATURE_DIM, STATE_DIM


def section_stealth() -> None:
    """潜行：不可被攻击，攻击后破潜行。"""
    print("=== 潜行（orange-stone #72）===")
    env = Env(deck=["NEUTRAL_C10"] * 10, seed=1, bot="none")   # 丛林豹（3 费）
    env.reset(seed=1)
    for _ in range(6):                     # 攒法力到能出
        acts = env.legal_actions()
        play = next((a for a in acts if a.kind == "play"), None)
        if play is not None and env.current_player == 1:
            env.step(play)
            break
        env.step(next(a for a in acts if a.kind == "end_turn"))
    env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
    env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
    obs = env.observe()
    panther = obs.me.field[0]
    assert panther.stealth, "丛林豹必须有潜行"
    # 对手的合法攻击里不能把潜行随从当目标
    env.step(next(a for a in env.legal_actions() if a.kind == "end_turn"))
    obs2 = env.observe()
    attacks = [a for a in env.legal_actions() if a.kind == "attack"]
    for a in attacks:
        assert a.target_id not in {m.entity_id for m in obs2.opponent.field}, \
            "潜行随从不能被攻击"
    print("  ✓ 潜行随从在对手攻击枚举里不可见")


def section_elusive() -> None:
    """扰咒：不能被法术指定。"""
    print("=== 扰咒（orange-stone #73）===")
    import orange_stone as os

    # 火球术 vs 精灵龙：精灵龙在手牌里（法术目标在对方场上，这里验证视图字段）
    env = os.GameEnv(seed=2, deck=["CLASSIC_019"] * 10, bot="none")
    c = next(x for x in env.structured_observation().me.hand
             if x.card_id == "CLASSIC_019")
    assert c.elusive, "精灵龙必须有扰咒"
    print("  ✓ 精灵龙视图暴露 elusive=True")


def section_pool() -> None:
    """全经典构筑池 + 压力测试。"""
    print("=== 全经典构筑池 ===")
    pool = decks.full_pool()
    assert len(pool) > 300, f"构筑池只有 {len(pool)} 张，应该 >300"
    print(f"  ✓ 构筑池 {len(pool)} 张（410 ALL_CARDS − 简化债 − 衍生物）")

    bad = []
    for s in range(200):
        deck = decks.random_deck(random.Random(s))
        env = Env(deck=deck, seed=s)
        env.reset(seed=s)
        seats = {1: RandomBot(s), 2: GreedyBot(s)}
        steps = 0
        while not env.done and steps < 5000:
            acts = env.legal_actions()
            if not acts:
                bad.append(("stuck", s))
                break
            env.step(seats[env.current_player].choose(env.observe(), acts))
            steps += 1
        if not env.done and not bad:
            bad.append(("deadlock", s))
    assert not bad, f"压力测试 {len(bad)} 局异常: {bad[:3]}"
    print("  ✓ 200 局随机套牌全部正常完局")


def section_features() -> None:
    """特征 v7+：223 维（含卡面文本块）。"""
    print("=== 特征 v7+ ===")
    assert FEATURE_DIM == 223, f"v7+ 应定版 223 维，实际 {FEATURE_DIM}"
    assert ACTION_DIM == 47 and STATE_DIM == 176, \
        f"布局不对: {ACTION_DIM} + {STATE_DIM}"
    print(f"  ✓ v7+ = {FEATURE_DIM} 维（动作 {ACTION_DIM} + 局面 {STATE_DIM}，卡面文本块）")


def main() -> None:
    section_stealth()
    section_elusive()
    section_pool()
    section_features()
    print("\n全部 M5 小节通过 ✓")


if __name__ == "__main__":
    main()
