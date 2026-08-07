"""orange-stone vs RosettaStone 大规模对战对比（统计口径）。

两个引擎的 RNG 语义不同（orange-stone 每局独立 GameRng 可精确播种；
RosettaStone 是进程级全局静态 RNG，reset(seed) 只保证同进程顺序执行时
可复现），所以**逐局对齐不可能**——本脚本对比的是终局分布：

- 同职业镜像（MAGE）+ 同构白板套牌（15 种 × 2，两侧卡 ID 等价映射）
- 每侧跑 N 局，RandomBot（uniform random，两侧同构）对阵自己，先后手轮换
- 动作集对齐：orange-stone 侧未实现英雄技能（M5 预留），rosetta 侧
  过滤 HERO_POWER 动作，两侧动作空间都是 play/attack/end_turn
- 记录每局 (winner, steps, turn, P1 终局血, P2 终局血)，汇总分布对比
- 每侧另做自洽验证：同 seed 序列重跑两遍，逐局结果必须一致
  （证明对拍数据是确定性的、可信的）

评估口径（CLAUDE.md）：同职业镜像 + 同构套牌、先后手轮换，P1 胜率
对角线应在 50% ± 2pp。

用法：
    .venv/bin/python tools/compare_rosetta_orange.py [episodes]
"""

from __future__ import annotations

import gc
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

gc.disable()  # rosetta_env 需要（pybind11 类型注册表内存 bug）

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SEED_BASE = 1000
MAX_STEPS = 5000

# 等价白板卡表：15 种 × 2 = 30 张（rosetta VANILLA_IDS 与 orange-stone 经典池
# 内容等价的 15 种：幽灵/银色侍从/石牙野猪/血沼迅猛龙/蓝腮战士/霜狼步兵/
# 精灵龙/狼骑兵/铁鬃灰熊/血色十字军战士/冰风雪人/森金持盾卫士/银月城卫兵/
# 荆棘谷猛虎/石拳食人魔——前 12 种直接对齐，后 3 种用两侧等价的替代卡）。
ROSETTA_DECK = [
    "CS2_231", "EX1_008", "CS2_171", "CS2_172", "CS2_173",
    "CS2_121", "NEW1_023", "CS2_124", "CS2_125", "EX1_020",
    "CS2_182", "CS2_179", "EX1_023", "EX1_028", "CS2_200",
] * 2
ORANGE_DECK = [
    "NEUTRAL_T01", "NEUTRAL_C01", "NEUTRAL_B03", "CLASSIC_001", "CLASSIC_002",
    "NEUTRAL_B05", "CLASSIC_019", "CLASSIC_017", "NEUTRAL_B08", "NEUTRAL_009",
    "NEUTRAL_T08", "CLASSIC_008", "NEUTRAL_014", "NEUTRAL_T14", "NEUTRAL_T09",
] * 2


BOT_NAME = os.environ.get("COMPARE_BOT", "random")


class DeterministicGreedy:
    """顺序无关的确定性 greedy（诊断用，双引擎共用一份代码）。

    与两侧 GreedyBot 同策略（硬币→打脸→攻击→出牌贵→结束），但攻击目标
    和出牌选择不依赖 legal_actions 的枚举顺序（打脸优先；打不到脸攻击
    敌方场攻最高的随从；出牌 cost 最大、同 cost 取 attack 最大的）——
    两侧枚举顺序不同时，原 GreedyBot 的"第一个 attack"会选择不同目标，
    这是 greedy 对拍 P1 胜率 25pp 差异的候选归因；本变体用于检验该假设。

    字段适配：rosetta Action 用 .type/.hand_idx，orange Action 用
    .kind/.card_index —— 统一成 (kind, card_index) 元组访问。
    """

    name = "det_greedy"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    @staticmethod
    def _kind(a) -> str:
        kind = getattr(a, "kind", None)
        if kind:
            return kind
        # rosetta：ActionType 枚举名 PLAY_CARD/ATTACK/HERO_POWER/END_TURN/CHOOSE
        return {
            "PLAY_CARD": "play",
            "ATTACK": "attack",
            "HERO_POWER": "hero_power",
            "END_TURN": "end_turn",
            "CHOOSE": "choose",
        }.get(getattr(a, "type", None).name, "?")

    @staticmethod
    def _hand_idx(a) -> int:
        return getattr(a, "card_index", None) if getattr(a, "kind", None) else a.hand_idx

    @staticmethod
    def _targets_face(a, obs) -> bool:
        """攻击动作的目标是英雄。orange 用 target_id（实体槽），rosetta 用
        target_side + target_pos（位置槽，HERO 在场上槽位之外）。"""
        kind = DeterministicGreedy._kind(a)
        if kind not in ("attack", "hero_power"):
            return False
        if hasattr(a, "target_id"):
            minion_ids = {
                m.entity_id if hasattr(m, "entity_id") else m.id
                for m in obs.opponent.field
            }
            return a.target_id not in minion_ids
        # rosetta：target_pos == -1 指向英雄（HERO 槽）
        return a.target_pos == -1

    @staticmethod
    def _target_minion(a, obs):
        """攻击/技能目标随从（打脸时返回 None）。"""
        if hasattr(a, "target_id"):
            return next(
                (
                    m
                    for m in obs.opponent.field
                    if (m.entity_id if hasattr(m, "entity_id") else m.id) == a.target_id
                ),
                None,
            )
        if a.target_pos < 0 or a.target_pos >= len(obs.opponent.field):
            return None
        return obs.opponent.field[a.target_pos]

    @staticmethod
    def _source_attack(a, obs) -> int:
        """攻击者的攻击力（打脸分支选攻击力最高的攻击者——顺序无关）。"""
        if hasattr(a, "source_pos"):
            field = obs.me.field
            if 0 <= a.source_pos < len(field):
                return field[a.source_pos].attack
            return obs.me.hero_attack if hasattr(obs.me, "hero_attack") else 0
        # orange：source 是 entity_id 槽（英雄在 field 之外）
        src_id = getattr(a, "entity_id", None)
        for m in obs.me.field:
            if getattr(m, "entity_id", None) == src_id:
                return m.attack
        return 0

    def choose(self, obs, actions: list[Action]) -> Action:
        # 幸运币：手里有就立刻用
        for a in actions:
            if self._kind(a) == "play":
                if obs.me.hand[self._hand_idx(a)].card_id == "GAME_005":
                    return a
        # 攻击：能打脸就打脸（攻击者按攻击力从高到低——不依赖枚举顺序）
        face_attacks = [a for a in actions if self._kind(a) == "attack" and self._targets_face(a, obs)]
        if face_attacks:
            return max(face_attacks, key=lambda a: DeterministicGreedy._source_attack(a, obs))
        # 打不到脸：攻击敌方场攻最高的随从（确定性，不依赖枚举顺序）
        attacks = [a for a in actions if self._kind(a) == "attack"]
        if attacks:
            return max(
                attacks,
                key=lambda a: (
                    (m.attack if (m := DeterministicGreedy._target_minion(a, obs)) else -1),
                    (m.health if (m := DeterministicGreedy._target_minion(a, obs)) else -1),
                    DeterministicGreedy._source_attack(a, obs),
                ),
            )
        # 出牌：cost 最大，同 cost 取 attack 最大（白板套牌下几乎唯一）
        plays = [a for a in actions if self._kind(a) == "play"]
        if plays:
            return max(
                plays,
                key=lambda a: (
                    obs.me.hand[self._hand_idx(a)].cost,
                    obs.me.hand[self._hand_idx(a)].attack,
                ),
            )
        return next(a for a in actions if self._kind(a) == "end_turn")


def run_rosetta(episodes: int) -> list[tuple]:
    """RosettaStone 侧：每局 bot × 2（过滤 HERO_POWER 对齐动作集）。"""
    from rosetta.env import Env, ActionType
    from rosetta.bots import GreedyBot as RGreedy, RandomBot as RRandom

    bot_cls = {"random": RRandom, "greedy": RGreedy, "det_greedy": DeterministicGreedy}[BOT_NAME]

    results = []
    for episode in range(episodes):
        env = Env(
            player1_class="MAGE",
            player2_class="MAGE",
            player1_deck=ROSETTA_DECK,
            player2_deck=ROSETTA_DECK,
        )
        env.reset(seed=SEED_BASE + episode)
        bot1_seed = SEED_BASE + episode * 2
        bot2_seed = bot1_seed + 1
        bot1, bot2 = bot_cls(bot1_seed), bot_cls(bot2_seed)
        steps = 0
        while not env.done and steps < MAX_STEPS:
            actions = [a for a in env.legal_actions() if a.type != ActionType.HERO_POWER]
            if not actions:
                break
            obs = env.observe()
            bot = bot1 if env.current_player == 1 else bot2
            env.step(bot.choose(obs, actions))
            steps += 1
        # 终局血线：胜负未决时的当前双方血量（罗盘视角取最后观察）
        obs = env.observe()
        p1_hp = obs.me.hero_health if env.current_player == 1 else obs.opponent.hero_health
        p2_hp = obs.opponent.hero_health if env.current_player == 1 else obs.me.hero_health
        results.append((env.winner, steps, env.turn, p1_hp, p2_hp))
    return results


def run_orange(episodes: int) -> list[tuple]:
    """orange-stone 侧：每局 bot × 2（动作集天然无 hero_power）。"""
    from hearthstone_os.env import Env
    from hearthstone_os.bots import GreedyBot as OGreedy, RandomBot as ORandom

    bot_cls = {"random": ORandom, "greedy": OGreedy, "det_greedy": DeterministicGreedy}[BOT_NAME]

    results = []
    for episode in range(episodes):
        env = Env(deck=ORANGE_DECK, seed=SEED_BASE + episode, bot="none")
        env.reset(seed=SEED_BASE + episode)
        bot1_seed = SEED_BASE + episode * 2
        bot2_seed = bot1_seed + 1
        bot1, bot2 = bot_cls(bot1_seed), bot_cls(bot2_seed)
        steps = 0
        while not env.done and steps < MAX_STEPS:
            actions = env.legal_actions()
            if not actions:
                break
            obs = env.observe()
            bot = bot1 if env.current_player == 1 else bot2
            env.step(bot.choose(obs, actions))
            steps += 1
        p1_hp, p2_hp = env.hero_healths()
        results.append((env.winner, steps, env.turn, p1_hp, p2_hp))
    return results


def summarize(results: list[tuple], name: str) -> dict:
    n = len(results)
    p1_wins = sum(1 for r in results if r[0] == 1)
    p2_wins = sum(1 for r in results if r[0] == 2)
    draws = sum(1 for r in results if r[0] == 0)
    steps = [r[1] for r in results]
    turns = [r[2] for r in results]
    p1_hp = [r[3] for r in results]
    p2_hp = [r[4] for r in results]
    # 终局血线只看有胜者的对局（平局血线无意义）
    fin = [r for r in results if r[0] != 0]
    # 胜者血线（P1 赢看 P1 血，P2 赢看 P2 血）+ 败者血线（对称）
    win_hp = [r[3] if r[0] == 1 else r[4] for r in fin]
    lose_hp = [r[4] if r[0] == 1 else r[3] for r in fin]
    win_hp.sort()
    lose_hp.sort()

    def quantiles(v: list[int]) -> tuple:
        return tuple(v[int(len(v) * q)] for q in (0.25, 0.5, 0.75))

    return {
        "name": name,
        "n": n,
        "p1_win": p1_wins / n,
        "p2_win": p2_wins / n,
        "draw": draws / n,
        "avg_steps": sum(steps) / n,
        "avg_turn": sum(turns) / n,
        "avg_p1_hp": sum(r[3] for r in fin) / len(fin) if fin else 0.0,
        "avg_p2_hp": sum(r[4] for r in fin) / len(fin) if fin else 0.0,
        "win_hp_q": quantiles(win_hp),
        "lose_hp_q": quantiles(lose_hp),
    }


def self_check(run_fn, episodes: int = 20, name: str = "") -> bool:
    """引擎自洽：同 seed 序列重跑两遍，逐局结果必须一致。"""
    r1 = run_fn(episodes)
    r2 = run_fn(episodes)
    ok = r1 == r2
    print(f"自洽检查 [{name}]: {episodes} 局同 seed 重跑 {'逐局一致 ✓' if ok else '不一致 ✗'}")
    if not ok:
        for i, (a, b) in enumerate(zip(r1, r2)):
            if a != b:
                print(f"  第 {i} 局: {a} vs {b}")
    return ok


def main() -> None:
    print(f"=== orange-stone vs RosettaStone 大规模对战对比（每侧 {EPISODES} 局，MAGE 镜像 + 等价白板套牌）===")
    print(f"动作集对齐：rosetta 侧过滤 HERO_POWER（orange-stone 未实现英雄技能，M5 预留）")

    if not self_check(run_rosetta, 20, "rosetta"):
        print("rosetta 自洽失败——对拍数据不可信，中止")
        return
    if not self_check(run_orange, 20, "orange-stone"):
        print("orange-stone 自洽失败——对拍数据不可信，中止")
        return

    r_rosetta = run_rosetta(EPISODES)
    r_orange = run_orange(EPISODES)

    s_rosetta = summarize(r_rosetta, "RosettaStone")
    s_orange = summarize(r_orange, "orange-stone")

    print()
    print(f"{'指标':<14}{'RosettaStone':>14}{'orange-stone':>14}{'差值':>10}")
    print("-" * 54)
    keys = [
        ("p1_win", "P1 胜率"),
        ("p2_win", "P2 胜率"),
        ("draw", "平局率"),
        ("avg_steps", "平均步数"),
        ("avg_turn", "平均回合"),
        ("avg_p1_hp", "胜者侧血线"),
        ("avg_p2_hp", "败者侧血线"),
    ]
    for k, label in keys:
        a, b = s_rosetta[k], s_orange[k]
        print(f"{label:<14}{a:>14.4f}{b:>14.4f}{b - a:>+10.4f}")

    print()
    p1_diff = abs(s_rosetta["p1_win"] - s_orange["p1_win"])
    steps_diff = abs(s_rosetta["avg_steps"] - s_orange["avg_steps"])
    print(f"P1 胜率差 {p1_diff * 100:.2f}pp（口径 ±2pp）→ "
          f"{'在口径内 ✓' if p1_diff <= 0.02 else '超出口径 ✗'}")
    print(f"平均步数差 {steps_diff:.1f} 步（相对 {s_orange['avg_steps']:.1f} = "
          f"{steps_diff / s_orange['avg_steps'] * 100:.1f}%）")

    # 平局率的绝对比较（白板镜像套牌下平局应极少或为 0）
    print(f"平局率：rosetta {s_rosetta['draw'] * 100:.2f}% vs orange {s_orange['draw'] * 100:.2f}%")

    # 胜者/败者血线分位数（胜者 = 赢家的剩余血，败者 = 输家的剩余血）
    print()
    print("胜者剩余血分位数 [25/50/75]: "
          f"rosetta {s_rosetta['win_hp_q']} vs orange {s_orange['win_hp_q']}")
    print("败者剩余血分位数 [25/50/75]: "
          f"rosetta {s_rosetta['lose_hp_q']} vs orange {s_orange['lose_hp_q']}")


if __name__ == "__main__":
    main()
