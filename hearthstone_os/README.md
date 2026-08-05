# hearthstone_os — orange-stone 炉石 RL 模块

基于 **orange-stone**（Rust 炉石模拟器）的 Python RL 模块（路线图 M2）。
API 形状复刻 `rosetta/`，是 `hearthstone/`（自研纯 Python 引擎）的迁移目标。

## 为什么是 orange-stone（对比 rosetta 的三个硬伤）

| rosetta/RosettaStone | orange-stone |
| --- | --- |
| `Game` 不可 clone（拷贝构造 `= delete`），整回合搜索做不了 | CoW GameState 克隆廉价，`Env.clone()` 直接可用（M3 搜索的恢复基础） |
| 全局静态 RNG，一个进程只能跑一局，并行只能多进程 | 每局独立 `GameRng`，同 seed 逐位可复现、实例互不干扰（可多线程） |
| AGPL-3.0（许可证传染） | MIT |

## 结构

```
hearthstone_os/
├── env.py        # Env 门面：reset / legal_actions / step / observe / clone
├── bots.py       # random / greedy / rule 三个手写 bot
├── arena.py      # duel 多局评测 + matrix 胜率矩阵（对角线 50%±2pp）
├── decks.py      # G9 子集卡池（28 张两引擎语义一致的卡）+ vanilla 镜像套牌
├── play.py       # 人机观战 / bench / matrix 命令行
└── tests/        # 三件套（完局/确定性/枚举一致性）+ bots + 对拍测试
```

## 快速上手

```bash
.venv/bin/python -m hearthstone_os.play                      # 观战 greedy vs random
.venv/bin/python -m hearthstone_os.play --bots rule greedy   # 换对手
.venv/bin/python -m hearthstone_os.play --bench 400          # 跑胜率
.venv/bin/python -m hearthstone_os.play --matrix 200         # 胜率矩阵（对角线校准）
.venv/bin/python -m unittest discover -s hearthstone_os/tests -t .   # 全部测试
.venv/bin/python -m tools.orange_stone_m2_smoke              # M2 冒烟
```

## Env 用法

```python
from hearthstone_os import Env, decks
from hearthstone_os.bots import RuleBot

env = Env(deck=decks.vanilla(), seed=42)   # 双 bot 模式（bot="none"，双方外部可控）
bot = RuleBot()
while not env.done:
    actions = env.legal_actions()          # 当前行动方的合法动作（Action 对象）
    env.step(bot.choose(env.observe(), actions))
print(env.winner)                          # 0=平局, 1=P1, 2=P2
```

要点：
- `observe()` 返回**当前行动方**视角（`me` 带手牌、`opponent` 手牌隐藏）。
  底层是双 GameEnv（perspective 0/1）锁步驱动——绑定层 perspective 固定，
  双 bot 对局需要任意一方都能看到自己的手牌。
- `clone()` 分支不影响原局，M3 的整回合搜索直接在 Python 侧做。
- 起手默认 3 张 + 后手硬币（官方/简化炉石口径）。

## 评测口径（M2 验收，实测 2026-08-06）

G9 子集镜像卡组（`decks.vanilla()`，15 种 × 2），胜率矩阵每对 1000 局：

| 行\列 | greedy | random | rule |
| --- | --- | --- | --- |
| greedy | 50.5% | 98.5% | 46.7% |
| random | 0.9% | 49.6% | 0.7% |
| rule | 52.9% | 99.3% | 50.5% |

对角线全部在 50%±2pp；强弱序 random < greedy < rule。

> rule vs greedy 只有 ~53%（rosetta 那边是 55%+）：G9 子集没有潜行/扰咒/
> 英雄技能这些 rule 能白嫖的点，优势被压缩。`test_rule_beats_greedy` 按
> 实测只断言 >50%。

## 对拍测试（tests/test_parity.py）

同 seed、同镜像卡组、同一受限随机策略，简版引擎（`hearthstone/`）与
orange-stone 各自完局，断言（2026-08 实测 40~80 局）：

- 两引擎全部打完（无死循环）；
- 简版动作类型 ⊆ orange-stone 动作类型，且每步都有 end_turn；
- 结局分布同量级：同 seed 胜者一致率 ~61%，P1 胜率简版 ~34% / os ~23%
  （官方先手第 1 回合不抽牌，orange-stone 的后手优势更大，符合预期）。

**已知规则差异**（路线图 G10，"迁移不是逐位复现"）：简版引擎回合开始就
抽牌（先手第 1 回合 4 张起手），orange-stone 按官方规则先手第 1 回合不抽
（3 张起手）。对拍只做统计口径对齐，不做逐动作比对。
