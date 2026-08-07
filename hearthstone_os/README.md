# hearthstone_os — orange-stone 炉石 RL 模块

基于 **orange-stone**（Rust 炉石模拟器）的 Python RL 模块（路线图 M2/M3）。
API 形状复刻 `rosetta/`，是 `hearthstone/`（自研纯 Python 引擎）的迁移目标。
M2 = 环境门面 + bots + arena + 对拍；M3 = 特征 v7 + PPO 训练 + 整回合搜索。

## 为什么是 orange-stone（对比 rosetta 的三个硬伤）

| rosetta/RosettaStone | orange-stone |
| --- | --- |
| `Game` 不可 clone（拷贝构造 `= delete`），整回合搜索做不了 | CoW GameState 克隆廉价，`Env.clone()` 直接可用（M3 搜索的恢复基础） |
| 全局静态 RNG，一个进程只能跑一局，并行只能多进程 | 每局独立 `GameRng`，同 seed 逐位可复现、实例互不干扰（可多线程） |
| AGPL-3.0（许可证传染） | MIT |

## 结构

```
hearthstone_os/
├── env.py        # Env 门面：reset / legal_actions / step / observe / clone / last_reward
├── bots.py       # random / greedy / rule 三个手写 bot
├── arena.py      # duel/matrix 胜率矩阵 + play_game/evaluate/final_reward（训练评测口径）
├── decks.py      # G9 子集卡池（28 张两引擎语义一致的卡）+ vanilla 镜像套牌
├── features.py   # 特征 v7（199 维：31 动作 + 168 局面，结构化视图上移植 v5/v6 思路）
├── policy.py     # PPO 网络与智能体（hearthstone/policy.py 平移，无先知价值头）
├── search.py     # 整回合 beam 搜索（Env.clone() 恢复，rosetta 做不到的收益）
├── train.py      # PPO+GAE(λ=0.5) 训练入口
├── bench.py      # 统一口径战绩表（模型 / 模型+搜索 / 规则对手）
├── play.py       # 人机观战 / bench / matrix 命令行
├── models/       # 训练好的权重
└── tests/        # 环境三件套 + 对拍 + 特征/搜索/训练管线
```

## 快速上手

```bash
.venv/bin/python -m hearthstone_os.play                      # 观战 greedy vs random
.venv/bin/python -m hearthstone_os.play --bots rule greedy   # 换对手
.venv/bin/python -m hearthstone_os.play --bench 400          # 跑胜率
.venv/bin/python -m hearthstone_os.play --matrix 200         # 胜率矩阵（对角线校准）
.venv/bin/python -m hearthstone_os.train --episodes 30000    # 训练（M3）
.venv/bin/python -m hearthstone_os.bench --model hearthstone_os/models/agent.pt  # 战绩表（含 +搜索）
.venv/bin/python -m unittest discover -s hearthstone_os/tests -t .   # 全部测试
.venv/bin/python -m tools.orange_stone_m2_smoke              # M2 冒烟
.venv/bin/python -m tools.orange_stone_m3_smoke              # M3 冒烟
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

## M3 训练（2026-08-06 实测）

- 训练速度 ~90 局/s（CPU，双 GameEnv 锁步 + 结构化视图的开销；M4 优化）
- 特征 v7 = 199 维（31 动作 + 168 局面），PPO + GAE(λ=0.5)、68k 参数，
  与 `hearthstone/` 的 v6 同一套更新代码
- 整回合搜索用 `Env.clone()` 恢复（rosetta 做不到的收益），beam 默认 8

**战绩表**（每格 600 局，先后手轮换，`bench.py`，seed 999）：

| 选手 | vs random | vs greedy | vs rule |
| --- | --- | --- | --- |
| agent.pt + 整回合搜索 | 99.8% | 75.3% | 69.7% |
| agent.pt (71k, PPO+GAE 3 万局) | 100.0% | 78.5% | 67.7% |
| rule | 99.7% | 64.7% | 52.7% |
| greedy | 99.5% | 53.5% | 37.2% |
| random | 52.5% | 0.2% | 0.0% |

（v7+ 特征 223 维重训后口径，每格 600 局；v7 199 维时为 70.5% / +搜索
72.5%——都在 ±2pp 内，M3 回归基准保持。）+搜索的增量（+2pp vs rule）
比 v6 小——v6 的 77.7% 里有座位分裂假象的成分（见下）。

**过程中发现并修掉的问题**：

1. **引擎 bug（orange-stone #70）**：`build_game_state` 把先手法力重置为
   0/0，覆盖了 `GameState::new()` 给的第一回合水晶——先手整局少一个水晶，
   P1 侧 RL 训练系统性坍缩（实测 P1 胜率 ~3%、P2 ~96%）。**v6 的 69% 战绩
   里也有同样的座位分裂假象**（简版引擎对照实测 P1 82.7% / P2 44.7%，
   均值 ≈ 64%）。修复后 P1-only 训练恢复正常（5k 局 43% 且持续上升）。
2. **搜索停滞（本模块）**：价值头对"快赢局面"超估（实测 +1.065 > 终局
   +1.0），搜索把"结束回合"排在真斩杀前面，对着 1 血空场对手无限拖
   回合。修复：价值预测裁剪到 [−1, +1] + 终局斩杀无条件优先。

## M5 卡池扩展与保真（2026-08-06 实测）

- **特征 v7+ = 223 维**（47 动作 + 176 局面）：引擎补上卡面文本视图字段后，
  A_TEXT/S_TEXT 块回归（标签 + 战吼/法术量级 + 亡语量级 + 光环量级）
- **潜行/扰咒入池**（orange-stone #72/#73）：丛林豹/荆棘谷猛虎/拉文霍德
  刺客真潜行、精灵龙真扰咒（法术不能以它为目标的枚举与结算两侧都堵上）
- **全经典构筑池 321 张**（410 ALL_CARDS − 68 简化债 − 衍生物/硬币），
  `random_deck()` 随机组牌；200 局压力测试全部正常完局
- **全卡池训练**（`train --pool full`，30k × 3 seed，随机组牌）：

| 选手（全卡池口径，321 卡池） | vs random | vs greedy | vs rule |
| --- | --- | --- | --- |
| agent_full（PPO+GAE 3 万局 × 3 seed） | 97.5-99.0% | 70.0-74.5% | **62.5-66.0%** |

### 2026-08-07 重训（orange-stone PR #108 之后，392 卡池）

引擎侧 PR #108 修了 10 张卡的实现并**接通了此前完全没接的法术伤害管线**
（`total_spell_damage` 零调用者——狗头人地卜师、达拉然法师、食人魔法师、大法师、
血法师萨尔诺斯、青玉龙、玛里苟斯、远古法师此前实际都是白板）。这改变了含法术
伤害随从卡组的实际强度，因此上表数字作废，同口径重跑：

| 选手（全卡池口径，392 卡池） | vs random | vs greedy | vs rule |
| --- | --- | --- | --- |
| agent_full（PPO+GAE 3 万局 × 3 seed） | 100.0% | 73.2-78.8% | **62.3-68.8%** |

- 口径与旧表完全一致（`--pool full --episodes 30000 --parallel 8`，
  `--final-eval-games 400`，评测 seed 999，先后手轮换）
- 对角线校准正常：`rule vs rule` = 50.0%，在 50% ± 2pp 内
- **不要把这解读为「变强了」**：vs rule 的 seed 间跨度 6.5pp（62.3 / 65.8 / 68.8），
  比旧表的 3.5pp 更宽，而两次的区间大面积重叠。卡池（321 → 392）和引擎语义同时
  变了，这里能说的只有「同量级」
- 3 万局 × 3 seed 在 M3 Pro 上约 6 分钟/seed（`--parallel 8`）
- 旧权重保留为 `models/agent_full_s{0,1,2}.pre-pr108.pt`（模型目录被 gitignore）

- **外部 SabberStone 对照**（orange-stone F5 补完）：dotnet 驱动镜像
  attack-trade 场景，两个模拟器结果一致（orange-stone #75）
- 简化债卡（68 张，如 Tauren Warrior 的 enrage）按路线图风险对策排除在
  训练卡池外，等引擎侧 F4/F5 审计清完再进

## M4 批量与性能（2026-08-06 实测）

**吞吐基准表**（M3 Pro，vanilla 镜像卡组）：

| 配置 | 局/s | 对照 |
| --- | --- | --- |
| M0 基线（GameEnv Python 驱动，随机 vs Greedy，随机卡组） | ~970 | — |
| M4 引擎批量（`battle_batch`，rayon 全 Rust，Greedy vs Greedy） | ~4,200 | ≈4.4× M0 基线 / **9.2× rosetta（460）** |
| M4 Python 批量（`BatchedEnv`，4 线程，随机 vs Greedy） | ~1,850 | ≈1.9× M0 基线 |
| M4 批量训练（`train --parallel 8`，vs rule） | ~150-185 | ≈1.8× 单局训练（~90） |

实现：
- **Rust 侧（orange-stone #71）**：热方法（step/legal_actions/structured_*）在
  `allow_threads` 里释放 GIL（引擎纯 Rust + 逐局 RNG，线程安全）；新增
  `BatchEnv`（一次调用驱动 N 局，结构化观测直接给**当前行动方视角**——
  批量训练不需要双实例锁步）；新增 `battle_batch`（rayon 批量 bot-vs-bot）
- **Python 侧**：`batched.py` 的 `BatchedEnv` 门面 + 吞吐跑分；`train.py
  --parallel N` 批量训练（决策 padded 矩阵一次前向，完成的局单独
  `reset_one` 重开）
- **确定性**：批量与单局逐 seed 结果一致（`test_batched.py` 断言；
  确定性策略下 BatchedEnv 与单局 Env 的 12 局 winner 完全一致）

## 对拍测试（tests/test_parity.py）

同 seed、同镜像卡组、同一受限随机策略，简版引擎（`hearthstone/`）与
orange-stone 各自完局，断言（2026-08 实测 40~80 局）：

- 两引擎全部打完（无死循环）；
- 简版动作类型 ⊆ orange-stone 动作类型，且每步都有 end_turn；
- 结局分布同量级：同 seed 胜者一致率 ~58%，P1 胜率简版 ~33% / os ~46%
  （先手第 1 回合抽牌已对齐官方规则：orange-stone 修掉了漏抽的保真债
  F-A9，P1 胜率从 ~23% 回升；余下互差来自简版引擎的其他简化，见下）。

**规则差异已收口**（原 G10"迁移不是逐位复现"里的抽牌差异已消除）：
两引擎现在都在先手第 1 回合按官方规则抽第 4 张牌、后手在自己第 1 回合抽
牌。仍存的差异只有简版引擎的既有简化（砍了职业、英雄技能、起手换牌等）。
对拍只做统计口径对齐，不做逐动作比对。
