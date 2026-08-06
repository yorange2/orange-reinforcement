# 路线图：hearthstone 基于 orange-stone 的对战模拟（orange-reinforcement 侧）

> 目标：把 `hearthstone/` 的炉石 AI 从「自研纯 Python 引擎 + 实验性 C++ RosettaStone 绑定」迁移到 **orange-stone（Rust 模拟器）** 上——让 `hearthstone/` 的训练、搜索、评测全部跑在真实规则引擎上，并吃下批量模拟与确定性回放的红利。
>
> 本文档记录 **orange-reinforcement 侧**的工作（环境打通、`hearthstone_os/` 模块、训练接入、批量、卡池、发布、决策点）；orange-stone 侧的接口/批量/保真工作见 `orange-stone/docs/finished/rl-interface-roadmap.md`。
>
> 现状核对时间：2026-08-05（两个仓库的代码均已实地核实，下文差距清单按当前代码口径写）。

---

## 1. 为什么是 orange-stone（对比 rosetta 方案的三个痛点）

`rosetta/`（RosettaStone C++ 绑定）是上一轮"接真实引擎"的尝试，README 里明确写着：**最终想把 `hearthstone/` 的训练链接到真实规则上**。它暴露了三个硬伤，而 orange-stone 恰好全部解决：

| rosetta/RosettaStone 的痛点 | orange-stone 的对应能力 |
| --- | --- |
| **没 clone()**：`Game` 拷贝/移动构造都是 `= delete`，整回合 beam 搜索（+14.4pp 的最大收益项）无对应物 | CoW GameState 是设计核心，克隆廉价——搜索/回滚天然可恢复（绑定层已暴露 `clone()`，见 M2） |
| **全局静态 RNG**：一个进程同一时刻只能跑一局，并行采样被迫多进程 | 每局独立 `GameRng`，`sim/batch.rs` 已用 rayon 线程级并行且逐局可复现 |
| **AGPL-3.0**：公开仓库 + GitHub Pages 存在许可证传染风险 | MIT |

此外还有工程红利：单核 ~7,000 局/s（README 口径，对比 rosetta ~460 局/s、自研 ~125 局/s）、PyO3 绑定已有雏形、确定性回放是设计目标、全经典卡池（`ALL_CARDS` 413 唯一条目，2026-08-06 审计修复轮后口径）。

## 2. 现状盘点

### 2.1 orange-reinforcement 侧（Python）

| 模块 | 说明 |
| --- | --- |
| `hearthstone/` | 自研纯 Python 引擎：44 随从 + 4 武器 + 12 法术 + 幸运币；**251 维特征（v6，含卡面文本感知）**；PPO+GAE(λ=0.5)、68k 参数；整回合 beam 搜索（靠 8µs 的 `Game.clone()`）；三个手写 bot（random/greedy/rule）；137 个测试；训练 ~125 局/s |
| `rosetta/` | C++ 绑定实验模块：`reset/legal_actions/step/observe` 四件套 + `Env` 门面 + bots/arena/play/tests。**它的 Env API 形状值得作为新模块的模板** |
| `policy.py` | 引擎无关：只要 `(局面, 动作) → 定长向量特征` 就能跑 PPO/GAE，全模块可直接复用 |

### 2.2 orange-stone 侧概览（RL 相关面）

引擎侧的 RL 环境（`GameEnv`：单智能体 vs 内置 bot、168 维观测、可配置奖励）、PyO3 绑定（`reset/observation/legal_actions/step`）、批量模拟（`sim/batch.rs`）、内置 bot（Greedy/Smart）等现状细节见 `orange-stone/docs/finished/rl-interface-roadmap.md` §2。

### 2.3 差距清单（G1~G10，按归属标注）

| # | 差距 | 现状 | 归属 |
| --- | --- | --- | --- |
| G1 | 绑定未安装 | wheel 在 `target/wheels/`，venv 里没有 | RL 侧（M0 装进 venv） |
| G2 | 无自定义卡组 | 卡组从全卡池随机生成（`deck_size` 张、单副内不重复） | 引擎侧（M1） |
| G3 | 无结构化观察/动作 | 只有 168 维向量 + `(index, 字符串描述)` | 引擎侧（M1） |
| G4 | 双方不可同时外部控制 | `GameEnv` 固定"智能体 vs 内置 bot"，EndTurn 后 bot 自动代打 | 引擎侧（M1） |
| G5 | clone 未暴露 | Rust 侧 CoW 便宜，但 py_bind 没有 clone | 引擎侧（M1） |
| G6 | 起手规则固定 | 双方各 3 张，无后手 4 张 + 幸运币 | 引擎侧（M1） |
| G7 | 奖励口径不同 | 稀疏胜负 +1/−1 | 引擎侧（M1） |
| G8 | 无 RL 批量 step | `batch.rs` 只支持 bot 驱动 | 引擎侧（M4） |
| G9 | 卡池不对齐 | 全经典池 vs 简化炉石的 44+4+12 子集 | RL 侧（建 G9 子集卡池） |
| G10 | 规则语义差异 | orange-stone 是全规则（职业/英雄技能/战吼/亡语/光环都在） | RL 侧（动作空间、特征、bot 适配） |

---

## 3. 路线图（orange-reinforcement 侧）

### M0 — 环境打通（约 1 天）✅ 已完成（2026-08-05）

- [x] 把 wheel 装进 `.venv`（`pip install target/wheels/orange_stone-*.whl`；装不上就 `cd orange-stone && maturin develop`）
- [x] 从 Python 跑通一局：`GameEnv(seed=42)` → 循环 `legal_actions()` + `step()` vs Greedy
- [x] 确定性测试：同 seed 两次完局的动作序列、终局 winner 逐位一致
- [x] 记基线吞吐（局/s），留作 M4 的对照

**验收**：一个 `test_env_smoke.py` 级别的脚本/测试，能完整打完一局并复现。

**M0 实测结果**（`tools/orange_stone_smoke.py`，M3 Pro）：
- 完局正常（随机策略 vs Greedy，平均 26 步/局，无死循环）；同 seed 两次完局动作序列/winner **逐位一致**（8 seed 全过）
- 基线吞吐：**~970 局/s、~25,000 步/s**（单进程；已比 rosetta 的 ~460 局/s 快 2 倍、自研引擎 ~125 局/s 快 7.7 倍）——M4 的对标基准
- 随机策略 vs Greedy 胜率 ~8~10%（合理：Greedy 会用满水晶）

**过程中处理的两个环境问题**（新机器复现时留意）：
1. `.venv` 是从旧路径 `~/Documents/orange-reinforcement/.venv` 迁移来的，所有入口脚本 shebang 失效——已就地批量修正路径并验证 torch/pip 正常
2. `maturin build` 默认误选系统 Python 3.9 导致 wheel 修复失败——必须显式 `--interpreter <venv>/bin/python`；且仓库内旧 wheel 过期（构建早于 `py_bind`/`rl/env.rs` 的改动），已用当前 HEAD 重编

### M2 — RL 侧新模块（新增 `hearthstone_os/`，约 1~2 周）✅ 已完成（2026-08-06）

**API 形状复刻 `rosetta/`**（`Env` 门面 + `Action`/`Observation` 四件套），bots/arena 可直接平移（决策点 D1-a）：

- [x] `env.py`：`reset(seed)` / `legal_actions()` / `step(action)` / `observe()` / `clone()`，底层是 `orange_stone.GameEnv`。双实例锁步（perspective 0/1 各一个 GameEnv，同 seed 同动作序列）解决"绑定层 perspective 固定但双 bot 对局双方都要看自己手牌"的问题
- [x] `bots.py`：random/greedy/rule 平移。三个适配点：动作按 `kind`/`card_index`/`entity_id`/`target_id`（无 rosetta 的 `target_side`/`target_pos`，打脸 = target 不在对方场上）；`EntityView` 只暴露基础关键词（剧毒/吸血等 M5 再说）；**RuleBot 补了两处 rosetta 没有的逻辑**——嘲讽强制交换时绕过负分阈值（否则被迫换嘲讽时直接跳过攻击，放任对面白打脸）、血量赛跑感知（自己血量明显落后/领先时按 0.9 权重抢脸）
- [x] `arena.py`：`duel` 多局评测、`matrix` 胜率矩阵、对角线 50% 校准（P1 永远先手，先后手轮换用"换座"实现）
- [x] `play.py`：观战/bench/matrix 命令行
- [x] `tests/`：`test_env.py` 三件套——完局、确定性、合法动作枚举一致性（用 `clone()` 逐个验证"枚举的动作执行后局面必变"）+ bots 强弱序；**`test_parity.py` 对拍测试**（G9 子集卡池、同 seed、受限随机策略，简版引擎与 orange-stone 各自完局，断言"动作类型包含关系 + 结局分布"）

**验收达成**：
- 胜率矩阵对角线 50%±2pp：每对 1000 局实测 greedy 50.5% / random 49.6% / rule 50.5%（`tools/orange_stone_m2_smoke.py` §3）
- 对拍测试在 G9 子集卡池上通过：40~80 seed 全部完局、简版动作类型 ⊆ os、同 seed 胜者一致率 ~61%、P1 胜率简版 ~34% vs os ~23%（官方先手第 1 回合不抽牌，os 后手优势更大，符合预期）
- 回归：hearthstone 224 + rosetta 18 + hearthstone_os 27 测试全过，M0/M1 冒烟不受影响

**过程中核实/处理的事实**：
- orange-stone 当时**没有英雄技能**（英雄是裸实体，`hero_power` 动作从未出现）——bots 里的英雄技能分支是 M5 预留；也没有潜行/扰咒实现（丛林豹/荆棘谷猛虎/精灵龙无对应字段），这几张因此**不进 G9 子集**（只收语义一致的白板+冲锋/嘲讽/圣盾/风怒共 28 张；潜行卡在引擎补潜行字段后随 M5 入池）
- rule vs greedy 只有 ~53%（rosetta 那边 55%+）：G9 子集没有潜行/扰咒/英雄技能这些 rule 能白嫖的点，优势被压缩，测试按实测只断言 >50%
- os 每回合开始抽牌、先手第 1 回合不抽（官方规则）；简版引擎回合开始就抽（先手 4 张起手）——对拍按统计口径对齐

### M3 — 训练接入（约 1~2 周）✅ 已完成（2026-08-06）

- [x] `features.py` 重写：**v7 = 199 维**（31 动作 + 168 局面），在结构化视图上移植 v5/v6 思路并**按当前引擎语义定版**——关键词收缩到视图暴露的 5 个（嘲讽/圣盾/潜行/风怒/冲锋）、无卡面文本块（G9 子集全白板）、无先知特征（绑定层不暴露对手手牌）、无英雄技能/疲劳（引擎没有）
- [x] `search.py`：用 `Env.clone()` 恢复整回合 beam 搜索（beam 8）——**rosetta 做不到、orange-stone 独有的收益**
- [x] `policy.py` 平移（无 oracle 分支）；`train.py`/`bench.py` 接新 env；`models/agent.pt` 入库
- [x] 回归基准：G9 子集卡池上纯策略 **69.2/69.8/70.8% vs rule**（三 seed，均值 69.9±0.7pp，对齐 v6 的 69.0% ±2pp）、+搜索 72.5%

**验收达成**：`train.py --episodes 30000` 三 seed 跑完（~90 局/s，每 seed 约 6 分钟）；战绩表对齐（多 seed：vs rule 69.9±0.7pp、vs greedy 78.5±1.9pp）。

**过程中发现并修掉的两个问题（对结果影响巨大）**：
1. **orange-stone #70（引擎保真 bug）**：`build_game_state` 把先手法力重置 0/0，覆盖了 `GameState::new()` 的第一回合水晶——先手整局少一个水晶，P1 侧训练系统性坍缩（P1 胜率 ~3%、P2 ~96%）。**v6 的 69% 战绩同样含座位分裂假象**（简版引擎同卡池对照实测 P1 82.7% / P2 44.7%，均值 ~64%）。修引擎后 P1-only 训练恢复（5k 局 43% 且上升）
2. **搜索停滞（本模块）**：价值头超估"快赢局面"（+1.065 > 终局 +1.0），搜索把结束回合排在真斩杀前，对 1 血空场对手无限拖回合。修复：价值裁剪 [−1,+1] + 终局斩杀无条件优先；修后 +搜索 全部完局（20/20 无平局）且带来正增量（vs rule +2pp、vs greedy +5pp）

**v6 基准 77.7%（+搜索）未复现的原因**：该数字同样含座位分裂假象；本模块诚实口径下 +搜索 vs rule = 72.5%（600 局），搜索增量小但为正。

### M4 — 批量与性能（约 1 周）✅ 已完成（2026-08-06）

- [x] Python 侧 `BatchedEnv`（`hearthstone_os/batched.py`）：线程级并行——**先决条件是绑定层释放 GIL**（orange-stone #71：热方法包 `py.allow_threads`，引擎纯 Rust + 逐局 RNG 线程安全；此前 GIL 把线程完全串行）
- [x] obs/动作直接出 `np.ndarray`/批量：`orange_stone.BatchEnv`（一次调用驱动 N 局、结构化观测给**当前行动方视角**——批量训练免掉双实例锁步、`reset_one` 单独重开）、`battle_batch`（rayon 批量，直接暴露 `sim/batch.rs`）；特征本来就是 `np.ndarray`，决策 padded 矩阵一次前向
- [x] 吞吐基准表入 README（hearthstone_os/README.md "M4 批量与性能"）：M0 基线 ~970 → **`battle_batch` ~4,200 局/s（≈9.2× rosetta 460，目标 ≥10× 未完全达成但同量级）** → `BatchedEnv` 4 线程 ~1,850 → `train --parallel 8` ~150-185 局/s（≈1.8× 单局训练）

**验收达成**：批量与单局逐 seed 一致（`test_batched.py`：确定性策略下 BatchedEnv 与单局 Env 12 局 winner 完全一致 + `battle_batch` 单局/批量一致）。

**注意**：≥50× 自研（125 局/s → 6250）的目标未达成——引擎批量已到 ~4,200，但训练侧仍受 Python 特征/前向的 GIL 串行限制；要再上量需要把特征和前向也批量下沉（GPU 批量推理是另一条路）。

### M5 — 卡池扩展与保真（持续，与 orange-stone Phase 3/4 同步）✅ 已完成（2026-08-06）

- [x] **卡池扩到全经典**：特征 v7+（223 维，A_TEXT/S_TEXT 卡面文本块回归）、全经典构筑池 **321 张**（413 唯一条目 ALL_CARDS − 67 简化债 − 衍生物/硬币；保真债清偿后 **391 张**）、`random_deck()` 套牌构筑逻辑、bots 法术/武器打分、`train --pool full`
- [x] **与 orange-stone 里程碑 F 同步**：F1~F5 内部完成（differential.rs）、外部 SabberStone 对照跑通（引擎侧 #75）；本侧对拍测试随卡池扩展复跑
- [x] **奖励与 GAE 调优重扫**（并行训练加速）：λ=0.5 → 65.4%、λ=0.95 → 65.6%、λ=1.0 → 61.4%（10k × 2 seed）。**λ=0.5 与 0.95 统计持平，λ=1.0 仍最差——v6 的"λ=0.5 最优"结论没有上漂**，默认值保持

**实测**：
- 全卡池训练 30k × 3 seed：vs rule **62.5-66.0%**（随机组牌口径）、vs greedy 70-74.5%、vs random 97.5-99%
- vanilla 口径重训（v7+）：vs rule **67.7%**、+搜索 **69.7%**（M3 基准 ±2pp 内保持）
- 压力测试 200 局随机套牌全部正常完局；54 个测试全过

**卡池执行 bug（已修）**：`decks.py::_load_debt_ids` 把简化注释记到上一张卡的 ID 上（PR #31 修——修前混进约 12 张简化卡、漏掉约 15 张干净卡，**M5 实测数字是修前卡池产出的，重训会漂移**）。修后提取器已提不到简化标记，`~/.cache/orange_stone_debt_ids.txt` 无需再维护。引擎侧保真债的清偿细节（W0~W7、differential 场景、账本归档）见 `orange-stone/docs/finished/rl-interface-roadmap.md` §4 M5。

### M6 — 发布（收尾）✅ 已完成（2026-08-06）

- [x] `play.py --human` 人机模式（终端输入动作下标，对手可选 random/greedy/rule）
- [x] `docs/hearthstone_os/index.html` 网页版 + `docs/index.html` 入口（战绩表 + 引擎说明 + 快速上手）。**偏差**：另外两个游戏的网页版是浏览器可玩（JS 引擎），orange-stone 版是静态信息页——浏览器可玩需要 JS/WASM 移植 Rust 引擎，超出收尾范围，留作后续
- [x] 战绩表、训练速度、模型权重入库：`hearthstone_os/models/agent.pt`（30k 局）、README 战绩表（每格 600 局）、训练速度 ~90 局/s（单局）/ ~170 局/s（并行）
- [x] `rosetta/` 去留决策：**保留**（决策点 D1 原判）——基于 382 卡全实现的 RosettaStone，是 AGPL 参考实现对照；等 hearthstone_os 全经典卡池铺开后按需再评

---

## 4. 决策点（按推荐顺序给出）

| # | 决策 | 选项 | 推荐 |
| --- | --- | --- | --- |
| D1 | 模块形态 | a) 新建 `hearthstone_os/`；b) 改造 `rosetta/` | **a)**。`rosetta/` 与 RosettaStone 深度耦合（pybind11 类型、GC 禁用 hack），改造成本 > 新写；且保留它可继续做正确性对照（它基于 382 卡全实现的 RosettaStone，是有价值的参考实现） |
| D2 | 特征管线 | a) 结构化视图 + RL 侧特征工程（251 维思路）；b) 直接吃原生 168 维 obs | **a) 起步**。特征工程是 hearthstone/ 两轮实验积累的资产（48→76→197→251），全部是"补信息比堆参数有效"的实证；b) 作为 M4/M5 的长期优化项（obs 张量化） |
| D3 | 卡池策略 | a) 先建 44+4+12 简化子集对齐；b) 直接全经典 316 | **a)**。快速建立与自研引擎的对拍口径和战绩参照；316 是 M5 的事 |
| D4 | 对拍基准 | a) 结果级（胜率矩阵/对角线）；b) 动作级（逐动作比对） | **a) 先，b) 后**。结果级先验证"没退化"，动作级再验证"语义一致"（后者工作量大，只对 G9 子集做） |
| D5 | 搜索策略 | a) 用 clone() 恢复整回合 beam；b) 先不恢复 | **a)**。这是切换引擎的**最大净收益**——rosetta 方案被删的构造函数堵死的东西，在 orange-stone 上是一行绑定 |

## 5. 风险与对策

| 风险 | 对策 |
| --- | --- |
| **规则语义差异**（G10）：orange-stone 全规则 vs 简化炉石砍掉的换牌/职业/英雄技能 | 明确"迁移不是逐位复现"：以胜率口径 + 行为统计（动作/回合等）对齐，不追求逐步一致；M2 对拍测试只断言"合法动作集包含关系 + 结局分布" |
| **引擎保真欠债**：个别卡可能仍有简化（引擎侧 F4 持续审计，账本见 orange-stone docs） | 训练卡池只用**已实现且通过 differential 的卡**；G9 子集优先挑白板/基础关键词卡 |
| **混沌训练**：环境训练结果不可逐位复现（<2pp 效应不可判） | 所有效果评估多 seed；基准表带 ±pp |
| **奖励口径漂移**（G7） | M1 就把 `final_reward` 参数化，训练和评测用同一口径 |
| **绑定层成为性能瓶颈**：Python↔Rust 每步往返 | M4 批量 + 张量化；必要时把搜索的整回合推演下沉到 Rust 侧（后续可选） |

## 6. 相关文档

- 引擎侧接口/批量/保真路线图：`orange-stone/docs/finished/rl-interface-roadmap.md`（M1 接口补齐、M4 批量绑定、M5 保真债清偿，中英双语）
- orange-stone 架构路线图：`orange-stone/docs/finished/architecture-roadmap.md`（Phase 4 = RL 接口；里程碑 G/F 是 M1/M5 引擎侧工作的前置）
- rosetta 经验（API 模板 + 三个教训）：`rosetta/README.md`
- 特征/训练实验史（v5/v6、GAE、搜索）：`hearthstone/README.md`
- 本模块战绩与基准表：`hearthstone_os/README.md`
