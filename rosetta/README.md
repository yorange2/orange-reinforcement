# RosettaStone 对战环境（实验中）

基于 [RosettaStone](https://github.com/utilForever/RosettaStone)——一个 C++ 实现的炉石模拟器，经典模式 382/382 张卡全部实现。官方只带了 Python 卡牌数据库绑定（`pyRosetta`），**打不了牌**。这里补上了缺的那一半：

- `rosetta_env` C++ 扩展：把 `Game`/`Player`/zones/entities 包起来，提供
  `reset` / `legal_actions` / `step` / `observe` 四件套
- `Env` Python 门面
- 合法动作枚举器 + 三种动作（出牌、攻击、英雄技能）
- `RandomBot` / `GreedyBot` 两个手写规则对手（和 `hearthstone/` 那边同一套打法）
- `arena.duel` 多局评测

> ⚠️ **这是实验模块。** 最终想把 `hearthstone/` 的训练链接到真实规则上，
> 但当前还有几个没解决的问题（见下文"和自研引擎的差异"）。

## 构建

RosettaStone 需要 C++23 编译器、cmake 和 vcpkg。这些在 macOS arm64 上都没问题
（Apple clang 17 + cmake 4.4.2 => 2468 个 C++ 单测全过）。

```bash
# 1. 把 RosettaStone 拉到这个仓库的同级目录（如果还没有的话）
git clone --depth 1 https://github.com/utilForever/RosettaStone.git ~/Documents/RosettaStone

# 2. 编译绑定（首次会顺便编译 RosettaStone，约 5~10 分钟）
./rosetta/build.sh

# 3. 看一局
.venv/bin/python -m rosetta.play --bots greedy random
```

产物是 `rosetta/rosetta_env.cpython-312-darwin.so`，已经在 `.gitignore` 里。

如果 RosettaStone 不在 `~/Documents/RosettaStone`：

```bash
ROSETTA_ROOT=/path/to/RosettaStone ./rosetta/build.sh
```

## 速度

同职业镜像 + 经典白板套牌（15×2=30 张），随机对手互打，M3 Pro：

| 对手 | 步/局 | 局/秒 |
| --- | --- | --- |
| random vs random | ~121 | ~460 |
| greedy vs greedy | ~54 | ~410 |
| greedy vs random | ~46 | ~540 |

比自研 Pure Python 引擎（~125 局/秒）快 3~4 倍，但有一个大的限制：
**并行采样只能多进程，不能多线程**——RosettaStone 用全局静态 RNG（`effolkronium::random_static`），
一个进程里同一时刻只能有一局在跑。

## 基准

每格 2000 局，先后手轮换，同职业镜像 + 同构白板套牌（和 `hearthstone/` 那边卡池同源）：

| 选手 | vs random | vs greedy | vs rule |
| --- | --- | --- | --- |
| rule | 100.0% | 86.1% | 48.6% |
| greedy | 100.0% | 50.1% | — |
| random | — | — | — |

两条对角线（48.6% / 50.1%）都在 50% 附近，口径没偏。`rule` 打 `greedy` 是 86.1%，
比自研引擎（60.5%）还高——因为真实炉石有英雄技能（法师 1 点直伤），rule 能用它
破圣盾、补刀 1 血随从，greedy 只会无脑打脸。

## 和自研引擎的差异

### 没 clone()

RosettaStone 的 `Game` 拷贝和移动构造函数都是 `= delete`。
你那个整回合 beam 搜索（+14.4pp）在这完全没有对应物。要恢复它，
得在 C++ 里给一个满是裸指针、任务队列、光环和触发器的 `Game` 实现深拷贝。

### 没把完整炉石的卡池潜力变现

目前只放了一副白板随从套牌。RosettaStone 经典模式 382 张全部实现——
武器、法术、战吼、亡语、光环全有——但模型还没见过它们。

### 全局 RNG

`effolkronium::random_static` 是线程局部的，但一个进程里用过的种子会互相影响。
并行训练的正确姿势是 multiprocessing + 每个 worker 独立 seed。

### AGPL-3.0

RosettaStone 是 AGPL，你的仓库公开且挂着 GitHub Pages。
链接 `libRosettaStone.a` 意味着传染。如果要发布训练过的权重，
得想清楚许可证问题。

## 目录

```
rosetta/
  native/                  C++ 绑定层（pybind11）
    bindings.cpp           Env 类 + EntityView/PlayerView/Observation
    actions.cpp            合法动作枚举 + ApplyAction
    CMakeLists.txt         链接 libRosettaStone.a
    build.sh               一键构建（含上游 RosettaStone）
  env.py                   Python 门面
  decks.py                 套牌定义
  bots.py                  RandomBot / GreedyBot
  arena.py                 多局评测
  play.py                  命令行入口
  tests/
    test_env.py            14 个测试——完局、确定性、枚举一致性
```

## 可以接着做的事

1. **给 `hearthstone/policy.py` 接上这个环境**——特征编码要重写（引擎的语义变了），
   但 PPO/GAE 那套基本能复用
2. **把卡池从 15 张白板扩到更多**——至少加武器、法术和几个战吼随从，
   让维度的差距有意义
3. **补更多法术的识别逻辑**——现在英雄技能是唯一的"法术"，加了真法术后
   `RuleBot._best_hero_power` 要扩展成能区分火球/奥弹/AoE/变形
2. **给 `hearthstone/policy.py` 接上这个环境**——特征编码要重写（引擎的语义变了），
   但 PPO/GAE 那套基本能复用
3. **把卡池从 15 张白板扩到更多**——至少加武器、法术和几个战吼随从，
   让维度的差距有意义
4. **补更多法术的识别逻辑**——现在英雄技能是唯一的"法术"，加了真法术后
   `RuleBot._best_hero_power` 要扩展成能区分火球/奥弹/AoE/变形

## 上手

```bash
.venv/bin/python -m rosetta.play                          # 观战
.venv/bin/python -m rosetta.play --bots random random     # 两个随机对打
.venv/bin/python -m rosetta.play --bench 400              # 跑胜率
.venv/bin/python -m unittest discover -s rosetta/tests -t .
```
