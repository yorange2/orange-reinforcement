# orange-stone vs RosettaStone 大规模对战对比（2026-08-07）

## 结论

**不完全一致**。随机策略下两引擎大规模对拍统计一致（P1 胜率差 0.88pp，
在 ±2pp 口径内）；节奏型策略（greedy）下发现一处**真实引擎差异**：**RosettaStone
的 The Coin（GAME_005）打出无效**——不提供 +1 本回合水晶，导致其后手失去
官方硬币补偿、先手优势被放大。**orange-stone 侧行为正确**（硬币 +1 本回合
水晶，符合官方规则）。差异源头在 RosettaStone。

## 方法

- **口径**：统计对拍。两引擎 RNG 语义不同（orange-stone 每局独立 GameRng
  可精确播种；RosettaStone 进程级全局静态 RNG，reset(seed) 只保证同进程
  顺序执行可复现），逐局对齐不可能，比终局分布。
- **设置**：MAGE 镜像 + 30 张等价白板套牌（15 种 × 2，两侧卡 ID 映射确认
  内容一致：幽灵/银色侍从/石牙野猪/血沼迅猛龙/蓝腮战士/霜狼步兵/精灵龙/
  狼骑兵/铁鬃灰熊/血色十字军战士/冰风雪人/森金持盾卫士/银月城卫兵/荆棘谷
  猛虎/石拳食人魔）；动作集对齐（orange-stone 未实现英雄技能 → rosetta 侧
  过滤 HERO_POWER）；先后手轮换；每侧先做 20 局同 seed 重跑**自洽验证**
  （两引擎均逐局一致，证明对拍数据可信）。
- **脚本**：`tools/compare_rosetta_orange.py`（`COMPARE_BOT=random|greedy|det_greedy`
  选择策略；det_greedy 是顺序无关的确定性变体，用于排除枚举顺序依赖）。

## 结果

### random vs random（N=5000/侧）

| 指标 | RosettaStone | orange-stone | 差值 |
| --- | --- | --- | --- |
| P1 胜率 | 49.78% | 48.90% | 0.88pp ✓（口径 ±2pp） |
| 平均步数 | 107.95 | 107.87 | 0.1% |
| 平均回合 | 32.35 | 32.30 | 0.05 |
| 平局率 | 0% | 0% | 0 |

### greedy vs greedy（N=3000/侧）

| 指标 | RosettaStone | orange-stone | 差值 |
| --- | --- | --- | --- |
| P1 胜率 | 81.8% | 54.7% | **27.1pp ✗** |
| 平均步数 | 53.8 | 54.4 | 1.1% |
| 平均回合 | 13.2 | 13.1 | 一致 |
| 胜者剩余血 | 9.6 | 4.9 | 4.7 |
| 败者剩余血 | 0.06 | 2.87 | 2.8 |

## 差异定位过程

逐一排除（每项两侧统计一致）：起手手牌（4/6/5 两侧相同）、步数、回合数、
回合级场面攻轨迹（±0.5 攻内）、攻击动作次数、打脸/打随从比例（22.6% vs
22.7%）、攻击顺序（确定性变体不消除差异）。

**根因**：greedy 下后手第 1 回合用硬币出 2 费随从是核心节奏操作——
实测 rosetta 打出硬币后 2 费随从**仍不可出**（硬币无效），orange 正常。
RosettaStone 源码中 GAME_005 只有分发给后手的任务
（`Sources/Rosetta/PlayMode/Games/Game.cpp:311`），**没有硬币自身的效果
定义**。

**因果验证**：orange 关掉硬币（`Env(second_player_coin=False)`）后 greedy
P1 胜率跳到 **85.1%**，与 rosetta 的 81.8% 收敛（剩余 ~3pp 为硬币占手牌位
的次生效应：rosetta 后手多一张废牌在手）。random 下差异被随机决策稀释
（双方浪费资源、硬币利用率低）→ 0.9pp。

## 判定

- orange-stone 硬币语义正确（+1 本回合水晶，官方规则；与 F-A9 及差分测试
  一致）。
- RosettaStone 的 The Coin 效果缺失（其经典卡实现未实现该效果）。
- 此前与 SabberStone 的单场景对照（combat）两引擎一致；本次是首次发现
  RosettaStone 侧的具体机制缺口。

## 复现

```bash
cd orange-reinforcement
.venv/bin/python tools/compare_rosetta_orange.py 5000            # random 对拍
COMPARE_BOT=greedy .venv/bin/python tools/compare_rosetta_orange.py 3000   # greedy 对拍
```
