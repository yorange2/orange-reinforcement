# Game::Clone() 设计笔记

> **状态：已回退。** 2024-08-04 实现后在 RosettaStone 上游留了 200 行代码，
> 因原版仓库需要保持干净而回退。分析和实现记录保留供参考。

## 为什么需要

整回合搜索在当前项目里是效应最大的单项改进（+14.4pp vs rule）。
它的核心操作是：「克隆一个 `Game`，往前走几步，评估叶子，选最好的那条路」。
RosettaStone 的 `Game` 拷贝和移动构造函数都是 `= delete`，所以现在没法做。

## 为什么硬

`Game` 的内部结构是一个活跃的模拟器，不只是数据。拷贝它要处理好：

### 1. 实体图 (entity graph)

```cpp
std::vector<Entity*> entityList;   // 全局实体表，用 entity_id 做下标
```

每个 `Entity`（`Minion` / `Hero` / `Spell` / `Weapon`）持有指向其他实体的
裸指针（owner、zone 等）。拷贝时必须重建整张图并修正所有指针。

### 2. Zone 所有权

```cpp
// Player 持有
DeckZone  deckZone;
HandZone  handZone;
FieldZone fieldZone;    // vector<Minion*>
```

Zone 是值成员，但里面存的是指向 entityList 里实体的指针。
拷贝时要重建 zone 里的指针。

### 3. 任务系统 (task system)

```cpp
TaskQueue taskQueue;    // 正在排队的 ITask
TaskStack taskStack;    // 任务执行栈
```

每个 `ITask` 持有 `Entity*` 裸指针（source、target），还有虚函数。
拷贝 `TaskQueue` → 要给每个 task 做 `Clone()`（`ITask` 已经有纯虚的
`CloneImpl()`），然后修正 task 里指向新实体的指针。

`TaskStack` 同理 —— 它是 `vector<int>`（存储 entity ID）+ 其他元数据。

### 4. 触发器和光环 (triggers & auras)

```cpp
std::vector<IAura*> auras;
std::vector<std::shared_ptr<Trigger>> triggers;
std::vector<std::pair<Entity*, std::shared_ptr<IEffect>>> oneTurnEffects;
```

`Trigger` 用 `shared_ptr` 管理（相对好拷贝），但内部的函数对象和
捕获可能涉及实体指针。`IAura` 是裸指针，保存在 `m_ownedAuras`
（unique_ptr）里 —— 但 `auras` 里的都是原始指针，拷贝后要对齐。

### 5. 全局状态

```cpp
State state;
Step step;
Step nextStep;
int m_turn;
PlayerType m_currentPlayer;
EventMetaData currentEventData;  // unique_ptr
```

相对简单 —— 值拷贝即可。

### 6. Player 内部

```cpp
Player {
    Hero hero;          // 值对象（但 hero 继承 Entity）
    HeroPower power;    // 值对象（继承 Playable → Entity）
    Weapon weapon;      // 值对象
    PlayerAuraEffects playerAuraEffects;
    Choice* choice;     // 裸指针，指向动态分配的选择数据
    std::vector<std::shared_ptr<Trigger>> triggers;
}
```

- `Hero`/`HeroPower`/`Weapon` 是值成员，`Game` 拷贝时跟着拷。
  但它们内部的 `player` 指针要指向新 `Player`。
- `Player` 也有 `triggers`、`choice` 等。

### 7. 卡牌数据是共享的

```cpp
Card* card;  // 指向 Cards::GetInstance() 里的单例数据
```

`Card` 对象不需要拷贝 —— 它们是只读的全局数据。
但 `Spell` 实体的 `Card*` 指针要保留不变。

## 实现策略（备选，不在原版仓库里做）

### 方案 A：fork RosettaStone 加 Clone()

在 fork 的 `Sources/Rosetta/PlayMode/Games/Game.cpp` 里加 `Game::Clone()`。

**优点**：效率最高。
**缺点**：自己维护 fork，同步上游更新时有合并成本。

### 方案 B：通过 pybind11 暴露内部，Python 侧重建

用 pybind11 暴露 Entity/Player 的足够多内部状态，Python 侧读旧状态、构造新 Game。

**优点**：不改 C++。
**缺点**：状态太多（task queue、trigger、aura），基本不可能完整重建。

### 当前结论

不改 RosettaStone 原版。用到 Clone 时再 fork。
