# orange-reinforcement

从零实现的棋牌游戏，配上用强化学习训练出来的对手。

每个游戏一个目录，自带引擎、规则对手、训练与评测脚本、命令行对战和网页版，互不干扰。

### 👉 [在线试玩](https://yorange2.github.io/orange-reinforcement/)

## 目前有什么

| 游戏 | 说明 | 战绩 |
| --- | --- | --- |
| [**跑得快**](paodekuai/) | 三人 16 张变体。PPO 训练的策略网络，对手是手写的启发式算法 | 夹在两个 `rule` 机器人中间拿 **53.7%**（随机基准 33.3%，`rule` 自己只有 34.9%） |
| [**简化版炉石**](hearthstone/) | 双人，60 张卡池（白板随从 + 法术 + 武器）。PPO + GAE 训练，推理时再叠一层整回合搜索 | 打 `rule` **80.2%**（随机基准 50%，`rule` 自己 50.6%） |

## 环境

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # torch + numpy
```

## 跑起来

以跑得快为例，详细说明见 [`paodekuai/README.md`](paodekuai/README.md)：

```bash
.venv/bin/python -m paodekuai.play                     # 跟模型对战，能看到它的打分
.venv/bin/python -m paodekuai.train                    # 训练（默认 2000 局，约 8 秒）
.venv/bin/python -m paodekuai.bench --model paodekuai/models/agent.pt
.venv/bin/python -m unittest discover -s paodekuai/tests -t .
```

简化版炉石见 [`hearthstone/README.md`](hearthstone/README.md)：

```bash
.venv/bin/python -m hearthstone.play                   # 跟规则机器人打一局
.venv/bin/python -m hearthstone.train                  # 训练（默认 2000 局，约 13 秒）
.venv/bin/python -m hearthstone.bench --model hearthstone/models/agent.pt --search
.venv/bin/python -m unittest discover -s hearthstone/tests -t .
```

## 加一个新游戏

照着 `paodekuai/` 的样子建一个目录就行，仓库层面没有需要改的东西：

- 引擎、手写规则对手、`(局面, 动作) -> 特征` 的编码各写一份
- `policy.py` 那套（逐动作打分 + 在合法动作上 softmax + PPO）基本可以照搬——
  它不依赖跑得快的任何细节，只要求动作能被编码成定长向量
- 网页版放 `docs/<游戏名>/`，再往 `docs/index.html` 加一个入口
