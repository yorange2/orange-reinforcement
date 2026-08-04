"""基于 RosettaStone 的完整炉石对战环境。

和 `hearthstone/` 那个从零写的简化版不同，这里的规则来自
[RosettaStone](https://github.com/utilForever/RosettaStone)——一个 C++ 实现的
炉石模拟器，经典模式 382/382 张卡全部实现。
"""

from .env import Action, ActionType, Env, describe_action

__all__ = ["Action", "ActionType", "Env", "describe_action"]
