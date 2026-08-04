// 合法动作枚举。RosettaStone 本身不提供这个——它的 C++ 单测都是手写具体的 task，
// 从来不需要"这一步能干什么"的完整列表。RL 环境需要，所以在这里补上。
#pragma once

#include <Rosetta/PlayMode/Games/Game.hpp>
#include <Rosetta/PlayMode/Models/Character.hpp>

#include <vector>

namespace RosettaEnv
{
enum class ActionType
{
    PLAY_CARD,
    ATTACK,
    HERO_POWER,
    END_TURN,
    CHOOSE,  // 发现 / 抉择 / 调度等待玩家选择时，这是唯一合法的动作类型
};

//! 目标用 (side, pos) 定位而不是实体指针——Python 侧拿到的必须是可序列化、
//! 且在 step 之后仍然可解释的东西。
//! side: 0 = 当前玩家, 1 = 对手；pos: -1 = 英雄, >= 0 = 随从在 field 里的下标。
struct Action
{
    ActionType type = ActionType::END_TURN;

    int handIdx = -1;     //!< PLAY_CARD：手牌下标
    int sourcePos = -1;   //!< ATTACK：-1 = 英雄，>= 0 = 随从下标
    int targetSide = -1;  //!< -1 表示没有目标
    int targetPos = -2;
    int fieldPos = -1;   //!< 随从落点，-1 = 放到最右边
    int chooseOne = 0;   //!< 抉择卡选哪一边（1 或 2），0 = 不是抉择卡
    int choice = -1;     //!< CHOOSE：被选中的实体 ID
};

//! 枚举当前玩家在当前局面下所有合法的动作。
std::vector<Action> LegalActions(RosettaStone::PlayMode::Game& game);

//! 把 (side, pos) 解回一个角色指针，找不到时返回 nullptr。
RosettaStone::PlayMode::Character* ResolveTarget(
    RosettaStone::PlayMode::Game& game, int side, int pos);

//! 在 game 上执行一个动作。
void ApplyAction(RosettaStone::PlayMode::Game& game, const Action& action);
}  // namespace RosettaEnv
