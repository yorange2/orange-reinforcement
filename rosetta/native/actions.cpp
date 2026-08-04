#include "actions.hpp"

#include <Rosetta/PlayMode/Models/HeroPower.hpp>
#include <Rosetta/PlayMode/Models/Player.hpp>
#include <Rosetta/PlayMode/Tasks/PlayerTasks/AttackTask.hpp>
#include <Rosetta/PlayMode/Tasks/PlayerTasks/ChooseTask.hpp>
#include <Rosetta/PlayMode/Tasks/PlayerTasks/EndTurnTask.hpp>
#include <Rosetta/PlayMode/Tasks/PlayerTasks/HeroPowerTask.hpp>
#include <Rosetta/PlayMode/Tasks/PlayerTasks/PlayCardTask.hpp>
#include <Rosetta/PlayMode/Zones/FieldZone.hpp>
#include <Rosetta/PlayMode/Zones/HandZone.hpp>

#include <stdexcept>

using namespace RosettaStone;
using namespace RosettaStone::PlayMode;
using namespace RosettaStone::PlayMode::PlayerTasks;

namespace RosettaEnv
{
namespace
{
//! 把一个角色指针反解成 (side, pos)。
void EncodeTarget(Game& game, Character* character, int& side, int& pos)
{
    Player* cur = game.GetCurrentPlayer();
    side = (character->player == cur) ? 0 : 1;
    pos = (character == character->player->GetHero())
              ? -1
              : character->GetZonePosition();
}

void AddTargetedVariants(Game& game, const std::vector<Character*>& targets,
                         const Action& base, std::vector<Action>& out)
{
    for (Character* target : targets)
    {
        Action action = base;
        EncodeTarget(game, target, action.targetSide, action.targetPos);
        out.emplace_back(action);
    }
}

//! 枚举一张牌（或英雄技能）所有能打出去的方式。
//!
//! 候选目标取自 `Card::GetValidPlayTargets`，但必须再用 `IsValidPlayTarget`
//! 过一遍——`Generic::PlayCard` 和 `HeroPowerTask::Impl` 最终判的是后者，
//! 两边不一致就会枚举出"看起来合法、执行时被静默丢掉"的动作，
//! 上层机器人会在这种动作上无限循环。
void AddPlayVariants(Game& game, Playable* playable, const Action& base,
                     std::vector<Action>& out)
{
    bool added = false;

    for (Character* target :
         playable->card->GetValidPlayTargets(playable->player))
    {
        if (!playable->IsValidPlayTarget(target, base.chooseOne))
        {
            continue;
        }

        Action action = base;
        EncodeTarget(game, target, action.targetSide, action.targetPos);
        out.emplace_back(action);
        added = true;
    }

    // 一个合法目标都没有时，看看这张牌能不能无目标打出去。
    if (!added && playable->IsValidPlayTarget(nullptr, base.chooseOne))
    {
        out.emplace_back(base);
    }
}
}  // namespace

Character* ResolveTarget(Game& game, int side, int pos)
{
    if (side < 0)
    {
        return nullptr;
    }

    Player* player =
        (side == 0) ? game.GetCurrentPlayer() : game.GetOpponentPlayer();

    if (pos < 0)
    {
        return player->GetHero();
    }

    FieldZone* field = player->GetFieldZone();
    if (pos >= field->GetCount())
    {
        return nullptr;
    }

    return (*field)[pos];
}

std::vector<Action> LegalActions(Game& game)
{
    std::vector<Action> out;

    Player* cur = game.GetCurrentPlayer();
    Player* opp = game.GetOpponentPlayer();

    // 处于选择状态（发现 / 调度 / 抉择）时，只能做选择，别的都不合法。
    if (cur->choice)
    {
        for (const int entityID : cur->choice->choices)
        {
            Action action;
            action.type = ActionType::CHOOSE;
            action.choice = entityID;
            out.emplace_back(action);
        }

        return out;
    }

    // 出牌
    const bool fieldIsFull = cur->GetFieldZone()->IsFull();
    HandZone* hand = cur->GetHandZone();
    for (int i = 0; i < hand->GetCount(); ++i)
    {
        Playable* playable = (*hand)[i];
        if (!playable->IsPlayableByPlayer())
        {
            continue;
        }

        // 场上满了就下不了随从（Generic::PlayCard 第一件事就是判这个）
        if (fieldIsFull && playable->card->GetCardType() == CardType::MINION)
        {
            continue;
        }

        // 抉择卡两边分别算一次，其余的走 chooseOne = 0 这一条。
        const bool chooseOne = playable->HasChooseOne();
        const int firstChoice = chooseOne ? 1 : 0;
        const int lastChoice = chooseOne ? 2 : 0;

        for (int choice = firstChoice; choice <= lastChoice; ++choice)
        {
            if (!playable->IsPlayableByCardReq(choice))
            {
                continue;
            }

            Action base;
            base.type = ActionType::PLAY_CARD;
            base.handIdx = i;
            base.chooseOne = choice;

            AddPlayVariants(game, playable, base, out);
        }
    }

    // 随从攻击
    FieldZone* field = cur->GetFieldZone();
    for (int i = 0; i < field->GetCount(); ++i)
    {
        Minion* minion = (*field)[i];
        if (!minion->CanAttack())
        {
            continue;
        }

        Action base;
        base.type = ActionType::ATTACK;
        base.sourcePos = i;
        AddTargetedVariants(game, minion->GetValidAttackTargets(opp), base, out);
    }

    // 英雄攻击（装了武器才可能成立）
    Hero* hero = cur->GetHero();
    if (hero->CanAttack())
    {
        Action base;
        base.type = ActionType::ATTACK;
        base.sourcePos = -1;
        AddTargetedVariants(game, hero->GetValidAttackTargets(opp), base, out);
    }

    // 英雄技能。注意 HeroPower 没有重写 IsPlayableByPlayer，而 Playable 那版
    // **不看 EXHAUSTED**——漏了 IsExhausted() 就会把"这回合已经按过了"的技能
    // 当成合法动作，而 HeroPowerTask 会直接 STOP，局面纹丝不动。
    HeroPower& heroPower = cur->GetHeroPower();
    if (!heroPower.IsExhausted() && heroPower.IsPlayableByPlayer() &&
        heroPower.IsPlayableByCardReq())
    {
        Action base;
        base.type = ActionType::HERO_POWER;
        AddPlayVariants(game, &heroPower, base, out);
    }

    // 结束回合永远合法
    Action endTurn;
    endTurn.type = ActionType::END_TURN;
    out.emplace_back(endTurn);

    return out;
}

void ApplyAction(Game& game, const Action& action)
{
    Player* cur = game.GetCurrentPlayer();

    switch (action.type)
    {
        case ActionType::CHOOSE:
            game.Process(cur, ChooseTask::Pick(cur, action.choice));
            break;

        case ActionType::PLAY_CARD:
        {
            HandZone* hand = cur->GetHandZone();
            if (action.handIdx < 0 || action.handIdx >= hand->GetCount())
            {
                throw std::runtime_error("PLAY_CARD 的手牌下标越界");
            }

            Playable* source = (*hand)[action.handIdx];
            Character* target =
                ResolveTarget(game, action.targetSide, action.targetPos);

            game.Process(cur, PlayCardTask(source, target, action.fieldPos,
                                           action.chooseOne));
            break;
        }

        case ActionType::ATTACK:
        {
            Character* source =
                (action.sourcePos < 0)
                    ? static_cast<Character*>(cur->GetHero())
                    : static_cast<Character*>((*cur->GetFieldZone())[action.sourcePos]);
            Character* target =
                ResolveTarget(game, action.targetSide, action.targetPos);

            if (source == nullptr || target == nullptr)
            {
                throw std::runtime_error("ATTACK 的攻击者或目标解不出来");
            }

            game.Process(cur, AttackTask(source, target));
            break;
        }

        case ActionType::HERO_POWER:
        {
            Character* target =
                ResolveTarget(game, action.targetSide, action.targetPos);
            game.Process(cur, HeroPowerTask(target));
            break;
        }

        case ActionType::END_TURN:
            game.Process(cur, EndTurnTask());
            break;
    }
}
}  // namespace RosettaEnv
