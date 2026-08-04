// rosetta_env —— 把 RosettaStone 的 PlayMode 包成一个可以从 Python 驱动的对局环境。
//
// 官方的 pyRosetta 只导出卡牌数据库（Card / Cards / Deck / DeckCode + 枚举），
// 没有 Game、没有 Player、没有任何 task，所以打不了牌。这里补上缺的那一半。
//
// Game 的拷贝和移动构造函数都是 = delete，所以 Env 只能持有 unique_ptr，
// 也因此这个环境**没有 clone()**——树搜索需要另想办法。

#include "actions.hpp"

#include <Rosetta/Common/Utils.hpp>
#include <Rosetta/PlayMode/Actions/Choose.hpp>
#include <Rosetta/PlayMode/Actions/Generic.hpp>
#include <Rosetta/PlayMode/Cards/Cards.hpp>
#include <Rosetta/PlayMode/Games/Game.hpp>
#include <Rosetta/PlayMode/Games/GameConfig.hpp>
#include <Rosetta/PlayMode/Models/HeroPower.hpp>
#include <Rosetta/PlayMode/Models/Player.hpp>
#include <Rosetta/PlayMode/Models/Weapon.hpp>
#include <Rosetta/PlayMode/Zones/DeckZone.hpp>
#include <Rosetta/PlayMode/Zones/FieldZone.hpp>
#include <Rosetta/PlayMode/Zones/HandZone.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

using namespace RosettaStone;
using namespace RosettaStone::PlayMode;
using RosettaEnv::Action;
using RosettaEnv::ActionType;

namespace
{
//! 一个随从 / 英雄 / 手牌在 Python 侧的样子。
struct EntityView
{
    int entityID = 0;
    std::string cardID;
    std::string name;

    int cost = 0;
    int attack = 0;
    int health = 0;

    bool canAttack = false;
    bool taunt = false;
    bool divineShield = false;
    bool stealth = false;
    bool poisonous = false;
    bool windfury = false;
    bool lifesteal = false;
    bool rush = false;
    bool charge = false;
    bool frozen = false;

    bool playable = false;  //!< 只对手牌有意义
};

struct PlayerView
{
    int heroHealth = 0;
    int heroArmor = 0;
    int heroAttack = 0;

    int remainingMana = 0;
    int totalMana = 0;

    int handCount = 0;
    int deckCount = 0;

    int weaponAttack = 0;
    int weaponDurability = 0;

    bool heroPowerUsable = false;

    std::vector<EntityView> field;
    std::vector<EntityView> hand;  //!< 对手的这一项是空的
};

struct Observation
{
    int turn = 0;
    bool myTurn = true;  //!< 恒为 true——观测总是从当前行动方的视角给出
    bool done = false;
    int winner = 0;  //!< 0 = 未结束或平局, 1 = player1, 2 = player2
    bool awaitingChoice = false;

    PlayerView me;
    PlayerView opponent;
};

EntityView ViewOf(Playable* playable, bool isHand)
{
    EntityView view;
    view.entityID = playable->GetGameTag(GameTag::ENTITY_ID);
    view.cardID = playable->card->id;
    view.name = playable->card->name;
    view.cost = playable->GetCost();

    if (auto* character = dynamic_cast<Character*>(playable); character)
    {
        view.attack = character->GetAttack();
        view.health = character->GetHealth();
        view.canAttack = !isHand && character->CanAttack();
    }

    view.taunt = playable->GetGameTag(GameTag::TAUNT) == 1;
    view.divineShield = playable->GetGameTag(GameTag::DIVINE_SHIELD) == 1;
    view.stealth = playable->GetGameTag(GameTag::STEALTH) == 1;
    view.poisonous = playable->GetGameTag(GameTag::POISONOUS) == 1;
    view.windfury = playable->GetGameTag(GameTag::WINDFURY) == 1;
    view.lifesteal = playable->GetGameTag(GameTag::LIFESTEAL) == 1;
    view.rush = playable->GetGameTag(GameTag::RUSH) == 1;
    view.charge = playable->GetGameTag(GameTag::CHARGE) == 1;
    view.frozen = playable->GetGameTag(GameTag::FROZEN) == 1;

    if (isHand)
    {
        view.playable = playable->IsPlayableByPlayer() &&
                        playable->IsPlayableByCardReq();
    }

    return view;
}

PlayerView ViewOf(Player* player, bool includeHand)
{
    PlayerView view;

    Hero* hero = player->GetHero();
    view.heroHealth = hero->GetHealth();
    view.heroArmor = hero->GetArmor();
    view.heroAttack = hero->GetAttack();

    view.remainingMana = player->GetRemainingMana();
    view.totalMana = player->GetTotalMana();

    view.handCount = player->GetHandZone()->GetCount();
    view.deckCount = player->GetDeckZone()->GetCount();

    if (hero->HasWeapon())
    {
        view.weaponAttack = player->GetWeapon().GetAttack();
        view.weaponDurability = player->GetWeapon().GetDurability();
    }

    HeroPower& heroPower = player->GetHeroPower();
    view.heroPowerUsable =
        heroPower.IsPlayableByPlayer() && heroPower.IsPlayableByCardReq();

    FieldZone* field = player->GetFieldZone();
    for (int i = 0; i < field->GetCount(); ++i)
    {
        view.field.emplace_back(ViewOf((*field)[i], false));
    }

    if (includeHand)
    {
        HandZone* hand = player->GetHandZone();
        for (int i = 0; i < hand->GetCount(); ++i)
        {
            view.hand.emplace_back(ViewOf((*hand)[i], true));
        }
    }

    return view;
}

//! 局面指纹，只用来判"这一步到底动了没有"。
//!
//! RosettaStone 的 task 在参数不合法时是 `return` / `TaskStatus::STOP`——
//! 不抛异常、不报错，局面原封不动。如果动作枚举和引擎的判定有一丁点不一致，
//! 上层就会拿到一个永远执行不掉的动作，然后在上面无限循环。
//! 与其让它表现成"跑了 5000 步判平局"，不如当场炸出来。
void MixInto(std::uint64_t& hash, std::uint64_t value)
{
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
}

void MixPlayer(std::uint64_t& hash, Player* player)
{
    Hero* hero = player->GetHero();
    MixInto(hash, static_cast<std::uint64_t>(hero->GetHealth()));
    MixInto(hash, static_cast<std::uint64_t>(hero->GetArmor()));
    MixInto(hash, static_cast<std::uint64_t>(hero->GetAttack()));
    MixInto(hash, static_cast<std::uint64_t>(player->GetRemainingMana()));
    MixInto(hash, static_cast<std::uint64_t>(player->GetTotalMana()));
    MixInto(hash, static_cast<std::uint64_t>(player->GetHandZone()->GetCount()));
    MixInto(hash, static_cast<std::uint64_t>(player->GetDeckZone()->GetCount()));
    MixInto(hash, player->GetHeroPower().IsExhausted() ? 1u : 0u);

    FieldZone* field = player->GetFieldZone();
    MixInto(hash, static_cast<std::uint64_t>(field->GetCount()));
    for (int i = 0; i < field->GetCount(); ++i)
    {
        Minion* minion = (*field)[i];
        MixInto(hash, static_cast<std::uint64_t>(
                          minion->GetGameTag(GameTag::ENTITY_ID)));
        MixInto(hash, static_cast<std::uint64_t>(minion->GetAttack()));
        MixInto(hash, static_cast<std::uint64_t>(minion->GetHealth()));
        MixInto(hash, minion->CanAttack() ? 1u : 0u);
        MixInto(hash, static_cast<std::uint64_t>(
                          minion->GetGameTag(GameTag::DIVINE_SHIELD)));
        MixInto(hash,
                static_cast<std::uint64_t>(minion->GetGameTag(GameTag::FROZEN)));
    }
}

std::uint64_t Fingerprint(Game& game)
{
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    MixInto(hash, static_cast<std::uint64_t>(game.GetTurn()));
    MixInto(hash, static_cast<std::uint64_t>(game.step));
    MixPlayer(hash, game.GetPlayer1());
    MixPlayer(hash, game.GetPlayer2());
    return hash;
}

//! PlayerType 没有 StrToEnum 特化（那套宏只覆盖了 .def 文件里定义的枚举），自己解。
PlayerType ParsePlayerType(const std::string& text)
{
    if (text == "PLAYER1")
    {
        return PlayerType::PLAYER1;
    }
    if (text == "PLAYER2")
    {
        return PlayerType::PLAYER2;
    }
    if (text == "RANDOM")
    {
        return PlayerType::RANDOM;
    }

    throw std::runtime_error("start_player 只能是 PLAYER1 / PLAYER2 / RANDOM，收到：" +
                             text);
}

//! 按卡名或卡 ID 找一张牌。
Card* FindCard(const std::string& nameOrID)
{
    Card* card = Cards::FindCardByID(nameOrID);
    if (card != nullptr && card->id == nameOrID)
    {
        return card;
    }

    card = Cards::FindCardByName(nameOrID);
    if (card == nullptr || card->id.empty())
    {
        throw std::runtime_error("找不到这张卡：" + nameOrID);
    }

    return card;
}

class Env
{
 public:
    Env(std::string player1Class, std::string player2Class,
        std::vector<std::string> player1Deck,
        std::vector<std::string> player2Deck, bool skipMulligan,
        std::string startPlayer)
        : m_player1Class(std::move(player1Class)),
          m_player2Class(std::move(player2Class)),
          m_player1Deck(std::move(player1Deck)),
          m_player2Deck(std::move(player2Deck)),
          m_skipMulligan(skipMulligan),
          m_startPlayer(std::move(startPlayer))
    {
        Cards::GetInstance();  // 第一次会把 cards.json 读进来
    }

    //! 开一局新的。seed 是进程级的——RosettaStone 用的是一个全局静态 RNG，
    //! 所以并行采样只能多进程，不能多线程。
    void Reset(int seed)
    {
        if (seed >= 0)
        {
            Random::seed(static_cast<unsigned int>(seed));
        }

        GameConfig config;
        config.player1Class = StrToEnum<CardClass>(m_player1Class);
        config.player2Class = StrToEnum<CardClass>(m_player2Class);
        config.startPlayer = ParsePlayerType(m_startPlayer);
        config.skipMulligan = m_skipMulligan;
        config.autoRun = false;

        if (m_player1Deck.empty() || m_player2Deck.empty())
        {
            config.doFillDecks = true;
        }
        else
        {
            FillDeck(config.player1Deck, m_player1Deck);
            FillDeck(config.player2Deck, m_player2Deck);
        }

        m_game = std::make_unique<Game>(config);
        m_game->Start();

        if (!m_skipMulligan)
        {
            // 调度阶段两边都留牌。真要做换牌，得把 mulligan 也做成动作暴露出去。
            Generic::ChoiceMulligan(m_game->GetPlayer1(), {});
            Generic::ChoiceMulligan(m_game->GetPlayer2(), {});
        }

        Advance();
    }

    std::vector<Action> LegalActions()
    {
        RequireGame();

        if (IsDone())
        {
            return {};
        }

        return RosettaEnv::LegalActions(*m_game);
    }

    void Step(const Action& action)
    {
        RequireGame();

        if (IsDone())
        {
            throw std::runtime_error("这一局已经结束了");
        }

        const std::uint64_t before = Fingerprint(*m_game);
        RosettaEnv::ApplyAction(*m_game, action);

        // 结束回合和做选择本来就可能不改变这些量，只守出牌 / 攻击 / 英雄技能。
        const bool shouldChange = action.type == ActionType::PLAY_CARD ||
                                  action.type == ActionType::ATTACK ||
                                  action.type == ActionType::HERO_POWER;
        if (shouldChange && Fingerprint(*m_game) == before)
        {
            throw std::runtime_error(
                "这个动作被引擎静默丢掉了，局面没有任何变化——"
                "说明动作枚举和引擎的合法性判定对不上（actions.cpp）");
        }

        Advance();
    }

    bool IsDone()
    {
        RequireGame();
        return m_game->state == State::COMPLETE;
    }

    //! 0 = 未结束或平局, 1 = player1 赢, 2 = player2 赢
    int Winner()
    {
        RequireGame();

        if (!IsDone())
        {
            return 0;
        }

        if (m_game->GetPlayer1()->playState == PlayState::WON)
        {
            return 1;
        }
        if (m_game->GetPlayer2()->playState == PlayState::WON)
        {
            return 2;
        }

        return 0;  // 双方同时归零 = 平局
    }

    //! 当前行动方是 1 还是 2。
    int CurrentPlayer()
    {
        RequireGame();
        return (m_game->GetCurrentPlayer() == m_game->GetPlayer1()) ? 1 : 2;
    }

    int Turn()
    {
        RequireGame();
        return m_game->GetTurn();
    }

    Observation Observe()
    {
        RequireGame();

        Observation obs;
        obs.turn = m_game->GetTurn();
        obs.done = IsDone();
        obs.winner = Winner();
        obs.awaitingChoice =
            !obs.done && m_game->GetCurrentPlayer()->choice != nullptr;

        obs.me = ViewOf(m_game->GetCurrentPlayer(), true);
        obs.opponent = ViewOf(m_game->GetOpponentPlayer(), false);

        return obs;
    }

 private:
    void RequireGame() const
    {
        if (!m_game)
        {
            throw std::runtime_error("还没开局，先调用 reset()");
        }
    }

    //! 推进到下一个需要玩家决策的点。注意 ProcessUntil 是个
    //! `while (nextStep != untilStep)` 的死循环，游戏已经结束时永远走不到
    //! MAIN_ACTION，所以必须先判 COMPLETE 再进去。
    void Advance()
    {
        if (m_game->state == State::COMPLETE)
        {
            return;
        }

        if (m_game->GetCurrentPlayer()->choice != nullptr)
        {
            return;  // 停在选择状态上，交给 LegalActions 去枚举
        }

        if (m_game->step != Step::MAIN_ACTION)
        {
            m_game->ProcessUntil(Step::MAIN_ACTION);
        }
    }

    static void FillDeck(std::array<Card*, START_DECK_SIZE>& target,
                         const std::vector<std::string>& source)
    {
        if (source.size() != START_DECK_SIZE)
        {
            throw std::runtime_error("套牌必须正好 " +
                                     std::to_string(START_DECK_SIZE) +
                                     " 张，收到 " +
                                     std::to_string(source.size()) + " 张");
        }

        for (std::size_t i = 0; i < START_DECK_SIZE; ++i)
        {
            target[i] = FindCard(source[i]);
        }
    }

    std::string m_player1Class;
    std::string m_player2Class;
    std::vector<std::string> m_player1Deck;
    std::vector<std::string> m_player2Deck;
    bool m_skipMulligan;
    std::string m_startPlayer;

    std::unique_ptr<Game> m_game;
};
}  // namespace

PYBIND11_MODULE(rosetta_env, m)
{
    m.doc() = "RosettaStone 炉石引擎的对局环境绑定";

    py::enum_<ActionType>(m, "ActionType")
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("ATTACK", ActionType::ATTACK)
        .value("HERO_POWER", ActionType::HERO_POWER)
        .value("END_TURN", ActionType::END_TURN)
        .value("CHOOSE", ActionType::CHOOSE);

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("hand_idx", &Action::handIdx)
        .def_readwrite("source_pos", &Action::sourcePos)
        .def_readwrite("target_side", &Action::targetSide)
        .def_readwrite("target_pos", &Action::targetPos)
        .def_readwrite("field_pos", &Action::fieldPos)
        .def_readwrite("choose_one", &Action::chooseOne)
        .def_readwrite("choice", &Action::choice);

    py::class_<EntityView>(m, "Entity")
        .def_readonly("entity_id", &EntityView::entityID)
        .def_readonly("card_id", &EntityView::cardID)
        .def_readonly("name", &EntityView::name)
        .def_readonly("cost", &EntityView::cost)
        .def_readonly("attack", &EntityView::attack)
        .def_readonly("health", &EntityView::health)
        .def_readonly("can_attack", &EntityView::canAttack)
        .def_readonly("taunt", &EntityView::taunt)
        .def_readonly("divine_shield", &EntityView::divineShield)
        .def_readonly("stealth", &EntityView::stealth)
        .def_readonly("poisonous", &EntityView::poisonous)
        .def_readonly("windfury", &EntityView::windfury)
        .def_readonly("lifesteal", &EntityView::lifesteal)
        .def_readonly("rush", &EntityView::rush)
        .def_readonly("charge", &EntityView::charge)
        .def_readonly("frozen", &EntityView::frozen)
        .def_readonly("playable", &EntityView::playable);

    py::class_<PlayerView>(m, "PlayerView")
        .def_readonly("hero_health", &PlayerView::heroHealth)
        .def_readonly("hero_armor", &PlayerView::heroArmor)
        .def_readonly("hero_attack", &PlayerView::heroAttack)
        .def_readonly("remaining_mana", &PlayerView::remainingMana)
        .def_readonly("total_mana", &PlayerView::totalMana)
        .def_readonly("hand_count", &PlayerView::handCount)
        .def_readonly("deck_count", &PlayerView::deckCount)
        .def_readonly("weapon_attack", &PlayerView::weaponAttack)
        .def_readonly("weapon_durability", &PlayerView::weaponDurability)
        .def_readonly("hero_power_usable", &PlayerView::heroPowerUsable)
        .def_readonly("field", &PlayerView::field)
        .def_readonly("hand", &PlayerView::hand);

    py::class_<Observation>(m, "Observation")
        .def_readonly("turn", &Observation::turn)
        .def_readonly("done", &Observation::done)
        .def_readonly("winner", &Observation::winner)
        .def_readonly("awaiting_choice", &Observation::awaitingChoice)
        .def_readonly("me", &Observation::me)
        .def_readonly("opponent", &Observation::opponent);

    py::class_<Env>(m, "Env")
        .def(py::init<std::string, std::string, std::vector<std::string>,
                      std::vector<std::string>, bool, std::string>(),
             py::arg("player1_class"), py::arg("player2_class"),
             py::arg("player1_deck") = std::vector<std::string>{},
             py::arg("player2_deck") = std::vector<std::string>{},
             py::arg("skip_mulligan") = true,
             py::arg("start_player") = "PLAYER1")
        .def("reset", &Env::Reset, py::arg("seed") = -1)
        .def("legal_actions", &Env::LegalActions)
        .def("step", &Env::Step, py::arg("action"))
        .def("observe", &Env::Observe)
        .def_property_readonly("done", &Env::IsDone)
        .def_property_readonly("winner", &Env::Winner)
        .def_property_readonly("current_player", &Env::CurrentPlayer)
        .def_property_readonly("turn", &Env::Turn);
}
