// Hearthstone engine — cards, game state, keywords, weapons, spells, features, model, bots.
// Ported from hearthstone/*.py, kept in sync manually.

"use strict";

// ================================================================ Constants

const N_PLAYERS = 2;
const HERO_HEALTH = 30, MAX_MANA = 10, BOARD_LIMIT = 7, HAND_LIMIT = 10;
const COIN_MANA = 1;
const STARTING_FIRST = 3, STARTING_SECOND = 4;
const HERO = -1, HERO_SOURCE = -2;
const PLAY = "play", ATTACK = "attack", END = "end";

// ================================================================ Keywords

const CHARGE = "冲锋", RUSH = "突袭", TAUNT = "嘲讽", STEALTH = "潜行";
const DIVINE_SHIELD = "圣盾", POISONOUS = "剧毒", WINDFURY = "风怒";
const LIFESTEAL = "吸血", REBORN = "复生", ELUSIVE = "扰咒", SPELL_DAMAGE = "法术增强";
const KEYWORDS = [CHARGE,RUSH,TAUNT,STEALTH,DIVINE_SHIELD,POISONOUS,WINDFURY,LIFESTEAL,REBORN,ELUSIVE,SPELL_DAMAGE];
const KW_IDX = Object.fromEntries(KEYWORDS.map((k,i)=>[k,i]));

// ================================================================ Card Pool

// CardDef(name, cost, attack, health, keywords, spell, weapon, spell fields...)
// Normalized into objects for readability
function cd(name,cost,atk,hp,kw,spell,weapon,opts={}) {
    return {name,cost,attack:atk,health:hp,keywords:kw||[],spell:!!spell,weapon:!!weapon,
            spell_damage:opts.sd||0, spell_draw:opts.draw||0, spell_missiles:opts.ms||0,
            spell_aoe_enemy_minions:opts.aem||0, spell_aoe_all_enemies:opts.aae||0,
            spell_aoe_all:opts.aaa||0, spell_splash:opts.splash||0,
            spell_transform:!!opts.tf, spell_destroy_all:!!opts.da, spell_brawl:!!opts.br,
            __str__:null};
}

const POOL = [
cd('奥术飞弹',1,0,0,[],true,false,{ms:3}),
cd('刀扇',3,0,0,[],true,false,{draw:1,aem:1}),
cd('奥术智慧',3,0,0,[],true,false,{draw:2}),
cd('火球术',4,0,0,[],true,false,{sd:6}),
cd('变形术',4,0,0,[],true,false,{tf:true}),
cd('奉献',4,0,0,[],true,false,{aae:2}),
cd('地狱烈焰',4,0,0,[],true,false,{aaa:3}),
cd('横扫',4,0,0,[],true,false,{sd:4,splash:1}),
cd('绝命乱斗',5,0,0,[],true,false,{br:true}),
cd('烈焰风暴',7,0,0,[],true,false,{aem:4}),
cd('疾跑',7,0,0,[],true,false,{draw:4}),
cd('扭曲虚空',8,0,0,[],true,false,{da:true}),
cd('圣光的正义',1,1,4,[],false,true,{}),
cd('炽炎战斧',2,3,2,[],false,true,{}),
cd('刺客之刃',4,2,5,[],false,true,{}),
cd('奥金斧',5,5,2,[],false,true,{}),
cd('幽灵',0,1,1,[],false,false,{}),
cd('鱼人袭击者',1,2,1,[],false,false,{}),
cd('血沼迅猛龙',2,3,2,[],false,false,{}),
cd('河鳄',2,2,3,[],false,false,{}),
cd('岩浆暴怒者',3,5,1,[],false,false,{}),
cd('冰风雪人',4,4,5,[],false,false,{}),
cd('绿洲钳嘴龟',4,2,7,[],false,false,{}),
cd('石拳食人魔',6,6,7,[],false,false,{}),
cd('熔火恶犬',7,9,5,[],false,false,{}),
cd('战争傀儡',7,7,7,[],false,false,{}),
cd('石牙野猪',1,1,1,['冲锋'],false,false,{}),
cd('蓝腮战士',2,2,1,['冲锋'],false,false,{}),
cd('狼骑兵',3,3,1,['冲锋'],false,false,{}),
cd('暴风城骑士',4,2,5,['冲锋'],false,false,{}),
cd('鲁莽火箭兵',6,5,2,['冲锋'],false,false,{}),
cd('闪金镇步兵',1,1,2,['嘲讽'],false,false,{}),
cd('霜狼步兵',2,2,2,['嘲讽'],false,false,{}),
cd('铁鬃灰熊',3,3,3,['嘲讽'],false,false,{}),
cd('银背族长',3,1,4,['嘲讽'],false,false,{}),
cd('森金持盾卫士',4,3,5,['嘲讽'],false,false,{}),
cd('魔古山守望者',4,1,7,['嘲讽'],false,false,{}),
cd('藏宝海湾保镖',5,5,4,['嘲讽'],false,false,{}),
cd('竞技场主宰',6,6,5,['嘲讽'],false,false,{}),
cd('铁木树人',8,8,8,['嘲讽'],false,false,{}),
cd('丛林豹',3,4,2,['潜行'],false,false,{}),
cd('荆棘谷猛虎',5,5,5,['潜行'],false,false,{}),
cd('银色侍从',1,1,1,['圣盾'],false,false,{}),
cd('血色十字军战士',3,3,1,['圣盾'],false,false,{}),
cd('银月城卫兵',4,3,3,['圣盾'],false,false,{}),
cd('年轻的多头龙鹰',1,1,1,['风怒'],false,false,{}),
cd('风怒鹰身人',6,4,5,['风怒'],false,false,{}),
cd('蛇皇',3,2,3,['剧毒'],false,false,{}),
cd('玛克扎尔',6,2,8,['剧毒'],false,false,{}),
cd('沼泽水蛭',1,2,1,['吸血'],false,false,{}),
cd('凶恶的鳞甲兽',2,1,3,['吸血','突袭'],false,false,{}),
cd('不安分的木乃伊',4,3,2,['突袭','复生'],false,false,{}),
cd('阿曼尼狂战熊',7,5,7,['突袭','嘲讽'],false,false,{}),
cd('骸骨怨灵',4,2,5,['嘲讽','复生'],false,false,{}),
cd('荒野刺客',5,4,2,['潜行','复生'],false,false,{}),
cd('精灵龙',2,3,2,['扰咒'],false,false,{}),
cd('狗头人地卜师',2,2,2,['法术增强'],false,false,{}),
cd('达拉然法师',3,1,4,['法术增强'],false,false,{}),
cd('食人魔法师',4,4,4,['法术增强'],false,false,{}),
cd('大法师',6,4,7,['法术增强'],false,false,{}),
];
const CARD_INDEX = Object.fromEntries(POOL.map((c,i)=>[c.name,i]));
const THE_COIN = {name:"幸运币",cost:0,attack:0,health:0,keywords:[],spell:true,weapon:false,
                  spell_damage:0,spell_draw:0,spell_missiles:0,
                  spell_aoe_enemy_minions:0,spell_aoe_all_enemies:0,spell_aoe_all:0,
                  spell_splash:0,spell_transform:false,spell_destroy_all:false,spell_brawl:false};

// Card helpers
function cardStats(c) { return c.attack + c.health; }
function cardHas(c, kw) { return c.keywords.includes(kw); }
function cardStr(c) {
    if (c.spell) return c.name + "(" + c.cost + "费)";
    if (c.weapon) return c.name + "(" + c.cost + "费 " + c.attack + "/" + c.health + ")";
    let tail = c.keywords.length ? " " + c.keywords.join("") : "";
    return c.name + "(" + c.cost + "费 " + c.attack + "/" + c.health + tail + ")";
}
const DECK_SIZE = 30, COPIES = 2, DISTINCT = 15;

// ================================================================ Game State

function Minion(card, uid) {
    this.card = card; this.attack = card.attack; this.health = card.health;
    this.max_health = card.health; this.attacks_left = cardHas(card, WINDFURY) ? 2 : 1;
    this.just_played = true; this.divine_shield = cardHas(card, DIVINE_SHIELD);
    this.stealth = cardHas(card, STEALTH); this.reborn = cardHas(card, REBORN);
    this.uid = uid || 0;
}
Minion.prototype = {
    has(kw) { return cardHas(this.card, kw); },
    get asleep() { return this.just_played && !this.has(CHARGE) && !this.has(RUSH); },
    get can_attack() { return this.attacks_left > 0 && this.attack > 0 && !this.asleep; },
    get can_hit_face() { return this.can_attack && !(this.just_played && this.has(RUSH) && !this.has(CHARGE)); },
    get taunting() { return this.has(TAUNT) && !this.stealth; },
    get damaged() { return this.health < this.max_health; },
    get value() { return this.attack + this.health; },
    stateWords() {
        let w = [];
        for (let kw of this.card.keywords) {
            if (kw === DIVINE_SHIELD && !this.divine_shield) continue;
            if (kw === STEALTH && !this.stealth) continue;
            if (kw === REBORN && !this.reborn) continue;
            w.push(kw);
        }
        return w;
    },
    toString() {
        let w = this.stateWords().join("");
        let z = this.can_attack ? "" : "z";
        let x = this.attacks_left > 1 ? "x" + this.attacks_left : "";
        return this.card.name + " " + this.attack + "/" + this.health + w + z + x;
    }
};

function effectiveHp(m) {
    return m.health + (m.divine_shield ? 1 : 0);
}

function Game(rng, first) {
    rng = rng || Math.random;
    first = first || 0;
    this.rng = rng;
    this.first = first;
    this.reset();
}
Game.prototype = {
    reset() {
        // Build mirrored decks
        let pool = POOL.slice();
        shuffle(this.rng, pool);
        let picked = pool.slice(0, DISTINCT);
        let dl = [];
        for (let c of picked) { dl.push(c); dl.push(c); }
        let d1 = dl.slice(), d2 = dl.slice();
        shuffle(this.rng, d1); shuffle(this.rng, d2);
        this.decks = [d1, d2];
        this.hands = [[], []];
        this.boards = [[], []];
        this.hero_health = [HERO_HEALTH, HERO_HEALTH];
        this.mana = [0, 0]; this.max_mana = [0, 0];
        this.fatigue = [0, 0]; this.burned = [[], []];
        this.weapons = [null, null]; this.weapon_durability = [0, 0];
        this.hero_attacked = [false, false];
        this.turns = 0; this.finished = false; this.winner = null; this._uid = 0;

        for (let p = 0; p < N_PLAYERS; p++) {
            let n = p === this.first ? STARTING_FIRST : STARTING_SECOND;
            for (let i = 0; i < n; i++) this._draw(p);
            if (p !== this.first) this.hands[p].push(THE_COIN);
        }
        this.current = this.first;
        this._beginTurn();
    },

    _takeUid() { return ++this._uid; },

    _draw(player) {
        let d = this.decks[player];
        if (!d.length) { this.fatigue[player]++; this._dmgHero(player, this.fatigue[player]); return null; }
        let c = d.pop();
        if (this.hands[player].length >= HAND_LIMIT) { this.burned[player].push(c); return null; }
        this.hands[player].push(c);
        return c;
    },

    _dmgHero(p, n) { this.hero_health[p] -= n; },

    observe(player) {
        if (player === undefined) player = this.current;
        let en = 1 - player;
        let myW = this.weapons[player], enW = this.weapons[en];
        return {
            player, turn: this.turns, mana: this.mana[player], max_mana: this.max_mana[player],
            hand: this.hands[player].slice(),
            board: this.boards[player].slice(), enemy_board: this.boards[en].slice(),
            hero_health: this.hero_health[player], enemy_hero_health: this.hero_health[en],
            hero_weapon_attack: myW ? myW.attack : 0,
            hero_weapon_durability: this.weapon_durability[player],
            hero_attacked: this.hero_attacked[player],
            enemy_weapon_attack: enW ? enW.attack : 0,
            enemy_weapon_durability: this.weapon_durability[en],
            deck_size: this.decks[player].length, enemy_deck_size: this.decks[en].length,
            enemy_hand_size: this.hands[en].length,
            fatigue: this.fatigue[player], enemy_fatigue: this.fatigue[en],
            legal: this.legalActions(player),
        };
    },

    legalActions(player) {
        if (player === undefined) player = this.current;
        let moves = [], enBoard = this.boards[1-player], boardFull = this.boards[player].length >= BOARD_LIMIT;
        let seen = new Set();

        // Play cards
        for (let i = 0; i < this.hands[player].length; i++) {
            let card = this.hands[player][i];
            if (card.cost > this.mana[player]) continue;
            // Minion/weapon
            if (!card.spell) {
                if (seen.has(card.name)) continue;
                if (!card.weapon && boardFull) continue;
                seen.add(card.name);
                moves.push({kind:PLAY, source:i, target:HERO});
            }
            // Targeted spell (damage, transform)
            else if (card.spell_damage > 0 || card.spell_transform) {
                if (seen.has(card.name)) continue;
                seen.add(card.name);
                for (let j = 0; j < enBoard.length; j++)
                    moves.push({kind:PLAY, source:i, target:j});
                moves.push({kind:PLAY, source:i, target:HERO});
            }
            // Non-targeted spell
            else if (card.spell_draw>0||card.spell_missiles>0||card.spell_aoe_enemy_minions>0||
                     card.spell_aoe_all_enemies>0||card.spell_aoe_all>0||card.spell_destroy_all||
                     card.spell_brawl||card.name===THE_COIN.name) {
                if (seen.has(card.name)) continue;
                seen.add(card.name);
                moves.push({kind:PLAY, source:i, target:HERO});
            }
        }

        // Minion attacks
        let targets = [];
        let tauntTargets = [];
        for (let j = 0; j < enBoard.length; j++) {
            if (!enBoard[j].stealth) targets.push(j);
            if (enBoard[j].taunting) tauntTargets.push(j);
        }
        let faceOpen = tauntTargets.length === 0;
        let atkTargets = faceOpen ? targets : tauntTargets;

        for (let i = 0; i < this.boards[player].length; i++) {
            let m = this.boards[player][i];
            if (!m.can_attack) continue;
            if (faceOpen && m.can_hit_face)
                moves.push({kind:ATTACK, source:i, target:HERO});
            for (let j of atkTargets)
                moves.push({kind:ATTACK, source:i, target:j});
        }

        // Hero weapon attack
        let w = this.weapons[player];
        if (w && w.attack > 0 && !this.hero_attacked[player]) {
            if (faceOpen)
                moves.push({kind:ATTACK, source:HERO_SOURCE, target:HERO});
            for (let j of atkTargets)
                moves.push({kind:ATTACK, source:HERO_SOURCE, target:j});
        }

        moves.push({kind:END});
        return moves;
    },

    step(action) {
        if (this.finished) throw Error("game over");
        if (action.kind === PLAY) this._play(action.source, action.target);
        else if (action.kind === ATTACK) this._attack(action.source, action.target);
        else if (action.kind === END) this._endTurn();
    },

    _play(hi, target) {
        let p = this.current, card = this.hands[p][hi];
        if (card.cost > this.mana[p]) throw Error("not enough mana");
        if (!card.spell && !card.weapon && this.boards[p].length >= BOARD_LIMIT)
            throw Error("board full");
        this.hands[p].splice(hi, 1);
        this.mana[p] -= card.cost;
        if (card.spell) { this._cast(p, card, target); return; }
        if (card.weapon) { this.weapons[p] = card; this.weapon_durability[p] = card.health; return; }
        this.boards[p].push(new Minion(card, this._takeUid()));
    },

    _cast(player, card, target) {
        if (target === undefined) target = HERO;
        if (card.name === THE_COIN.name) { this.mana[player] += COIN_MANA; return; }
        let en = 1 - player;

        if (card.spell_draw > 0)
            for (let i = 0; i < card.spell_draw; i++) this._draw(player);

        if (card.spell_destroy_all) { this.boards[0] = []; this.boards[1] = []; }

        if (card.spell_brawl) {
            let all = [];
            for (let p = 0; p < 2; p++)
                for (let m of this.boards[p]) all.push({p, m});
            if (all.length) {
                let s = this._pick(all);
                for (let p = 0; p < 2; p++)
                    this.boards[p] = this.boards[p].filter(m => m.uid === s.m.uid);
            }
        }

        if (card.spell_transform) {
            let b = this.boards[en];
            if (target >= 0 && target < b.length) {
                let old = b[target];
                let sheep = new Minion({name:"绵羊",cost:1,attack:1,health:1,keywords:[]}, this._takeUid());
                sheep.just_played = old.just_played;
                sheep.attacks_left = old.attacks_left;
                b[target] = sheep;
            }
        }

        if (card.spell_aoe_enemy_minions > 0) {
            for (let m of this.boards[en]) this._hit(m, card.spell_aoe_enemy_minions);
        }
        if (card.spell_aoe_all_enemies > 0) {
            let d = card.spell_aoe_all_enemies;
            for (let m of this.boards[en]) this._hit(m, d);
            this._dmgHero(en, d);
        }
        if (card.spell_aoe_all > 0) {
            let d = card.spell_aoe_all;
            for (let p = 0; p < 2; p++)
                for (let m of this.boards[p]) this._hit(m, d);
            this._dmgHero(0, d); this._dmgHero(1, d);
        }

        if (card.spell_damage > 0) {
            if (target === HERO) this._dmgHero(en, card.spell_damage);
            else { let b = this.boards[en]; if (target>=0&&target<b.length) this._hit(b[target], card.spell_damage); }
        }
        if (card.spell_splash > 0) {
            for (let j = 0; j < this.boards[en].length; j++)
                if (j !== target) this._hit(this.boards[en][j], card.spell_splash);
            if (target !== HERO) this._dmgHero(en, card.spell_splash);
        }
        if (card.spell_missiles > 0) {
            for (let i = 0; i < card.spell_missiles; i++) {
                let c = [];
                for (let j = 0; j < this.boards[en].length; j++) c.push(j);
                if (this.hero_health[en] > 0) c.push(HERO);
                if (!c.length) break;
                let t = this._pick(c);
                if (t === HERO) this._dmgHero(en, 1);
                else this._hit(this.boards[en][t], 1);
            }
        }

        this._clearDead(); this._checkOver();
    },

    _attack(src, target) {
        let p = this.current, en = 1 - p;
        if (src === HERO_SOURCE) { this._heroAttack(p, target); return; }
        let b = this.boards[p], eb = this.boards[en];
        let att = b[src];
        if (!att.can_attack) throw Error("cannot attack");
        att.attacks_left--; att.stealth = false;

        if (target === HERO) {
            this._dmgHero(en, att.attack);
            this._drain(p, att, att.attack);
        } else {
            let def = eb[target];
            let out = this._hit(def, att.attack);
            let inc = this._hit(att, def.attack);
            if (out && att.has(POISONOUS)) def.health = 0;
            if (inc && def.has(POISONOUS)) att.health = 0;
            this._drain(p, att, out); this._drain(en, def, inc);
            this._clearDead();
        }
        this._checkOver();
    },

    _heroAttack(player, target) {
        let w = this.weapons[player], en = 1 - player;
        if (!w || this.hero_attacked[player]) throw Error("cannot hero attack");
        let eb = this.boards[en];

        if (target === HERO) {
            this._dmgHero(en, w.attack);
        } else {
            let def = eb[target];
            this._dmgHero(player, def.attack);
            let dealt = this._hit(def, w.attack);
            if (dealt && cardHas(w, POISONOUS)) def.health = 0;
            this._drain(player, w, dealt);
            this._clearDead();
        }
        this.weapon_durability[player]--;
        if (this.weapon_durability[player] <= 0) this.weapons[player] = null;
        this.hero_attacked[player] = true;
        this._checkOver();
    },

    _hit(minion, amount) {
        if (amount <= 0) return 0;
        if (minion.divine_shield) { minion.divine_shield = false; return 0; }
        minion.health -= amount;
        return amount;
    },

    _drain(player, source, dealt) {
        if (dealt <= 0) return;
        let hasLS = source.has ? source.has(LIFESTEAL) : cardHas(source, LIFESTEAL);
        if (!hasLS) return;
        this.hero_health[player] = Math.min(this.hero_health[player] + dealt, HERO_HEALTH);
    },

    _clearDead() {
        for (let p = 0; p < 2; p++) {
            let alive = [], nBefore = this.boards[p].length;
            for (let m of this.boards[p]) {
                if (m.health > 0) alive.push(m);
                else if (m.reborn && nBefore < BOARD_LIMIT && alive.length < BOARD_LIMIT) {
                    let back = new Minion(m.card, this._takeUid());
                    back.health = 1; back.reborn = false;
                    alive.push(back);
                }
            }
            this.boards[p] = alive;
        }
    },

    _checkOver() {
        let dead = [];
        for (let p = 0; p < 2; p++) if (this.hero_health[p] <= 0) dead.push(p);
        if (!dead.length) return;
        this.finished = true;
        this.winner = dead.length === 2 ? null : 1 - dead[0];
    },

    _endTurn() {
        this.turns++;
        if (this.turns >= 120) { this.finished = true; return; }
        this.current = 1 - this.current;
        this._beginTurn();
    },

    _beginTurn() {
        let p = this.current;
        this.max_mana[p] = Math.min(this.max_mana[p] + 1, MAX_MANA);
        this.mana[p] = this.max_mana[p];
        this.hero_attacked[p] = false;
        for (let m of this.boards[p]) {
            m.just_played = false;
            m.attacks_left = cardHas(m.card, WINDFURY) ? 2 : 1;
        }
        this._draw(p);
        this._checkOver();
    },

    _pick(arr) { return arr[Math.floor(this.rng() * arr.length)]; },
};

// ================================================================ Features

const STATE_OFFSET = 42;

function batchFeatures(obs) {
    let tail = stateFeatures(obs);
    let rows = [];
    for (let a of obs.legal) {
        let feats = actionFeatures(obs, a);
        feats.push(Math.min(obs.legal.length, 30) / 30);
        rows.push(feats.concat(tail));
    }
    return rows;
}

function actionFeatures(obs, a) {
    if (a.kind === PLAY) return playFeatures(obs, a);
    if (a.kind === ATTACK) return attackFeatures(obs, a);
    return endFeatures();
}

function kwVec(kws) {
    let v = new Array(KEYWORDS.length).fill(0);
    for (let kw of kws) { let i = KW_IDX[kw]; if (i !== undefined) v[i] = 1; }
    return v;
}

function playFeatures(obs, a) {
    let card = obs.hand[a.source], kw = kwVec(card.keywords);
    let left = obs.mana - card.cost;
    let isTargeted = card.spell_damage > 0 || card.spell_transform;
    let tgtMinion = a.target !== HERO && isTargeted;
    let defAtk = 0, defHp = 0, defTaunt = 0, defShield = 0, defPoison = 0, kills = 0;
    if (tgtMinion && a.target < obs.enemy_board.length) {
        let d = obs.enemy_board[a.target];
        defAtk = d.attack/10; defHp = d.health/10;
        defTaunt = d.taunting?1:0; defShield = d.divine_shield?1:0;
        defPoison = d.has(POISONOUS)?1:0;
        kills = 1;
    }
    return [
        1,0,0, card.cost/10, card.attack/10, card.health/10, ...kw,
        0,0,0,0,0,0,0,                                         // attacker zeros
        a.target===HERO?1:0, defAtk, defHp, defTaunt, defShield, defPoison,
        kills,0,0,0,0,0,0,0,                                    // trade zeros
        left/10, (card.cost===obs.mana&&card.cost>0)?1:0, obs.board.length>=6?1:0,
    ];
}

function attackFeatures(obs, a) {
    if (a.source === HERO_SOURCE) return heroAttackFeatures(obs, a);
    let att = obs.board[a.source], face = a.target === HERO;
    let defAtk=0,defHp=0,defTaunt=0,defShield=0,defPoison=0,defReborn=0,defLS=0;
    let killsDef=0,killsAtt=0,overkill=0,boardAfter=0;
    if (face) {
        boardAfter = (obs.board.reduce((s,m)=>s+m.attack,0) - obs.enemy_board.reduce((s,m)=>s+m.attack,0)) / 20;
    } else {
        let d = obs.enemy_board[a.target];
        defAtk=d.attack/10; defHp=d.health/10;
        defTaunt=d.taunting?1:0; defShield=d.divine_shield?1:0;
        defPoison=d.has(POISONOUS)?1:0; defReborn=d.reborn?1:0; defLS=d.has(LIFESTEAL)?1:0;
        let deh = effectiveHp(d), aeh = effectiveHp(att);
        killsDef = att.attack >= deh ? 1 : 0;
        killsAtt = d.attack >= aeh ? 1 : 0;
        overkill = Math.max(0, att.attack - deh) / 5;
        let myRest = obs.board.reduce((s,m)=>s+(m.uid===att.uid?0:m.attack),0);
        let enRest = obs.enemy_board.reduce((s,m)=>s+(m.uid===d.uid?0:m.attack),0);
        if (!killsDef) enRest += d.attack;
        if (!killsAtt) myRest += att.attack;
        boardAfter = (myRest - enRest) / 20;
    }
    return [
        0,1,0, 0,0,0, ...new Array(KEYWORDS.length).fill(0),
        att.attack/10, att.health/10, att.attacks_left-1,
        att.divine_shield?1:0, att.has(POISONOUS)?1:0, att.has(LIFESTEAL)?1:0, att.has(WINDFURY)?1:0,
        face?1:0, defAtk, defHp, defTaunt, defShield, defPoison,
        killsDef, killsAtt, overkill, boardAfter,
        defReborn, defLS, att.reborn?1:0, (att.has(CHARGE)||att.has(RUSH))?1:0,
        0,0,0,
    ];
}

function heroAttackFeatures(obs, a) {
    let watk = obs.hero_weapon_attack, face = a.target === HERO;
    let defAtk=0,defHp=0,defTaunt=0,defShield=0,defPoison=0,defReborn=0,defLS=0;
    let killsDef=0,overkill=0,boardAfter=0;
    if (face) {
        boardAfter = (obs.board.reduce((s,m)=>s+m.attack,0) - obs.enemy_board.reduce((s,m)=>s+m.attack,0)) / 20;
    } else {
        let d = obs.enemy_board[a.target];
        defAtk=d.attack/10; defHp=d.health/10;
        defTaunt=d.taunting?1:0; defShield=d.divine_shield?1:0;
        defPoison=d.has(POISONOUS)?1:0; defReborn=d.reborn?1:0; defLS=d.has(LIFESTEAL)?1:0;
        let deh = effectiveHp(d);
        killsDef = watk >= deh ? 1 : 0;
        overkill = Math.max(0, watk - deh) / 5;
        boardAfter = (obs.board.reduce((s,m)=>s+m.attack,0) - obs.enemy_board.reduce((s,m)=>s+(m===d?0:m.attack),0)) / 20;
    }
    return [
        0,1,0, 0,0,0, ...new Array(KEYWORDS.length).fill(0),
        watk/10, 0, 0, 0,0,0,0,
        face?1:0, defAtk, defHp, defTaunt, defShield, defPoison,
        killsDef, 0, overkill, boardAfter,
        defReborn, defLS, 0, 0,
        0,0,0,
    ];
}

function endFeatures() {
    return [0,0,1, 0,0,0, ...new Array(KEYWORDS.length).fill(0),
            0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0];
}

function stateFeatures(obs) {
    let my = obs.board, en = obs.enemy_board;
    let playable = obs.legal.filter(a=>a.kind===PLAY&&!obs.hand[a.source].spell);
    let playAtk = playable.reduce((s,a)=>s+obs.hand[a.source].attack,0);
    let playHp = playable.reduce((s,a)=>s+obs.hand[a.source].health,0);
    let hasCharge = playable.some(a=>cardHas(obs.hand[a.source],CHARGE))?1:0;
    let hasTaunt = playable.some(a=>cardHas(obs.hand[a.source],TAUNT))?1:0;
    let hasRush = playable.some(a=>cardHas(obs.hand[a.source],RUSH))?1:0;
    let iLethal = canKill(obs, obs.player)?1:0;
    let oLethal = canKill(obs, 1-obs.player)?1:0;
    let goFirst = obs.deck_size > obs.enemy_deck_size ? 1 : (obs.deck_size < obs.enemy_deck_size ? 0 : 0.5);
    return [
        obs.mana/10, obs.max_mana/10, obs.hero_health/30, obs.enemy_hero_health/30, obs.fatigue/10,
        obs.hero_weapon_attack/10, obs.hero_weapon_durability/5, obs.hero_attacked?1:0,
        obs.enemy_weapon_attack/10, obs.enemy_weapon_durability/5,
        obs.hand.length/10, playable.length/6, playAtk/20, playHp/30, hasCharge, hasTaunt, hasRush,
        my.length/7, en.length/7,
        my.reduce((s,m)=>s+m.attack,0)/20, en.reduce((s,m)=>s+m.attack,0)/20,
        my.reduce((s,m)=>s+m.health,0)/30, en.reduce((s,m)=>s+m.health,0)/30,
        obs.enemy_board.some(m=>m.taunting)?1:0,
        my.filter(m=>m.has(POISONOUS)).length, my.filter(m=>m.has(LIFESTEAL)).length,
        my.filter(m=>m.has(WINDFURY)).length, my.filter(m=>m.reborn).length,
        en.filter(m=>m.has(POISONOUS)).length, en.filter(m=>m.has(LIFESTEAL)).length,
        en.filter(m=>m.has(WINDFURY)).length, en.filter(m=>m.reborn).length,
        iLethal, oLethal, obs.deck_size/30, obs.enemy_hand_size/10, obs.enemy_fatigue/10,
        goFirst, 1,
    ];
}

function canKill(obs, player) {
    let board = player === obs.player ? obs.board : obs.enemy_board;
    let enHealth = player === obs.player ? obs.enemy_hero_health : obs.hero_health;
    let enBoard = player === obs.player ? obs.enemy_board : obs.board;
    if (enBoard.some(m=>m.taunting)) return false;
    let dmg = 0;
    for (let m of board) if (m.can_hit_face) dmg += m.attack * m.attacks_left;
    return dmg >= enHealth;
}

// ================================================================ Model

function forwardNet(layers, x) {
    for (let l of layers) {
        if (l.type === "linear") {
            x = l.w.map((row, i) => row.reduce((s, w, j) => s + w * x[j], 0) + l.b[i]);
        } else if (l.type === "layernorm") {
            let m = x.reduce((s, v) => s + v, 0) / x.length;
            let v = x.reduce((s, v) => s + (v - m) ** 2, 0) / x.length;
            let scale = Math.sqrt(v + l.eps);
            x = x.map((v, i) => (v - m) / scale * l.w[i] + l.b[i]);
        } else if (l.type === "relu") {
            x = x.map(v => v > 0 ? v : 0);
        }
    }
    return x[0];
}

function scoreActions(netLayers, features) {
    return features.map(f => forwardNet(netLayers, f));
}

// ================================================================ Bots

function bodyValue(m) {
    let base = m.attack + m.health;
    for (let kw of [TAUNT, LIFESTEAL, POISONOUS, WINDFURY])
        if (m.has(kw)) base += 0.5;
    if (m.divine_shield) base += 1;
    if (m.reborn) base += 0.5;
    return base;
}

function tradeValue(att, def) {
    let deh = effectiveHp(def), aeh = effectiveHp(att);
    let killsDef = att.attack >= deh, killsAtt = def.attack >= aeh;
    if (!killsDef && killsAtt) return att.has(POISONOUS) ? bodyValue(def)*0.8 : -10;
    let gain = killsDef ? bodyValue(def) : 0;
    let loss;
    if (att.divine_shield && !killsDef) loss = 0.8;
    else if (att.divine_shield && killsDef) loss = 1.5;
    else if (killsAtt) loss = bodyValue(att);
    else { let r = Math.min(def.attack, att.health) / att.max_health; loss = bodyValue(att) * r; }
    let score = gain - loss;
    if (killsDef && !killsAtt) score += 1.5;
    if (killsDef && att.has(POISONOUS)) score += 2;
    if (killsDef && att.has(LIFESTEAL) && att.attack >= 2) score += 0.3;
    return score;
}

function boardPower(board) {
    return board.reduce((s, m) => s + bodyValue(m), 0);
}

function ruleBotChoose(obs) {
    // Coin first
    for (let a of obs.legal) {
        if (a.kind === PLAY && obs.hand[a.source].name === THE_COIN.name) return a;
    }
    // Lethal
    if (canKill(obs, obs.player)) {
        let taunts = obs.enemy_board.filter(m => m.taunting);
        for (let a of obs.legal) {
            if (a.kind === ATTACK && taunts.length && taunts.some(t => obs.enemy_board[a.target]?.uid === t.uid)) return a;
        }
        for (let a of obs.legal) {
            if (a.kind === ATTACK && a.target === HERO) return a;
        }
    }
    // Spells
    let bestSpell = null, bestSpellScore = -999;
    for (let a of obs.legal) {
        if (a.kind !== PLAY) continue;
        let c = obs.hand[a.source];
        if (!c.spell) continue;
        let score = -999;
        if (c.spell_damage > 0 || c.spell_transform) {
            if (a.target === HERO) {
                let urg = 1 + Math.max(0, 10 - obs.enemy_hero_health) / 5;
                score = c.spell_damage * 0.7 * urg;
                if (c.spell_damage >= obs.enemy_hero_health) score = 100;
            } else {
                let d = obs.enemy_board[a.target];
                let kills = c.spell_damage >= effectiveHp(d) || c.spell_transform;
                score = kills ? bodyValue(d) + (d.has(POISONOUS)?1.5:0) + (d.taunting?1:0) : (c.spell_damage||4)*0.3;
            }
        } else if (c.spell_aoe_enemy_minions||c.spell_aoe_all_enemies||c.spell_aoe_all) {
            let dmg = c.spell_aoe_enemy_minions||c.spell_aoe_all_enemies||c.spell_aoe_all;
            score = obs.enemy_board.length ? obs.enemy_board.reduce((s,m)=>s+Math.min(dmg,m.health),0)*0.5 : 0;
        } else if (c.spell_destroy_all||c.spell_brawl) {
            let ep = boardPower(obs.enemy_board), mp = boardPower(obs.board);
            score = ep > mp*1.5 ? ep - mp : 0;
        } else if (c.spell_draw) {
            score = obs.mana >= c.cost + 2 ? c.spell_draw : 0;
        } else if (c.spell_missiles) {
            score = obs.enemy_board.length ? 3 : 0;
        }
        if (score > bestSpellScore) { bestSpellScore = score; bestSpell = a; }
    }
    if (bestSpell && bestSpellScore > 0) return bestSpell;

    // Attacks
    let ahead = boardPower(obs.board) >= boardPower(obs.enemy_board) * 1.2;
    let bestAtk = null, bestAtkScore = -999;
    for (let a of obs.legal) {
        if (a.kind !== ATTACK) continue;
        let score;
        if (a.source === HERO_SOURCE) {
            let watk = obs.hero_weapon_attack;
            if (a.target === HERO) {
                let urg = 1 + Math.max(0, 10 - obs.enemy_hero_health) / 10;
                score = watk * (ahead?0.9:0.6) * urg;
            } else {
                let d = obs.enemy_board[a.target];
                let kills = watk >= effectiveHp(d);
                score = kills ? bodyValue(d) + 1 : d.attack * 0.5 * -1;
            }
        } else {
            let att = obs.board[a.source];
            if (a.target === HERO) {
                let dmg = att.attack * att.attacks_left;
                let w = ahead ? 0.9 : 0.6;
                let urg = 1 + Math.max(0, 10 - obs.enemy_hero_health) / 10;
                if (att.health <= 1) urg += 0.5;
                score = dmg * w * urg;
            } else {
                score = tradeValue(att, obs.enemy_board[a.target]);
            }
        }
        if (score > bestAtkScore) { bestAtkScore = score; bestAtk = a; }
    }
    let threshold = ahead ? 1 : -5;
    if (bestAtk && bestAtkScore > threshold) return bestAtk;

    // Yolo
    for (let a of obs.legal) {
        if (a.kind !== ATTACK || a.source===HERO_SOURCE) continue;
        let att = obs.board[a.source];
        if (att.health > 1) continue;
        let score = a.target===HERO ? att.attack*att.attacks_left*0.8 : tradeValue(att, obs.enemy_board[a.target]);
        if (score > 0.5) return a;
    }

    // Play
    let plays = obs.legal.filter(a => a.kind===PLAY && !obs.hand[a.source].spell);
    if (plays.length) {
        plays.sort((a,b) => {
            let ca=obs.hand[a.source], cb=obs.hand[b.source];
            return cb.cost-ca.cost || (cb.attack+cb.health)-(ca.attack+ca.health) || cb.attack-ca.attack;
        });
        return plays[0];
    }

    return obs.legal[obs.legal.length-1]; // end turn
}

// ================================================================ Helpers

function shuffle(rng, arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        let j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
}

function seededRng(seed) {
    let s = seed | 0;
    return function() {
        s = (s * 1664525 + 1013904223) | 0;
        return (s >>> 0) / 4294967296;
    };
}
