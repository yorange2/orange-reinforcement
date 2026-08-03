// Hearthstone web UI
"use strict";

let game, modelNet, modelLoaded = false;
let playerSeat = 0; // human always seat 0
let autoStep = false, stepDelay = 800;

// ================================================================ Init

async function init() {
    try {
        let resp = await fetch("model.json");
        let blob = await resp.json();
        modelNet = blob.net;
        modelLoaded = true;
        document.getElementById("model-info").textContent =
            blob.layers + " layers x " + blob.hidden + ", " + (blob.n_params/1000).toFixed(1) + "k params";
    } catch (e) {
        document.getElementById("model-info").textContent = "model not loaded — using rule bot";
    }
    newGame();
}

function newGame() {
    let seed = Math.floor(Math.random() * 2147483647);
    game = new Game(seededRng(seed), 0);
    document.getElementById("log").innerHTML = "";
    render();
}

// ================================================================ Render

function render() {
    if (game.finished) return showResult();
    let obs = game.observe();
    let en = 1 - obs.player;
    let isHuman = obs.player === playerSeat;

    let html = "<div class=board>";

    // Enemy side
    html += '<div class="side enemy">';
    html += `<div class=hero>对手 英雄 <span class=hp>${obs.enemy_hero_health}</span>`;
    if (obs.enemy_weapon_attack) html += ` 武器 ${obs.enemy_weapon_attack}/${obs.enemy_weapon_durability}`;
    html += `</div>`;
    html += `<div class=minions>${boardHtml(obs.enemy_board, false)}</div>`;
    html += "</div>";

    // Middle
    html += '<div class=middle>';
    html += `<div>手牌 ${obs.enemy_hand_size} 牌堆 ${obs.enemy_deck_size}</div>`;
    html += `<div class=turn>T${obs.turn+1} 玩家${obs.player} 水晶 ${obs.mana}/${obs.max_mana}`;
    if (obs.fatigue) html += ` 疲劳${obs.fatigue}`;
    html += `</div>`;
    html += `<div>牌堆 ${obs.deck_size} 手牌 ${obs.hand.length}</div>`;
    html += "</div>";

    // My side
    html += '<div class="side friendly">';
    html += `<div class=minions>${boardHtml(obs.board, true)}</div>`;
    html += `<div class=hero>我 英雄 <span class=hp>${obs.hero_health}</span>`;
    if (obs.hero_weapon_attack) html += ` 武器 ${obs.hero_weapon_attack}/${obs.hero_weapon_durability}`;
    html += `</div>`;
    html += "</div></div>";

    // Hand
    if (!isHuman && !autoStep) {
        html += `<div class=hand>${handHtml(obs, false)}</div>`;
    } else if (isHuman) {
        html += `<div class=hand>${handHtml(obs, true)}</div>`;
    }

    // Actions
    if (isHuman) {
        html += '<div class=actions>';
        let groups = {play:[], attack:[], end:[]};
        for (let i = 0; i < obs.legal.length; i++) {
            let a = obs.legal[i], label = describeAction(obs, a);
            if (a.kind === "play") groups.play.push({i, label});
            else if (a.kind === "attack") groups.attack.push({i, label});
            else groups.end.push({i, label});
        }
        for (let [title, items] of [["出牌", groups.play], ["攻击", groups.attack], ["", groups.end]]) {
            if (!items.length) continue;
            html += `<div class=group><b>${title}</b> `;
            for (let it of items)
                html += `<button onclick="doAction(${it.i})" title="#${it.i}">${it.label}</button> `;
            html += "</div>";
        }
        html += "</div>";
    } else if (autoStep) {
        html += '<div class=actions><button onclick="toggleAuto()">暂停</button></div>';
    } else {
        html += '<div class=actions><button onclick="doBotStep()">下一步</button> ';
        html += '<button onclick="toggleAuto()">自动</button></div>';
    }

    document.getElementById("app").innerHTML = html;

    if (!isHuman && autoStep) {
        setTimeout(() => doBotStep(), stepDelay);
    }
}

function boardHtml(board, clickable) {
    if (!board.length) return "（空）";
    return board.map((m,i) => {
        let words = m.stateWords().join("");
        let z = m.can_attack ? "" : "z";
        return `<span class=minion title="${m.card.name}">${m.card.name} ${m.attack}/${m.health}${words}${z}</span>`;
    }).join(" ");
}

function handHtml(obs, show) {
    let parts = [];
    for (let i = 0; i < obs.hand.length; i++) {
        let c = obs.hand[i];
        let affordable = c.cost <= obs.mana ? "" : "×";
        let text = c.spell ? c.name : (c.weapon ? `${c.name} ${c.attack}/${c.health}` : `${c.name} ${c.attack}/${c.health}`);
        let kw = c.keywords.join("");
        parts.push(`${affordable}${text}${kw} (${c.cost}费)`);
    }
    return parts.join(" &nbsp;|&nbsp; ");
}

function describeAction(obs, a) {
    if (a.kind === "end") return "结束";
    if (a.kind === "play") {
        let c = obs.hand[a.source];
        if ((c.spell_damage>0||c.spell_transform) && a.target !== HERO)
            return `${c.name}→[${a.target}]`;
        return c.spell ? `用${c.name}` : `出${c.name}`;
    }
    if (a.kind === "attack") {
        if (a.source === HERO_SOURCE) return `英雄→${a.target===HERO?"脸":"["+a.target+"]"}`;
        let m = obs.board[a.source];
        return `${m.card.name}→${a.target===HERO?"脸":"["+a.target+"]"}`;
    }
    return "?";
}

function showResult() {
    let obs = game.observe(game.current);
    let w = game.winner;
    let msg = w === null ? "平局！" : (w === playerSeat ? "你赢了！" : "对手赢了");
    document.getElementById("app").innerHTML = `
        <div class=result>
        <h2>${msg}</h2>
        <p>血量: ${game.hero_health[playerSeat]} vs ${game.hero_health[1-playerSeat]}</p>
        <p>回合: ${game.turns}</p>
        <button onclick="newGame()">再来一局</button>
        </div>`;
}

// ================================================================ Actions

function doAction(i) {
    if (game.finished) return;
    let obs = game.observe();
    if (obs.player !== playerSeat) return;
    let a = obs.legal[i];
    game.step(a);
    logAction(obs, a);
    if (!game.finished && game.current !== playerSeat) {
        if (autoStep) setTimeout(() => doBotStep(), stepDelay);
        else render();
    } else {
        render();
    }
}

function doBotStep() {
    if (game.finished) return;
    let obs = game.observe();
    if (obs.player === playerSeat) { render(); return; }

    let action;
    if (modelLoaded && obs.legal.length > 1) {
        let feats = batchFeatures(obs);
        let scores = scoreActions(modelNet, feats);
        let best = 0;
        for (let i = 1; i < scores.length; i++)
            if (scores[i] > scores[best]) best = i;
        action = obs.legal[best];
    } else {
        action = ruleBotChoose(obs);
    }

    game.step(action);
    logAction(obs, action);
    render();
}

function logAction(obs, a) {
    let log = document.getElementById("log");
    let text = describeAction(obs, a);
    if (obs.player === 0) text = "我: " + text;
    else text = "对手: " + text;
    log.innerHTML += text + "<br>";
    log.scrollTop = log.scrollHeight;
}

function toggleAuto() {
    autoStep = !autoStep;
    if (autoStep) doBotStep();
    else render();
}

// ================================================================ Keyboard

document.addEventListener("keydown", function(e) {
    if (game.finished) { if (e.key === "Enter") newGame(); return; }
    let obs = game.observe();
    if (obs.player !== playerSeat) return;

    if (e.key >= "0" && e.key <= "9") {
        let i = parseInt(e.key);
        if (i < obs.legal.length) doAction(i);
    } else if (e.key === "e" || e.key === "E") {
        for (let i = 0; i < obs.legal.length; i++)
            if (obs.legal[i].kind === "end") { doAction(i); break; }
    } else if (e.key === " ") {
        e.preventDefault();
        if (obs.current !== playerSeat) doBotStep();
    }
});
