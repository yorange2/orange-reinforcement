// 网页版跑得快的界面。牌局逻辑和打分都在 engine.js 里，这里只管交互和渲染。

import * as E from './engine.js';

const HUMAN = 0;
const $ = (id) => document.getElementById(id);

let model = null;
let game = null;
let selected = new Set();
let log = [];

// ------------------------------------------------------------------ 模型

/** 给每个候选打分，按概率从高到低返回。 */
function think(obs, top = Infinity) {
  const features = E.batchFeatures(obs);
  const scores = features.map((row) => E.score(model.net, row));
  const probs = E.softmax(scores);
  return obs.legal
    .map((move, i) => ({ move, score: scores[i], prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, top);
}

// ------------------------------------------------------------------ 渲染

const moveText = (move) => {
  if (!move) return '不要';
  const attached = new Set(E.attachmentRanks(move));
  const label = E.KIND_NAMES_CN[move.kind];
  if (!attached.size) return `${label} ${E.handToStr(move.cards)}`;
  const body = move.cards.filter((c) => !attached.has(E.rankOf(c)));
  const kicker = move.cards.filter((c) => attached.has(E.rankOf(c)));
  return `${label} ${E.handToStr(body)} 带 ${E.handToStr(kicker)}`;
};

function cardEl(card, { selectable = false } = {}) {
  const el = document.createElement('div');
  el.className = 'c' + ([0, 2].includes(E.suitOf(card)) ? ' red' : '');
  el.textContent = E.cardName(card);
  if (selectable) {
    if (selected.has(card)) el.classList.add('on');
    el.onclick = () => {
      selected.has(card) ? selected.delete(card) : selected.add(card);
      render();
    };
  } else {
    el.classList.add('dead');
  }
  return el;
}

function renderSeats() {
  const box = $('seats');
  box.innerHTML = '';
  for (let i = 0; i < 3; i++) {
    const seat = document.createElement('div');
    seat.className = 'seat' + (game.current === i && !game.finished ? ' turn' : '');
    const leading = game.leader === i && game.required;
    seat.innerHTML = `
      <span class="who">${i === HUMAN ? '你' : `模型 ${i}`}</span>
      <span class="cnt">${game.hands[i].length} 张</span>
      <span class="last">${log.filter((l) => l.seat === i).slice(-1)[0]?.text ?? '—'}</span>
      <span class="tag ${leading ? 'lead' : ''}">${leading ? '当前牌面' : ''}</span>`;
    box.appendChild(seat);
  }
}

function renderBoard() {
  const label = $('boardLabel');
  const cards = $('boardCards');
  cards.innerHTML = '';
  if (game.finished) {
    label.textContent = game.winner === HUMAN ? '你赢了 🎉' : `模型 ${game.winner} 先走完了`;
    return;
  }
  if (!game.required) {
    label.textContent = game.current === HUMAN ? '轮到你自由出牌（不能不要）' : '等对手出牌…';
  } else {
    label.textContent = `要压的牌：${E.KIND_NAMES_CN[game.required.kind]}`;
    for (const card of [...game.required.cards].sort((a, b) => a - b)) cards.appendChild(cardEl(card));
  }
}

function renderThinking(rows, title) {
  const box = $('thinkBox');
  box.innerHTML = '';
  if (!rows) return;
  const wrap = document.createElement('div');
  wrap.className = 'think';
  wrap.innerHTML = `<div class="t">${title}</div>`;
  for (const { move, score, prob } of rows) {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.innerHTML = `<span class="mv">${moveText(move)}</span>
      <span class="pc">${(prob * 100).toFixed(1)}%</span>
      <span class="g"><i style="width:${Math.max(prob * 100, 1)}%"></i></span>`;
    bar.title = `打分 ${score.toFixed(2)}`;
    wrap.appendChild(bar);
  }
  box.appendChild(wrap);
}

function render() {
  renderSeats();
  renderBoard();

  const hand = $('hand');
  hand.innerHTML = '';
  const myTurn = game.current === HUMAN && !game.finished;
  for (const card of game.hands[HUMAN]) hand.appendChild(cardEl(card, { selectable: myTurn }));

  $('playBtn').disabled = !myTurn || !selected.size;
  $('passBtn').disabled = !myTurn || !game.required;
  $('hintBtn').disabled = !myTurn;

  $('log').innerHTML = log.map((l) => `<div>${l.who}：${l.text}</div>`).join('');
}

function say(text, cls = '') {
  $('msg').textContent = text;
  $('msg').className = 'msg ' + cls;
}

// ------------------------------------------------------------------ 流程

function record(seat, text) {
  log.push({ seat, who: seat === HUMAN ? '你' : `模型 ${seat}`, text });
}

function apply(seat, move) {
  record(seat, moveText(move));
  game.step(move);
  selected.clear();
}

/** 对手依次出牌，直到轮回给人或者牌局结束。 */
async function runBots() {
  while (!game.finished && game.current !== HUMAN) {
    const obs = game.observe();
    const seat = obs.player;

    if (obs.legal.length === 1) {
      apply(seat, obs.legal[0]);
      render();
    } else {
      const rows = think(obs, 4);
      renderThinking(rows, `模型 ${seat} 在想：`);
      render();
      await sleep(650);
      apply(seat, rows[0].move);
      render();
    }
    await sleep(350);
  }
  if (game.finished) finish();
  else say('轮到你了。', '');
}

function finish() {
  renderThinking(null);
  const mine = game.hands[HUMAN].length;
  say(game.winner === HUMAN ? '你赢了！' : `模型 ${game.winner} 赢了，你还剩 ${mine} 张。`,
    game.winner === HUMAN ? 'win' : '');
  render();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function playSelected() {
  const obs = game.observe();
  const want = [...selected].sort((a, b) => a - b);
  const match = obs.legal.find((m) =>
    m && m.cards.length === want.length &&
    [...m.cards].sort((a, b) => a - b).every((c, i) => c === want[i]));

  if (!match) {
    const anyShape = obs.legal.some((m) => m && m.cards.length === want.length);
    say(game.required
      ? (anyShape ? '这手牌压不过当前的牌。' : '这不是一手合法的牌型。')
      : '这不是一手合法的牌型。', 'bad');
    return;
  }
  say('');
  renderThinking(null);
  apply(HUMAN, match);
  render();
  if (game.finished) finish(); else runBots();
}

function newGame() {
  game = new E.Game();
  selected.clear();
  log = [];
  renderThinking(null);
  say(game.current === HUMAN ? '你拿到 ♦3，先出，第一手必须包含 ♦3。'
    : '对手拿到 ♦3，先出。', '');
  render();
  if (game.current !== HUMAN) runBots();
}

// ------------------------------------------------------------------ 启动

$('playBtn').onclick = playSelected;
$('passBtn').onclick = () => {
  renderThinking(null);
  apply(HUMAN, null);
  render();
  runBots();
};
$('hintBtn').onclick = () => {
  renderThinking(think(game.observe(), 4), '模型建议你：');
};
$('clearBtn').onclick = () => { selected.clear(); say(''); render(); };
$('newBtn').onclick = newGame;

fetch('./model.json')
  .then((r) => r.json())
  .then((blob) => {
    model = blob;
    if (blob.feature_dim !== E.FEATURE_DIM) {
      say(`模型是 ${blob.feature_dim} 维特征训练的，前端是 ${E.FEATURE_DIM} 维，对不上。`, 'bad');
      return;
    }
    newGame();
  })
  .catch(() => say('模型加载失败，刷新试试。', 'bad'));
