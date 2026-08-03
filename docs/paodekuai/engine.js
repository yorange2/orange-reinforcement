// 跑得快引擎 + 特征 + 打分网络，从 Python 版逐函数移植过来。
//
// 移植是有风险的：合法牌的枚举顺序、特征的每一维、拆解估计的贪心步骤，
// 差一点模型的判断就变了。所以 tools/parity_check.mjs 会拿 Python 导出的
// 上千个真实局面来核对合法动作、特征向量和打分，三者必须逐一对上。
//
// 牌用整数表示：card = rank * 4 + suit，这样直接比大小就等于按 (点数, 花色) 排序。

export const MIN_RANK = 3;
export const MAX_RANK = 14;
export const RANKS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
export const RANK_NAMES = {
  3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '10',
  11: 'J', 12: 'Q', 13: 'K', 14: 'A',
};
export const SUITS = ['♦', '♣', '♥', '♠'];
export const DIAMOND_THREE = MIN_RANK * 4 + 0;

export const rankOf = (card) => Math.floor(card / 4);
export const suitOf = (card) => card % 4;
export const cardName = (card) => SUITS[suitOf(card)] + RANK_NAMES[rankOf(card)];
export const handToStr = (cards) => [...cards].sort((a, b) => a - b).map(cardName).join(' ');

// ---------------------------------------------------------------- 牌型

export const SINGLE = 'single', PAIR = 'pair', TRIPLE = 'triple';
export const TRIPLE_ONE = 'triple_one', TRIPLE_TWO = 'triple_two';
export const STRAIGHT = 'straight', PAIR_STRAIGHT = 'pair_straight';
export const PLANE = 'plane', PLANE_ONE = 'plane_one', PLANE_TWO = 'plane_two';
export const BOMB = 'bomb';

export const KINDS = [SINGLE, PAIR, TRIPLE, TRIPLE_ONE, TRIPLE_TWO,
  STRAIGHT, PAIR_STRAIGHT, PLANE, PLANE_ONE, PLANE_TWO, BOMB];
export const KIND_INDEX = Object.fromEntries(KINDS.map((k, i) => [k, i]));
export const KIND_NAMES_CN = {
  single: '单张', pair: '对子', triple: '三张', triple_one: '三带一',
  triple_two: '三带二', straight: '顺子', pair_straight: '连对',
  plane: '飞机', plane_one: '飞机带单', plane_two: '飞机带对', bomb: '炸弹',
};

const MIN_STRAIGHT = 5, MIN_PAIR_STRAIGHT = 2, MIN_PLANE = 2, MAX_ATTACH_SETS = 6;

const combo = (kind, rank, length, cards) => ({ kind, rank, length, cards });

export function beats(candidate, current) {
  if (!current) return true;
  if (candidate.kind === BOMB && current.kind !== BOMB) return true;
  if (current.kind === BOMB && candidate.kind !== BOMB) return false;
  if (candidate.kind !== current.kind || candidate.length !== current.length) return false;
  return candidate.rank > current.rank;
}

/** 按点数分组，组内按花色升序（所以 3 的第一张一定是 ♦3）。 */
export function groupByRank(hand) {
  const groups = new Map();
  for (const card of [...hand].sort((a, b) => a - b)) {
    const rank = rankOf(card);
    if (!groups.has(rank)) groups.set(rank, []);
    groups.get(rank).push(card);
  }
  return groups;
}

/** 所有长度 >= minLength 的连续片段（含所有子片段）。 */
function runs(ranks, minLength) {
  const ordered = [...new Set(ranks)].sort((a, b) => a - b);
  const out = [];
  for (let i = 0; i < ordered.length; i++) {
    const run = [ordered[i]];
    if (run.length >= minLength) out.push([...run]);
    for (let j = i + 1; j < ordered.length; j++) {
      if (ordered[j] !== run[run.length - 1] + 1) break;
      run.push(ordered[j]);
      if (run.length >= minLength) out.push([...run]);
    }
  }
  return out;
}

const sortedRanks = (groups) => [...groups.keys()].sort((a, b) => a - b);

function straights(groups) {
  return runs(sortedRanks(groups), MIN_STRAIGHT).map((run) =>
    combo(STRAIGHT, Math.max(...run), run.length, run.map((r) => groups.get(r)[0])));
}

function pairStraights(groups) {
  const pairRanks = sortedRanks(groups).filter((r) => groups.get(r).length >= 2);
  return runs(pairRanks, MIN_PAIR_STRAIGHT).map((run) =>
    combo(PAIR_STRAIGHT, Math.max(...run), run.length,
      run.flatMap((r) => groups.get(r).slice(0, 2))));
}

/**
 * 挑 count 个点数、每个出 size 张作为带牌。
 * 候选按"拆得起"排序（先用零散单张，再用小牌）；飞机带牌最多生成 MAX_ATTACH_SETS 种。
 */
function attachmentSets(groups, usedRanks, count, size) {
  const candidates = sortedRanks(groups).filter(
    (r) => !usedRanks.has(r) && groups.get(r).length >= size);
  if (candidates.length < count) return [];

  candidates.sort((a, b) => (groups.get(a).length - groups.get(b).length) || (a - b));

  let chosen;
  if (count === 1) {
    chosen = [...candidates].sort((a, b) => a - b).map((r) => [r]);
  } else {
    const pool = candidates.slice(0, Math.max(count, Math.min(candidates.length, count + 2)));
    pool.sort((a, b) => a - b);
    chosen = combinations(pool, count).slice(0, MAX_ATTACH_SETS);
  }
  return chosen.map((ranks) => ranks.flatMap((r) => groups.get(r).slice(0, size)));
}

function combinations(items, k) {
  if (k === 0) return [[]];
  const out = [];
  for (let i = 0; i <= items.length - k; i++) {
    for (const rest of combinations(items.slice(i + 1), k - 1)) out.push([items[i], ...rest]);
  }
  return out;
}

function planes(groups) {
  const tripleRanks = sortedRanks(groups).filter((r) => groups.get(r).length >= 3);
  const out = [];
  for (const run of runs(tripleRanks, MIN_PLANE)) {
    const k = run.length;
    const body = run.flatMap((r) => groups.get(r).slice(0, 3));
    const used = new Set(run);
    const top = Math.max(...run);
    out.push(combo(PLANE, top, k, body));
    for (const a of attachmentSets(groups, used, k, 1)) out.push(combo(PLANE_ONE, top, k, [...body, ...a]));
    for (const a of attachmentSets(groups, used, k, 2)) out.push(combo(PLANE_TWO, top, k, [...body, ...a]));
  }
  return out;
}

function triplesWithAttachments(groups) {
  const out = [];
  for (const rank of sortedRanks(groups)) {
    const cards = groups.get(rank);
    if (cards.length < 3) continue;
    const body = cards.slice(0, 3);
    const used = new Set([rank]);
    for (const a of attachmentSets(groups, used, 1, 1)) out.push(combo(TRIPLE_ONE, rank, 1, [...body, ...a]));
    for (const a of attachmentSets(groups, used, 1, 2)) out.push(combo(TRIPLE_TWO, rank, 1, [...body, ...a]));
  }
  return out;
}

export function allCombos(hand) {
  const groups = groupByRank(hand);
  const out = [];
  for (const rank of sortedRanks(groups)) {
    const cards = groups.get(rank);
    out.push(combo(SINGLE, rank, 1, cards.slice(0, 1)));
    if (cards.length >= 2) out.push(combo(PAIR, rank, 1, cards.slice(0, 2)));
    if (cards.length >= 3) out.push(combo(TRIPLE, rank, 1, cards.slice(0, 3)));
    if (cards.length >= 4) out.push(combo(BOMB, rank, 1, cards.slice(0, 4)));
  }
  out.push(...straights(groups), ...pairStraights(groups), ...planes(groups),
    ...triplesWithAttachments(groups));
  return out;
}

function combosOfKind(groups, kind, length) {
  const ranks = sortedRanks(groups);
  if (kind === SINGLE) return ranks.map((r) => combo(SINGLE, r, 1, groups.get(r).slice(0, 1)));
  if (kind === PAIR) return ranks.filter((r) => groups.get(r).length >= 2)
    .map((r) => combo(PAIR, r, 1, groups.get(r).slice(0, 2)));
  if (kind === TRIPLE) return ranks.filter((r) => groups.get(r).length >= 3)
    .map((r) => combo(TRIPLE, r, 1, groups.get(r).slice(0, 3)));
  if (kind === TRIPLE_ONE || kind === TRIPLE_TWO)
    return triplesWithAttachments(groups).filter((c) => c.kind === kind);
  if (kind === STRAIGHT) return straights(groups).filter((c) => c.length === length);
  if (kind === PAIR_STRAIGHT) return pairStraights(groups).filter((c) => c.length === length);
  if (kind === PLANE || kind === PLANE_ONE || kind === PLANE_TWO)
    return planes(groups).filter((c) => c.kind === kind && c.length === length);
  return [];
}

export function legalMoves(hand, required) {
  if (!required) return allCombos(hand);

  const groups = groupByRank(hand);
  const bombs = sortedRanks(groups).filter((r) => groups.get(r).length >= 4);

  if (required.kind === BOMB) {
    return bombs.filter((r) => r > required.rank)
      .map((r) => combo(BOMB, r, 1, groups.get(r).slice(0, 4)));
  }

  const moves = combosOfKind(groups, required.kind, required.length)
    .filter((c) => c.rank > required.rank);
  moves.push(...bombs.map((r) => combo(BOMB, r, 1, groups.get(r).slice(0, 4))));
  return moves;
}

/** 贪心估计"打完这手牌还要出几轮"，越小越好。 */
export function estimateTurns(hand) {
  if (!hand.length) return 0;
  const counts = new Map();
  for (const card of hand) counts.set(rankOf(card), (counts.get(rankOf(card)) || 0) + 1);

  let turns = 0;
  for (const [rank, n] of [...counts]) if (n === 4) { counts.delete(rank); turns += 1; }

  const take = (minLen, per) => {
    for (;;) {
      const pool = [...counts.keys()].filter((r) => counts.get(r) >= per).sort((a, b) => a - b);
      const found = runs(pool, minLen);
      if (!found.length) return;
      let best = found[0];
      for (const run of found) if (run.length > best.length) best = run;
      for (const rank of best) {
        counts.set(rank, counts.get(rank) - per);
        if (counts.get(rank) === 0) counts.delete(rank);
      }
      turns += 1;
    }
  };
  take(MIN_STRAIGHT, 1);
  take(MIN_PAIR_STRAIGHT, 2);

  const values = [...counts.values()];
  const triples = values.filter((v) => v === 3).length;
  const pairs = values.filter((v) => v === 2).length;
  const singles = values.filter((v) => v === 1).length;

  turns += triples + pairs + singles - Math.min(triples, singles);
  return Math.max(turns, 1);
}

export function attachmentRanks(move) {
  let body;
  if (move.kind === TRIPLE_ONE || move.kind === TRIPLE_TWO) body = new Set([move.rank]);
  else if (move.kind === PLANE_ONE || move.kind === PLANE_TWO) {
    body = new Set();
    for (let r = move.rank - move.length + 1; r <= move.rank; r++) body.add(r);
  } else return [];
  return [...new Set(move.cards.map(rankOf).filter((r) => !body.has(r)))].sort((a, b) => a - b);
}

function bodySize(move) {
  if ([PLANE, PLANE_ONE, PLANE_TWO].includes(move.kind)) return 3 * move.length;
  if ([TRIPLE, TRIPLE_ONE, TRIPLE_TWO].includes(move.kind)) return 3;
  if (move.kind === PAIR_STRAIGHT) return 2 * move.length;
  if (move.kind === STRAIGHT) return move.length;
  return move.cards.length;
}

// ---------------------------------------------------------------- 特征

export const STATE_OFFSET = 28;
export const FEATURE_DIM = 42;

function stateFeatures(obs, turnsNow) {
  const n = obs.handSizes.length;
  const oppHands = obs.handSizes.filter((_, i) => i !== obs.player);
  const minOpp = Math.min(...oppHands);
  const req = obs.required;
  return [
    req ? 0 : 1,
    obs.hand.length / 16,
    minOpp / 16,
    minOpp <= 2 ? 1 : 0,
    minOpp <= 1 ? 1 : 0,
    turnsNow / 10,
    obs.handSizes[(obs.player + 1) % n] / 16,
    obs.handSizes[(obs.player - 1 + n) % n] / 16,
    req ? (req.rank - 3) / 11 : 0,
    req ? req.cards.length / 5 : 0,
    obs.legal.length / 40,
    Math.min(obs.trick, 60) / 60,
    obs.leader === obs.player ? 1 : 0,
    1,
  ];
}

/** 对手手里还可能有哪些牌：整副 - 已出的 - 自己手上的。 */
export function unseenCounts(obs) {
  const unseen = new Map(RANKS.map((r) => [r, 4]));
  for (const [rank, count] of obs.playedCounts) unseen.set(rank, unseen.get(rank) - count);
  for (const card of obs.hand) unseen.set(rankOf(card), unseen.get(rankOf(card)) - 1);
  return unseen;
}

export function batchFeatures(obs) {
  const turnsNow = estimateTurns(obs.hand);
  const tail = stateFeatures(obs, turnsNow);
  const unseen = unseenCounts(obs);
  let totalUnseen = 0;
  for (const v of unseen.values()) totalUnseen += v;
  totalUnseen = Math.max(totalUnseen, 1);

  const counts = new Map();
  for (const card of obs.hand) counts.set(rankOf(card), (counts.get(rankOf(card)) || 0) + 1);
  const handSize = obs.hand.length;

  return obs.legal.map((move) => {
    const onehot = new Array(KINDS.length).fill(0);
    let head;

    if (!move) {
      head = [1, 0, 0, 0, 0, handSize / 16, 0, turnsNow / 10, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    } else {
      onehot[KIND_INDEX[move.kind]] = 1;
      const played = move.cards.length;
      const handAfter = handSize - played;

      const rest = [...obs.hand];
      for (const card of move.cards) rest.splice(rest.indexOf(card), 1);
      const turnsAfter = estimateTurns(rest);

      const moveCounts = new Map();
      for (const card of move.cards) moveCounts.set(rankOf(card), (moveCounts.get(rankOf(card)) || 0) + 1);
      let breaksPair = 0, breaksTriple = 0, breaksBomb = 0;
      for (const [rank, used] of moveCounts) {
        const have = counts.get(rank);
        if (used < have) {
          if (have === 4) breaksBomb += 1;
          else if (have === 3) breaksTriple += 1;
          else if (have === 2) breaksPair += 1;
        }
      }

      let higher = 0;
      for (const [rank, count] of unseen) if (rank > move.rank) higher += count;

      const attach = attachmentRanks(move);
      head = [
        0,
        move.kind === BOMB ? 1 : 0,
        (move.rank - 3) / 11,
        played / 5,
        move.length / 6,
        handAfter / 16,
        handAfter === 0 ? 1 : 0,
        turnsAfter / 10,
        (turnsNow - turnsAfter) / 3,
        breaksPair,
        breaksTriple,
        breaksBomb,
        (played - bodySize(move)) / 4,
        higher / totalUnseen,
        move.rank === 14 ? 1 : 0,
        attach.length ? (Math.max(...attach) - 3) / 11 : 0,
        attach.length ? (Math.min(...attach) - 3) / 11 : 0,
      ];
    }
    return [...onehot, ...head, ...tail];
  });
}

// ---------------------------------------------------------------- 网络

/** 按导出的层描述顺序走一遍，返回这一手牌的分数。 */
export function score(net, x) {
  for (const layer of net) {
    if (layer.type === 'linear') {
      const out = new Array(layer.out);
      for (let i = 0; i < layer.out; i++) {
        const row = layer.w[i];
        let sum = layer.b[i];
        for (let j = 0; j < row.length; j++) sum += row[j] * x[j];
        out[i] = sum;
      }
      x = out;
    } else if (layer.type === 'layernorm') {
      const n = x.length;
      let mean = 0;
      for (const v of x) mean += v;
      mean /= n;
      let variance = 0;
      for (const v of x) variance += (v - mean) ** 2;
      variance /= n;
      const scale = Math.sqrt(variance + layer.eps);
      x = x.map((v, i) => ((v - mean) / scale) * layer.w[i] + layer.b[i]);
    } else if (layer.type === 'relu') {
      x = x.map((v) => (v > 0 ? v : 0));
    }
  }
  return x[0];
}

export function softmax(scores) {
  const top = Math.max(...scores);
  const exp = scores.map((s) => Math.exp(s - top));
  const total = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / total);
}

// ---------------------------------------------------------------- 引擎

export function fullDeck() {
  const deck = [];
  for (const rank of RANKS) for (let suit = 0; suit < 4; suit++) deck.push(rank * 4 + suit);
  return deck;
}

/** 和 Python 的 random.Random 不是一套，只用于网页发牌。 */
export function shuffle(deck, rand = Math.random) {
  for (let i = deck.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [deck[i], deck[j]] = [deck[j], deck[i]];
  }
  return deck;
}

export class Game {
  constructor(rand = Math.random, nPlayers = 3, handSize = 16) {
    const deck = shuffle(fullDeck(), rand);
    this.hands = [];
    for (let i = 0; i < nPlayers; i++) {
      this.hands.push(deck.slice(i * handSize, (i + 1) * handSize).sort((a, b) => a - b));
    }
    this.nPlayers = nPlayers;
    this.required = null;
    this.leader = null;
    this.passes = 0;
    this.trick = 0;
    this.finished = false;
    this.winner = null;
    this.playedCounts = new Map();
    this.firstMove = true;
    this.current = this.hands.findIndex((hand) => hand.includes(DIAMOND_THREE));
  }

  legalActions(player = this.current) {
    let moves = legalMoves(this.hands[player], this.required);
    if (this.firstMove) moves = moves.filter((m) => m.cards.includes(DIAMOND_THREE));
    return this.required ? [...moves, null] : moves;
  }

  observe(player = this.current) {
    return {
      player,
      hand: [...this.hands[player]],
      handSizes: this.hands.map((h) => h.length),
      required: this.required,
      leader: this.leader,
      playedCounts: new Map(this.playedCounts),
      legal: this.legalActions(player),
      trick: this.trick,
    };
  }

  step(action) {
    if (this.finished) throw new Error('这一局已经结束了');

    if (!action) {
      if (!this.required) throw new Error('自由出牌时不能过');
      this.passes += 1;
    } else {
      const hand = this.hands[this.current];
      for (const card of action.cards) {
        hand.splice(hand.indexOf(card), 1);
        const rank = rankOf(card);
        this.playedCounts.set(rank, (this.playedCounts.get(rank) || 0) + 1);
      }
      this.required = action;
      this.leader = this.current;
      this.passes = 0;
      this.firstMove = false;
      if (!hand.length) {
        this.finished = true;
        this.winner = this.current;
        return;
      }
    }

    this.trick += 1;
    if (this.passes >= this.nPlayers - 1) {
      this.required = null;
      this.passes = 0;
      this.current = this.leader ?? this.current;
    } else {
      this.current = (this.current + 1) % this.nPlayers;
    }
  }
}
