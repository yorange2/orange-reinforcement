// 核对 docs/engine.js 的移植有没有走样。
//
// 拿 Python 导出的真实局面，用 JS 重算合法动作、特征和打分，三者必须逐一对上。
// 用法：node tools/parity_check.mjs

import { readFileSync } from 'node:fs';
import * as E from '../docs/engine.js';

const cases = JSON.parse(readFileSync(new URL('./parity_cases.json', import.meta.url)));
const model = JSON.parse(readFileSync(new URL('../docs/model.json', import.meta.url)));

let worstFeature = 0, worstScore = 0;
const failures = [];

function fail(index, what, detail) {
  if (failures.length < 6) failures.push(`  局面 #${index} ${what}: ${detail}`);
}

cases.forEach((c, index) => {
  // 从 Python 给的状态还原一个观测；合法动作由 JS 自己算
  const required = c.required
    ? { kind: c.required_kind, rank: c.required_rank, length: c.required_length, cards: c.required }
    : null;

  const obs = {
    player: c.player,
    hand: c.hand,
    handSizes: c.hand_sizes,
    required,
    leader: c.leader,
    playedCounts: new Map(c.played_counts),
    trick: c.trick,
    legal: [],
  };
  let moves = E.legalMoves(obs.hand, required);
  // 首手必须包含 ♦3，这条约束在引擎里，重放局面时要一并还原
  if (c.first_move) moves = moves.filter((m) => m.cards.includes(E.DIAMOND_THREE));
  obs.legal = required ? [...moves, null] : moves;

  // 1) 合法动作：数量、顺序、每手的牌都要一致
  if (obs.legal.length !== c.legal.length) {
    fail(index, '合法动作数量', `JS ${obs.legal.length} vs Python ${c.legal.length}`);
    return;
  }
  for (let i = 0; i < obs.legal.length; i++) {
    const mine = obs.legal[i] ? [...obs.legal[i].cards].sort((a, b) => a - b) : null;
    const theirs = c.legal[i] ? [...c.legal[i]].sort((a, b) => a - b) : null;
    if (JSON.stringify(mine) !== JSON.stringify(theirs)) {
      fail(index, `第 ${i} 个动作`, `JS ${JSON.stringify(mine)} vs Python ${JSON.stringify(theirs)}`);
      return;
    }
  }

  // 2) 特征矩阵
  const features = E.batchFeatures(obs);
  for (let i = 0; i < features.length; i++) {
    if (features[i].length !== model.feature_dim) {
      fail(index, '特征维度', `${features[i].length} != ${model.feature_dim}`);
      return;
    }
    for (let j = 0; j < features[i].length; j++) {
      const gap = Math.abs(features[i][j] - c.features[i][j]);
      if (gap > worstFeature) worstFeature = gap;
      if (gap > 1e-4) {
        fail(index, `特征 [${i}][${j}] (${model.feature_names[j]})`,
          `JS ${features[i][j]} vs Python ${c.features[i][j]}`);
        return;
      }
    }
  }

  // 3) 网络打分
  for (let i = 0; i < features.length; i++) {
    const gap = Math.abs(E.score(model.net, features[i]) - c.scores[i]);
    if (gap > worstScore) worstScore = gap;
    if (gap > 2e-3) {
      fail(index, `打分 [${i}]`, `差 ${gap.toFixed(5)}`);
      return;
    }
  }
});

const moves = cases.reduce((n, c) => n + c.legal.length, 0);
console.log(`核对了 ${cases.length} 个局面、${moves} 个候选动作`);
console.log(`  特征最大偏差 ${worstFeature.toExponential(2)}`);
console.log(`  打分最大偏差 ${worstScore.toExponential(2)}（权重导出时保留 6 位小数）`);

if (failures.length) {
  console.error(`\n✗ 有 ${failures.length}+ 处对不上：`);
  failures.forEach((line) => console.error(line));
  process.exit(1);
}
console.log('\n✓ 合法动作、特征、打分三项全部一致');
