#!/usr/bin/env node
// Solo-mode balance simulator for Meeple Melee.
//
// Solo rules modelled (Variant A):
//   - Player side: 8 meeples, full action + pick phases.
//   - Dice side: no actions; rolls a configurable number of meeples per battle and reveals them all.
//     Starting pool (8 + advantage) acts as the casualty/NZ absorber.
//   - Resolution is identical to the standard game (sweep-3 / kick-2 / punch-1, in that order).
//   - Game ends when either side hits 6 casualties.
//
// Run:
//   node sim.js                     # full sweep
//   node sim.js --n 20000           # override games per config
//   node sim.js --debug             # print one sample game

const SWEEP = 'sweep', KICK = 'kick', PUNCH = 'punch';
const ORIENTATIONS = [SWEEP, KICK, PUNCH];
const ACTIONS_TABLE = [1, 1, 2, 2, 3, 3];
const PICK_TABLE    = [6, 5, 5, 4, 4, 3];

// ------------------------------------------------------------
// Core rolling + resolution (ported from play/index.html)
// ------------------------------------------------------------
function rollMeeple() {
  const r = Math.random();
  if (r < 0.5) return SWEEP;
  if (r < 0.833) return KICK;
  return PUNCH;
}
function rollMeeples(count) {
  const n = Math.max(0, count | 0);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = rollMeeple();
  return sortMeeples(out);
}
function sortMeeples(meeples) {
  const order = { sweep: 0, kick: 1, punch: 2 };
  return [...meeples].sort((a, b) => order[a] - order[b]);
}
function countOrientations(picks) {
  const c = { sweep: 0, kick: 0, punch: 0 };
  for (const o of picks) c[o]++;
  return c;
}
function resolveBattle(p1picks, p2picks) {
  const c1 = countOrientations(p1picks);
  const c2 = countOrientations(p2picks);
  for (const [type, threshold] of [[SWEEP, 3], [KICK, 2], [PUNCH, 1]]) {
    const diff = c1[type] - c2[type];
    if (Math.abs(diff) >= threshold) return { winner: diff > 0 ? 'p1' : 'p2' };
  }
  return { winner: null };
}

// ------------------------------------------------------------
// AI action + pick heuristics (ported from play/index.html)
// ------------------------------------------------------------
function aiTakeActions(meeples, actionsLeft) {
  let m = [...meeples];
  for (let a = 0; a < actionsLeft; a++) {
    const c = countOrientations(m);
    if (c.punch === 0 && c.sweep > 0) {
      const idx = m.indexOf(SWEEP);
      if (idx >= 0) m[idx] = PUNCH;
    } else if (c.punch > 0 && c.punch < m.length) {
      const nonPunch = m.length - c.punch;
      if (nonPunch >= 3) {
        for (let i = 0; i < m.length; i++) if (m[i] !== PUNCH) m[i] = rollMeeple();
      } else {
        const idx = m.findIndex(x => x !== PUNCH);
        if (idx >= 0) m[idx] = PUNCH;
      }
    } else if (c.kick >= 2) {
      const idx = m.findIndex(x => x === SWEEP);
      if (idx >= 0) m[idx] = KICK;
    } else {
      if (c.sweep >= 4) {
        const idx = m.findIndex(x => x !== SWEEP);
        if (idx >= 0) m[idx] = SWEEP;
      } else {
        for (let i = 0; i < m.length; i++) m[i] = rollMeeple();
      }
    }
  }
  return sortMeeples(m);
}

function combinations(n, k) {
  const result = [], combo = [];
  function gen(start) {
    if (combo.length === k) { result.push([...combo]); return; }
    for (let i = start; i < n; i++) { combo.push(i); gen(i + 1); combo.pop(); }
  }
  gen(0);
  return result;
}
function evaluatePicks(picks) {
  const c = countOrientations(picks);
  let score = c.punch * 6 + c.kick * 2.5 + c.sweep * 0.8;
  if (c.punch >= 2) score += 10;
  if (c.kick >= 3) score += 8;
  if (c.sweep >= 4) score += 6;
  return score;
}
function aiBestPick(meeples, pickCount) {
  const combos = combinations(meeples.length, pickCount);
  let bestScore = -Infinity, bestCombo = combos[0];
  for (const combo of combos) {
    const score = evaluatePicks(combo.map(i => meeples[i]));
    if (score > bestScore) { bestScore = score; bestCombo = combo; }
  }
  return bestCombo;
}
function aiPickMeeples(meeples, pickCount) {
  if (pickCount >= meeples.length) return [...Array(meeples.length).keys()];
  return aiBestPick(meeples, pickCount);
}

// ------------------------------------------------------------
// Solo-mode game loop
// ------------------------------------------------------------
//   playerStart    — meeples the player starts with (default 8)
//   diceStart      — meeples the dice opponent starts with (8 + advantage)
//   playerMode     — 'ai' | 'baseline' (baseline = no actions, random pick)
//   diceRollCount  — 'pool'  : roll all remaining meeples (strictly physical "+1 advantage")
//                    'pick'  : roll pickCount meeples (matches player's reveal count)
//                    'pick+N': roll pickCount + N meeples (bonus dice each battle)
// Dice side reveals every meeple it rolled.
function simulateGame({ playerStart = 8, diceStart = 9, playerMode = 'ai', diceRollCount = 'pool', debug = false }) {
  const player = { casualties: 0, nz: 0, start: playerStart };
  const dice   = { casualties: 0, nz: 0, start: diceStart };
  let battles = 0;
  let actionsUsed = 0;
  let actionsAvailable = 0;

  const remaining = p => p.start - p.casualties - p.nz;

  while (player.casualties < 6 && dice.casualties < 6) {
    if (remaining(player) <= 0) { player.casualties = 6; break; }
    if (remaining(dice)   <= 0) { dice.casualties   = 6; break; }
    battles++;

    const maxCasPreview = Math.max(player.casualties, dice.casualties);
    const pickCountPreview = maxCasPreview >= 6 ? 0 : PICK_TABLE[maxCasPreview];

    let diceRoll;
    if (diceRollCount === 'pool') {
      diceRoll = remaining(dice);
    } else if (diceRollCount === 'pick') {
      diceRoll = Math.min(remaining(dice), pickCountPreview);
    } else if (typeof diceRollCount === 'string' && diceRollCount.startsWith('pick+')) {
      const bonus = parseInt(diceRollCount.slice(5), 10);
      diceRoll = Math.min(remaining(dice), pickCountPreview + bonus);
    } else {
      throw new Error('bad diceRollCount ' + diceRollCount);
    }

    const playerMeeples = rollMeeples(remaining(player));
    const diceMeeples   = rollMeeples(diceRoll);

    // Player: actions + pick
    const playerActions = player.casualties >= 6 ? 0 : ACTIONS_TABLE[player.casualties];
    actionsAvailable += playerActions;
    let pm = playerMeeples;
    if (playerMode === 'ai') {
      const before = pm.slice();
      pm = aiTakeActions(pm, playerActions);
      if (JSON.stringify(before) !== JSON.stringify(pm)) actionsUsed += playerActions;
    }
    // pickCount uses MAX casualties of both sides
    const maxCas = Math.max(player.casualties, dice.casualties);
    const pickCount = maxCas >= 6 ? 0 : PICK_TABLE[maxCas];
    const playerPickCount = Math.min(pickCount, pm.length);
    let playerPicks;
    if (playerMode === 'ai') {
      const idxs = aiPickMeeples(pm, playerPickCount);
      playerPicks = idxs.map(i => pm[i]);
    } else {
      // baseline: random subset
      const shuffled = [...pm].sort(() => Math.random() - 0.5);
      playerPicks = shuffled.slice(0, playerPickCount);
    }

    // Dice side: no actions, reveal everything rolled
    const dicePicks = diceMeeples;

    const result = resolveBattle(playerPicks, dicePicks);
    // map p1->player, p2->dice in our resolveBattle call
    const winner = result.winner === 'p1' ? 'player' : result.winner === 'p2' ? 'dice' : null;

    if (debug) {
      console.log(`Battle ${battles}: player=${JSON.stringify(playerPicks)} dice=${JSON.stringify(dicePicks)} -> ${winner || 'draw'}`);
    }

    if (winner === 'player') {
      dice.casualties += 1 + dice.nz;
      dice.nz = 0;
      player.nz = 0;
    } else if (winner === 'dice') {
      player.casualties += 1 + player.nz;
      player.nz = 0;
      dice.nz = 0;
    } else {
      player.nz++;
      dice.nz++;
      if (remaining(player) <= 0) player.casualties = 6;
      if (remaining(dice)   <= 0) dice.casualties   = 6;
    }
    if (player.casualties >= 6) player.casualties = 6;
    if (dice.casualties   >= 6) dice.casualties   = 6;

    // safety
    if (battles > 200) break;
  }

  const winner = player.casualties >= 6 && dice.casualties >= 6
    ? 'mutual'
    : player.casualties >= 6 ? 'dice' : 'player';
  return { winner, battles, playerCas: player.casualties, diceCas: dice.casualties,
           actionsUsed, actionsAvailable };
}

// ------------------------------------------------------------
// Runner
// ------------------------------------------------------------
function runBatch(config, n) {
  let pw = 0, dw = 0, draws = 0;
  let totalBattles = 0;
  let totalActionsUsed = 0, totalActionsAvail = 0;
  const battleHist = [];
  for (let i = 0; i < n; i++) {
    const r = simulateGame(config);
    if (r.winner === 'player') pw++;
    else if (r.winner === 'dice') dw++;
    else draws++;
    totalBattles += r.battles;
    totalActionsUsed += r.actionsUsed;
    totalActionsAvail += r.actionsAvailable;
    battleHist.push(r.battles);
  }
  battleHist.sort((a, b) => a - b);
  const median = battleHist[Math.floor(n / 2)];
  return {
    n,
    playerWinPct: (pw / n * 100).toFixed(1),
    diceWinPct:   (dw / n * 100).toFixed(1),
    drawPct:      (draws / n * 100).toFixed(1),
    avgBattles:   (totalBattles / n).toFixed(1),
    medianBattles: median,
    actionUseRate: totalActionsAvail ? (totalActionsUsed / totalActionsAvail * 100).toFixed(0) + '%' : 'n/a',
  };
}

function main() {
  const args = process.argv.slice(2);
  const N = Number((args.find(a => a.startsWith('--n=')) || '--n=10000').split('=')[1]);
  const debug = args.includes('--debug');

  if (debug) {
    console.log('=== Sample game (advantage +1, AI player) ===');
    simulateGame({ playerStart: 8, diceStart: 9, playerMode: 'ai', debug: true });
    return;
  }

  console.log(`Meeple Melee solo-mode balance sim  (N=${N} games per row)`);
  console.log(`Dice side: no actions, reveals every meeple rolled.`);
  console.log('');

  const modes = ['ai', 'baseline'];

  function section(label, rollCount, advRange) {
    console.log(`=== ${label}  (diceRollCount=${rollCount}) ===`);
    const header = 'adv  mode      player%  dice%   draw%   battles(avg/med)  action-use';
    console.log(header);
    console.log('-'.repeat(header.length));
    for (const adv of advRange) {
      for (const mode of modes) {
        const r = runBatch({ playerStart: 8, diceStart: 8 + adv, playerMode: mode, diceRollCount: rollCount }, N);
        const tag = mode === 'ai' ? 'AI      ' : 'baseline';
        const advStr = (adv >= 0 ? '+' : '') + adv;
        console.log(`${advStr.padStart(3)}  ${tag}  ${r.playerWinPct.padStart(6)}   ${r.diceWinPct.padStart(5)}   ${r.drawPct.padStart(5)}   ${r.avgBattles.padStart(5)} / ${String(r.medianBattles).padStart(3)}        ${r.actionUseRate}`);
      }
      console.log('');
    }
  }

  section('A  — dice rolls entire pool (literal +N reading)', 'pool',   [-4, -2, 0, 1, 2, 4]);
  section('B  — dice rolls pickCount (matches player reveal)', 'pick',  [0, 1, 2, 3, 4]);
  section('C  — dice rolls pickCount + 1 (one extra die)',      'pick+1',[0, 1, 2, 3]);
  section('D  — dice rolls pickCount + 2 (two extra dice)',     'pick+2',[0, 1, 2, 3]);
}

main();
