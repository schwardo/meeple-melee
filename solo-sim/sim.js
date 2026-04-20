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
// DICE-AWARE AI
//   Uses exact multinomial probabilities against the known dice
//   distribution (3/6 sweep, 2/6 kick, 1/6 punch) to pick optimally
//   and to choose actions via 1-step greedy lookahead.
// ------------------------------------------------------------
const DICE_P = [0.5, 1/3, 1/6]; // sweep, kick, punch

const logFactCache = [0];
function logFact(n) {
  while (logFactCache.length <= n) {
    logFactCache.push(logFactCache[logFactCache.length - 1] + Math.log(logFactCache.length));
  }
  return logFactCache[n];
}
function multinomialProb(counts, probs, n) {
  let lp = logFact(n);
  for (let i = 0; i < counts.length; i++) {
    lp -= logFact(counts[i]);
    if (counts[i] > 0) lp += counts[i] * Math.log(probs[i]);
  }
  return Math.exp(lp);
}

// Precompute P(ds, dk, dp) for dice rolling N — flat list of [ds, dk, dp, prob].
const diceDistCache = new Map();
function diceDistribution(N) {
  if (diceDistCache.has(N)) return diceDistCache.get(N);
  const out = [];
  for (let ds = 0; ds <= N; ds++) {
    for (let dk = 0; dk <= N - ds; dk++) {
      const dp = N - ds - dk;
      const prob = multinomialProb([ds, dk, dp], DICE_P, N);
      if (prob > 1e-12) out.push([ds, dk, dp, prob]);
    }
  }
  diceDistCache.set(N, out);
  return out;
}

// P(player wins) when player reveals (ps,pk,pp) vs dice of size diceN.
// Resolution order: sweep (diff>=3), kick (diff>=2), punch (diff>=1).
const winProbCache = new Map();
function winProb(ps, pk, pp, diceN) {
  const key = ps * 100000 + pk * 1000 + pp * 100 + diceN;
  const hit = winProbCache.get(key);
  if (hit !== undefined) return hit;
  const dist = diceDistribution(diceN);
  let p = 0;
  for (let i = 0; i < dist.length; i++) {
    const ds = dist[i][0], dk = dist[i][1], dp = dist[i][2], prob = dist[i][3];
    const sDiff = ps - ds;
    if (Math.abs(sDiff) >= 3) { if (sDiff > 0) p += prob; continue; }
    const kDiff = pk - dk;
    if (Math.abs(kDiff) >= 2) { if (kDiff > 0) p += prob; continue; }
    const pDiff = pp - dp;
    if (Math.abs(pDiff) >= 1) { if (pDiff > 0) p += prob; /* else nothing */ }
  }
  winProbCache.set(key, p);
  return p;
}

// Best pick (sp,kp,pp) given meeples multiset (s,k,p) and pickCount; returns its win prob.
const bestPickCache = new Map();
function bestPickWinProb(s, k, p, pickCount, diceN) {
  const n = s + k + p;
  const pc = Math.min(pickCount, n);
  const key = `${s},${k},${p},${pc},${diceN}`;
  const hit = bestPickCache.get(key);
  if (hit !== undefined) return hit;
  let best = -1;
  for (let sp = 0; sp <= Math.min(s, pc); sp++) {
    for (let kp = 0; kp <= Math.min(k, pc - sp); kp++) {
      const pp = pc - sp - kp;
      if (pp < 0 || pp > p) continue;
      const wp = winProb(sp, kp, pp, diceN);
      if (wp > best) best = wp;
    }
  }
  bestPickCache.set(key, best);
  return best;
}

function bestPickCounts(s, k, p, pickCount, diceN) {
  const pc = Math.min(pickCount, s + k + p);
  let best = -1, bestT = [0, 0, pc];
  for (let sp = 0; sp <= Math.min(s, pc); sp++) {
    for (let kp = 0; kp <= Math.min(k, pc - sp); kp++) {
      const pp = pc - sp - kp;
      if (pp < 0 || pp > p) continue;
      const wp = winProb(sp, kp, pp, diceN);
      if (wp > best) { best = wp; bestT = [sp, kp, pp]; }
    }
  }
  return bestT;
}

// Expand meeples array to (s,k,p) counts.
function toCounts(m) {
  let s = 0, k = 0, p = 0;
  for (const x of m) { if (x === SWEEP) s++; else if (x === KICK) k++; else p++; }
  return [s, k, p];
}
function fromCounts(s, k, p) {
  const out = [];
  for (let i = 0; i < s; i++) out.push(SWEEP);
  for (let i = 0; i < k; i++) out.push(KICK);
  for (let i = 0; i < p; i++) out.push(PUNCH);
  return out;
}

// Expected best-pick win prob after rerolling (rs, rk, rp) of sweep/kick/punch meeples.
// Remaining kept: (s-rs, k-rk, p-rp); rerolled count = nr; new multinomial(nr; 1/2,1/3,1/6).
function expectedWinProbAfterReroll(s, k, p, rs, rk, rp, pickCount, diceN) {
  const nr = rs + rk + rp;
  if (nr === 0) return bestPickWinProb(s, k, p, pickCount, diceN);
  const keptS = s - rs, keptK = k - rk, keptP = p - rp;
  let ev = 0;
  for (let ns = 0; ns <= nr; ns++) {
    for (let nk = 0; nk <= nr - ns; nk++) {
      const np = nr - ns - nk;
      const prob = multinomialProb([ns, nk, np], DICE_P, nr);
      const wp = bestPickWinProb(keptS + ns, keptK + nk, keptP + np, pickCount, diceN);
      ev += prob * wp;
    }
  }
  return ev;
}

// Greedy 1-step action selector: at each step, try all reposition & reroll candidates,
// pick the one with highest expected best-pick win prob; allow skipping remaining actions.
function aiTakeActionsDiceAware(meeples, actionsLeft, pickCount, diceN) {
  let [s, k, p] = toCounts(meeples);
  const types = [['s', 'k', 'p'], [SWEEP, KICK, PUNCH]];
  for (let a = 0; a < actionsLeft; a++) {
    const base = bestPickWinProb(s, k, p, pickCount, diceN);
    let bestEv = base;     // baseline = do nothing (skip)
    let bestState = [s, k, p];
    // Reposition: convert one meeple of type X to type Y.
    const cur = [s, k, p];
    for (let from = 0; from < 3; from++) {
      if (cur[from] === 0) continue;
      for (let to = 0; to < 3; to++) {
        if (from === to) continue;
        const nc = cur.slice();
        nc[from]--;
        nc[to]++;
        const wp = bestPickWinProb(nc[0], nc[1], nc[2], pickCount, diceN);
        if (wp > bestEv + 1e-12) { bestEv = wp; bestState = nc; }
      }
    }
    // Reroll: choose how many of each type to reroll.
    for (let rs = 0; rs <= s; rs++) {
      for (let rk = 0; rk <= k; rk++) {
        for (let rp2 = 0; rp2 <= p; rp2++) {
          if (rs + rk + rp2 === 0) continue;
          const ev = expectedWinProbAfterReroll(s, k, p, rs, rk, rp2, pickCount, diceN);
          if (ev > bestEv + 1e-12) {
            bestEv = ev;
            // sample realized outcome for state continuation
            // (we commit to the action but need concrete meeples for next iteration)
            bestState = sampleRerollOutcome(s, k, p, rs, rk, rp2);
          }
        }
      }
    }
    if (bestState[0] === s && bestState[1] === k && bestState[2] === p) break; // skip
    s = bestState[0]; k = bestState[1]; p = bestState[2];
  }
  return { meeples: sortMeeples(fromCounts(s, k, p)), counts: [s, k, p] };
}

function sampleRerollOutcome(s, k, p, rs, rk, rp) {
  let nS = s - rs, nK = k - rk, nP = p - rp;
  const nr = rs + rk + rp;
  for (let i = 0; i < nr; i++) {
    const r = Math.random();
    if (r < 0.5) nS++;
    else if (r < 0.833) nK++;
    else nP++;
  }
  return [nS, nK, nP];
}

// ------------------------------------------------------------
// Solo-mode game loop
// ------------------------------------------------------------
//   playerStart       — meeples the player starts with (default 8)
//   diceStart         — meeples the dice opponent starts with (8 + advantage)
//   playerMode        — 'ai' | 'ai-dice' | 'baseline'
//   diceRollCount     — 'pool'  : roll all remaining meeples
//                       'pick'  : roll pickCount meeples (matches player's reveal count)
//                       'pick+N': roll pickCount + N meeples
//   playerPickBonus   — adjust player's pickCount (e.g. -1 = player reveals one fewer)
// Dice side reveals every meeple it rolled.
function simulateGame({ playerStart = 8, diceStart = 9, playerMode = 'ai', diceRollCount = 'pool', playerPickBonus = 0, debug = false }) {
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
    // pickCount uses MAX casualties of both sides
    const maxCas = Math.max(player.casualties, dice.casualties);
    const pickCount = maxCas >= 6 ? 0 : PICK_TABLE[maxCas];
    const playerPickCount = Math.max(0, Math.min(pickCount + playerPickBonus, pm.length));
    if (playerMode === 'ai') {
      const before = pm.slice();
      pm = aiTakeActions(pm, playerActions);
      if (JSON.stringify(before) !== JSON.stringify(pm)) actionsUsed += playerActions;
    } else if (playerMode === 'ai-dice') {
      const before = pm.slice();
      const r = aiTakeActionsDiceAware(pm, playerActions, playerPickCount, diceRoll);
      pm = r.meeples;
      if (JSON.stringify(before) !== JSON.stringify(pm)) actionsUsed += playerActions;
    }
    let playerPicks;
    if (playerMode === 'ai') {
      const idxs = aiPickMeeples(pm, playerPickCount);
      playerPicks = idxs.map(i => pm[i]);
    } else if (playerMode === 'ai-dice') {
      const [ms, mk, mp] = toCounts(pm);
      const [sp, kp, pp] = bestPickCounts(ms, mk, mp, playerPickCount, diceRoll);
      playerPicks = [];
      for (let i = 0; i < sp; i++) playerPicks.push(SWEEP);
      for (let i = 0; i < kp; i++) playerPicks.push(KICK);
      for (let i = 0; i < pp; i++) playerPicks.push(PUNCH);
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

  const modes = args.includes('--all-modes')
    ? ['ai-dice', 'ai', 'baseline']
    : ['ai-dice', 'ai'];

  function section(label, rollCount, advRange) {
    console.log(`=== ${label}  (diceRollCount=${rollCount}) ===`);
    const header = 'adv  mode       player%  dice%   draw%   battles(avg/med)  action-use';
    console.log(header);
    console.log('-'.repeat(header.length));
    for (const adv of advRange) {
      for (const mode of modes) {
        const r = runBatch({ playerStart: 8, diceStart: 8 + adv, playerMode: mode, diceRollCount: rollCount }, N);
        const tag = mode === 'ai-dice' ? 'AI-dice ' : mode === 'ai' ? 'AI      ' : 'baseline';
        const advStr = (adv >= 0 ? '+' : '') + adv;
        console.log(`${advStr.padStart(3)}  ${tag}  ${r.playerWinPct.padStart(6)}   ${r.diceWinPct.padStart(5)}   ${r.drawPct.padStart(5)}   ${r.avgBattles.padStart(5)} / ${String(r.medianBattles).padStart(3)}        ${r.actionUseRate}`);
      }
      console.log('');
    }
  }

  if (args.includes('--handicap')) {
    // Player handicap: both start with 8, player picks one fewer than standard.
    // Compare across dice-roll variants.
    console.log('Player picks pickCount-1, both start with 8 meeples.');
    console.log('');
    const header = 'variant   adv  mode       player%  dice%   battles(avg/med)  action-use';
    for (const [label, rollCount] of [['pool', 'pool'], ['pick', 'pick'], ['pick+1', 'pick+1']]) {
      console.log(`=== diceRollCount=${rollCount}, playerPickBonus=-1 ===`);
      console.log(header);
      console.log('-'.repeat(header.length));
      for (const adv of [-2, -1, 0, 1, 2]) {
        for (const mode of ['ai-dice', 'ai']) {
          const r = runBatch({
            playerStart: 8, diceStart: 8 + adv,
            playerMode: mode, diceRollCount: rollCount, playerPickBonus: -1,
          }, N);
          const tag = mode === 'ai-dice' ? 'AI-dice ' : 'AI      ';
          const advStr = (adv >= 0 ? '+' : '') + adv;
          console.log(`${label.padEnd(8)}  ${advStr.padStart(3)}  ${tag}  ${r.playerWinPct.padStart(6)}   ${r.diceWinPct.padStart(5)}   ${r.avgBattles.padStart(5)} / ${String(r.medianBattles).padStart(3)}        ${r.actionUseRate}`);
        }
        console.log('');
      }
    }
    return;
  }
  if (args.includes('--sweep')) {
    // Fine-grained advantage sweep to find the interesting (~50%) sweet spot.
    const r = (a, b) => { const o = []; for (let i = a; i <= b; i++) o.push(i); return o; };
    section('A  — pool',   'pool',   r(-4, 4));
    section('B  — pick',   'pick',   r(0, 10));
    section('C  — pick+1', 'pick+1', r(-2, 4));
    section('D  — pick+2', 'pick+2', r(-2, 4));
    return;
  }
  section('A  — dice rolls entire pool (literal +N reading)', 'pool',   [-4, -2, 0, 1, 2, 4]);
  section('B  — dice rolls pickCount (matches player reveal)', 'pick',  [0, 1, 2, 3, 4]);
  section('C  — dice rolls pickCount + 1 (one extra die)',      'pick+1',[0, 1, 2, 3]);
  section('D  — dice rolls pickCount + 2 (two extra dice)',     'pick+2',[0, 1, 2, 3]);
}

main();
