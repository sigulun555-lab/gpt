'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const {
  WINDOW20, CONTROL12, buildWindow20Report, priorTeamMatchCount,
  teamWindowStatus, validateWindow20OutputPath,
} = require('../scripts/run_toto13_window20_shadow.js');
const { validateHistory, predictToto13 } = require('../scripts/run_toto13_shadow.js');

function fixtureDocument() {
  return {
    round: 1674,
    marketType: 'toto',
    deadline: '2026-11-25T18:50:00+09:00',
    source: {
      url: 'https://store.toto-dream.com/dcs/subos/screen/pi01/spin000/PGSPIN00001DisptotoLotInfo.form?holdCntId=1674',
      observedAt: '2026-11-18T18:37:00+09:00',
      revalidatedAt: '2026-11-21T14:01:00+09:00',
    },
    fixtures: Array.from({ length: 13 }, (_, i) => ({
      number: i + 1, date: '2026-11-25', competition: 'J1',
      home: 'Alpha', away: 'Beta',
    })),
  };
}

function historyRows(count) {
  return Array.from({ length: count }, (_, i) => ({
    date: '2026-01-' + String(i + 1).padStart(2, '0'),
    competition: 'J1',
    home: 'Alpha',
    away: 'Beta',
    home_goals: i % 4,
    away_goals: (i + 1) % 4,
    fetchedAt: '2026-01-31T00:00:00Z',
  }));
}

const options = {
  nowIso: '2026-11-24T22:00:00+09:00',
  historyDocument: { updatedAt: '2026-11-20T20:14:03.043Z', source: { name: 'test', kind: 'fixture' } },
  execution: { repository: 'sigulun555-lab/gpt', workflowRunId: 'test', eventName: 'push' },
};

test('preregistered arm changes only the window: k=10/window20 vs k=10/window12', () => {
  assert.equal(WINDOW20.shrink, 10);
  assert.equal(WINDOW20.window, 20);
  assert.equal(CONTROL12.shrink, 10);
  assert.equal(CONTROL12.window, 12);
});

test('20+ prior matches use latest 20 and control uses latest 12', () => {
  const h = historyRows(21);
  const report = buildWindow20Report(h, fixtureDocument(), options);
  assert.equal(report.coverage.pairedEligibleFixtures, 13);
  assert.equal(report.coverage.partialWindowTeamSides, 0);
  const row = report.predictions[0];
  assert.equal(row.teamStatus.home.priorCompletedMatches, 21);
  assert.equal(row.arms[WINDOW20.id].homeWindowUsed, 20);
  assert.equal(row.arms[WINDOW20.id].homePartialWindow, false);
  assert.equal(row.arms[CONTROL12.id].homeWindowUsed, 12);
  assert.equal(row.arms[CONTROL12.id].homePartialWindow, false);
  assert.ok(Math.abs(Object.values(row.arms[WINDOW20.id].probabilities).reduce((a,b)=>a+b,0)-1)<1e-12);
});

test('12-19 prior matches use all available for W20 and set partial_window=true', () => {
  const report = buildWindow20Report(historyRows(15), fixtureDocument(), options);
  const row = report.predictions[0];
  assert.equal(row.pairedEligible, true);
  assert.equal(row.arms[WINDOW20.id].homeWindowUsed, 15);
  assert.equal(row.arms[WINDOW20.id].homePartialWindow, true);
  assert.equal(row.arms[CONTROL12.id].homeWindowUsed, 12);
  assert.equal(row.arms[CONTROL12.id].homePartialWindow, false);
});

test('fewer than 12 prior matches are marked missing with no neutral fallback', () => {
  const report = buildWindow20Report(historyRows(11), fixtureDocument(), options);
  assert.equal(report.coverage.pairedEligibleFixtures, 0);
  assert.equal(report.coverage.missingTeamSides, 26);
  const row = report.predictions[0];
  assert.equal(row.teamStatus.home.status, 'missing_below_12_match_floor');
  assert.equal(row.arms[WINDOW20.id].probabilities, null);
  assert.equal(row.arms[CONTROL12.id].probabilities, null);
});

test('same-day and future results are excluded from team counts', () => {
  const h = historyRows(15);
  h.push({ date:'2026-11-25', competition:'J1', home:'Alpha', away:'Beta', home_goals:9, away_goals:0 });
  h.push({ date:'2026-11-26', competition:'J1', home:'Alpha', away:'Beta', home_goals:9, away_goals:0 });
  assert.equal(priorTeamMatchCount(validateHistory(h), 'Alpha', '2026-11-25'), 15);
});

test('GOAL3, mini toto, round 1673 and wrong deadlines are rejected', () => {
  const goal3 = fixtureDocument(); goal3.marketType='totoGOAL3'; goal3.fixtures=goal3.fixtures.slice(0,3);
  assert.throws(()=>buildWindow20Report(historyRows(20),goal3,options),/ordinary-toto13/);
  const mini = fixtureDocument(); mini.marketType='mini toto A'; mini.fixtures=mini.fixtures.slice(0,5);
  assert.throws(()=>buildWindow20Report(historyRows(20),mini,options),/ordinary-toto13/);
  const old = fixtureDocument(); old.round=1673;
  assert.throws(()=>buildWindow20Report(historyRows(20),old,options),/round 1674 only/);
  const wrongDeadline = fixtureDocument(); wrongDeadline.deadline='2026-11-25T17:00:00+09:00';
  assert.throws(()=>buildWindow20Report(historyRows(20),wrongDeadline,options),/deadline/);
});

test('generation before 22:00 previous day or after deadline is refused', () => {
  assert.throws(()=>buildWindow20Report(historyRows(20),fixtureDocument(),{...options,nowIso:'2026-11-24T21:59:59+09:00'}),/not allowed before/);
  assert.throws(()=>buildWindow20Report(historyRows(20),fixtureDocument(),{...options,nowIso:'2026-11-25T18:50:00+09:00'}),/deadline has passed/);
});

test('output path is restricted to the separate Window-20 directory', () => {
  const root='/repo';
  assert.equal(validateWindow20OutputPath('shadow/toto13/window20/predictions/round_1674.json',root),'/repo/shadow/toto13/window20/predictions/round_1674.json');
  assert.throws(()=>validateWindow20OutputPath('shadow/toto13/predictions/round_1674.json',root),/window20/);
  assert.throws(()=>validateWindow20OutputPath('data/predictions.json',root),/window20/);
});

test('the W20 and V1 predictions call the existing T1 function with the preregistered configs', () => {
  const h=validateHistory(historyRows(21));
  const fx=fixtureDocument().fixtures[0];
  const report=buildWindow20Report(h,fixtureDocument(),options);
  const row=report.predictions[0];
  const directW20=predictToto13(h,fx,WINDOW20);
  const directV1=predictToto13(h,fx,CONTROL12);
  assert.deepEqual(row.arms[WINDOW20.id].probabilities,directW20.probabilities);
  assert.deepEqual(row.arms[CONTROL12.id].probabilities,directV1.probabilities);
});
