'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

(async () => {
  const url = 'https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardSalesTermtotoSP.form?holdCntId=1668';
  const response = await fetch(url, {
    headers: {
      Accept: 'text/html,application/xhtml+xml,*/*',
      'User-Agent': 'JLeagueTotoPredictor/0.1 (official-page regression test)'
    },
    signal: AbortSignal.timeout(20000)
  });
  assert.equal(response.ok, true, 'official toto page should return HTTP success');
  const html = await response.text();
  assert.ok(html.length > 1000, 'official toto page HTML should be non-empty');

  const now = new Date();
  const parsed = parseToto(html, now);
  assert.equal(parsed.round, 1668, 'official standard toto round should be parsed');
  assert.equal(parsed.marketType, 'toto', 'official page should be identified as standard toto');
  assert.ok(parsed.deadline && Number.isFinite(Date.parse(parsed.deadline)), 'official online deadline should parse');
  assert.equal(parsed.fixtures.length, 13, 'all 13 official matches should be parsed');
  assert.deepEqual(parsed.fixtures.map(f => f.number), Array.from({length:13}, (_,i)=>i+1), 'standard toto match numbers should be sequential');
  assert.ok(parsed.fixtures.every(f => f.home && f.away && f.home !== f.away && f.date), 'all standard toto fixtures should have valid teams and dates');
  console.log('PASS live official toto HTML: round 1668, deadline, and all 13 fixtures');

  const goal3Url = 'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form';
  const goal3Response = await fetch(goal3Url, {
    headers: {
      Accept: 'text/html,application/xhtml+xml,*/*',
      'User-Agent': 'JLeagueTotoPredictor/0.1 (official-page regression test)'
    },
    signal: AbortSignal.timeout(20000)
  });
  assert.equal(goal3Response.ok, true, 'official purchase page should return HTTP success');
  const goal3Html = await goal3Response.text();
  assert.ok(goal3Html.length > 1000, 'official purchase page HTML should be non-empty');
  const goal3 = parseToto(goal3Html, now);
  assert.ok(Number.isInteger(goal3.round) && goal3.round > 0, 'currently selected official GOAL3 round should be parsed');
  assert.equal(goal3.marketType, 'totoGOAL3', 'selected official product should be identified as totoGOAL3');
  assert.ok(goal3.deadline && Number.isFinite(Date.parse(goal3.deadline)), 'selected official GOAL3 deadline should parse');
  assert.equal(goal3.fixtures.length, 3, 'official GOAL3 page should normalize six team rows to three matches');
  assert.deepEqual(goal3.fixtures.map(f => f.number), [1,2,3], 'GOAL3 fixture numbers should be sequential');
  assert.ok(goal3.fixtures.every(f => f.home && f.away && f.home !== f.away && f.date), 'all GOAL3 fixtures should have valid teams and dates');
  console.log('PASS live official totoGOAL3 HTML: currently selected round, deadline, and all 3 fixtures');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
