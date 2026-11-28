'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

// The official selection page is dynamic. Never pin a live test to an expired round.
(async () => {
  const headers = {
    Accept: 'text/html,application/xhtml+xml,*/*',
    'User-Agent': 'JLeagueTotoPredictor/0.1 (official-page regression test)'
  };
  const options = { headers, signal: AbortSignal.timeout(20000) };
  const now = new Date();

  // Standard toto: explicitly select commodityId=01 and read the currently selected round.
  const totoUrl = 'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=01&gameAssortment=1';
  const totoResponse = await fetch(totoUrl, options);
  assert.equal(totoResponse.ok, true, 'official standard-toto selection page should return HTTP success');
  const totoHtml = await totoResponse.text();
  assert.ok(totoHtml.length > 1000, 'official standard-toto HTML should be non-empty');
  const toto = parseToto(totoHtml, now);
  assert.equal(toto.marketType, 'toto', 'commodityId=01 should be identified as standard toto');
  assert.ok(Number.isInteger(toto.round) && toto.round > 0, 'selected official standard-toto round should be parsed');
  assert.ok(toto.deadline && Date.parse(toto.deadline) > now.getTime(), 'standard-toto deadline should be in the future');
  assert.equal(toto.fixtures.length, 13, 'all 13 current standard-toto fixtures should be parsed');
  assert.deepEqual(toto.fixtures.map(f => f.number), Array.from({length:13}, (_,i)=>i+1),
    'standard-toto fixture numbers should be exactly 1..13');
  assert.ok(toto.fixtures.every(f => f.date && f.home && f.away && f.home !== f.away),
    'all standard-toto fixtures should have date, home and away teams');
  console.log('PASS live official standard toto:', JSON.stringify({
    round: toto.round, deadline: toto.deadline, parsedAt: toto.parsedAt,
    fixtures: toto.fixtures.map(({number,date,dateLabel,home,away})=>({number,date,dateLabel,home,away}))
  }));

  // GOAL3 remains a separate 3-match product and must never be used as toto13 input.
  const goal3Url = 'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1';
  const goal3Response = await fetch(goal3Url, options);
  assert.equal(goal3Response.ok, true, 'official GOAL3 selection page should return HTTP success');
  const goal3Html = await goal3Response.text();
  assert.ok(goal3Html.length > 1000, 'official GOAL3 HTML should be non-empty');
  const goal3 = parseToto(goal3Html, now);
  assert.equal(goal3.marketType, 'totoGOAL3', 'commodityId=02 should be identified as totoGOAL3');
  assert.ok(Number.isInteger(goal3.round) && goal3.round > 0, 'selected official GOAL3 round should be parsed');
  assert.ok(goal3.deadline && Date.parse(goal3.deadline) > now.getTime(), 'GOAL3 deadline should be in the future');
  assert.equal(goal3.fixtures.length, 3, 'GOAL3 should normalize six team rows to three matches');
  assert.deepEqual(goal3.fixtures.map(f=>f.number), [1,2,3], 'GOAL3 fixture numbers should be 1..3');
  console.log('PASS live official totoGOAL3:', JSON.stringify({
    round: goal3.round, deadline: goal3.deadline,
    fixtures: goal3.fixtures.map(({number,date,dateLabel,home,away})=>({number,date,dateLabel,home,away}))
  }));
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
