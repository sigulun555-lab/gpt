'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

function selectedRoundFromHtml(html) {
  const selectedOption = [...String(html).matchAll(/<option\b([^>]*)>([\s\S]*?)<\/option>/gi)]
    .find(([, attrs, label]) => /\bselected(?:\s|=|$)/i.test(attrs) && /第\s*\d+\s*回/.test(label.replace(/<[^>]*>/g, ' ')));
  if (selectedOption) {
    const match = selectedOption[2].replace(/<[^>]*>/g, ' ').match(/第\s*(\d+)\s*回/);
    if (match) return Number(match[1]);
  }
  const selectedInput = [...String(html).matchAll(/<input\b([^>]*)>/gi)]
    .find(([, attrs]) => /\bname=["'](?:holdCntId|kaiCntId|roundNo)["']/i.test(attrs)
      && /\bvalue=["']\d+["']/i.test(attrs));
  if (selectedInput) {
    const match = selectedInput[1].match(/\bvalue=["'](\d+)["']/i);
    if (match) return Number(match[1]);
  }
  return null;
}

// Regression: the selected round must win over a newer round listed in navigation.
{
  const html = '<label>開催回</label><select name="holdCntId"><option value="1668">第1668回</option><option value="1666" selected="selected">第1666回</option></select><div>第1668回 totoくじ情報</div><div>〖くじ〗 totoGOAL3</div>';
  assert.equal(parseToto(html, new Date('2026-10-28T00:00:00+09:00')).round, 1666,
    'selected official round should take precedence over unselected navigation round');
}

(async () => {
  const headers = {
    Accept: 'text/html,application/xhtml+xml,*/*',
    'User-Agent': 'JLeagueTotoPredictor/0.1 (official-page regression test)'
  };
  const options = { headers, signal: AbortSignal.timeout(20000) };

  // Standard toto remains supported and must be distinguished from GOAL3.
  const totoUrl = 'https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardSalesTermtotoSP.form?holdCntId=1668';
  const totoResponse = await fetch(totoUrl, options);
  assert.equal(totoResponse.ok, true, 'official standard-toto page should return HTTP success');
  const totoHtml = await totoResponse.text();
  assert.ok(totoHtml.length > 1000, 'official standard-toto HTML should be non-empty');
  const toto = parseToto(totoHtml, new Date('2026-10-28T00:00:00+09:00'));
  const expectedToto = [
    ['岡山', 'FC東京'], ['水戸', '広島'], ['神戸', '名古屋'],
    ['鹿島', 'Ｃ大阪'], ['柏', '浦和'], ['京都', '清水'],
    ['千葉', '福岡'], ['長崎', '東京Ｖ'], ['町田', 'Ｇ大阪'],
    ['横浜FM', '川崎Ｆ'], ['仙台', '大宮'], ['新潟', '富山'],
    ['宮崎', '大分']
  ];
  assert.equal(toto.round, 1668, 'official standard-toto round should be parsed');
  assert.equal(toto.marketType, 'toto', 'commodityId=01 should be identified as standard toto');
  assert.equal(toto.deadline, '2026-10-31T13:50:00+09:00', 'standard-toto deadline should be parsed');
  assert.equal(toto.fixtures.length, 13, 'all 13 standard-toto fixtures should be parsed');
  assert.deepEqual(toto.fixtures.map(f => [f.home, f.away]), expectedToto,
    'standard-toto home/away pairs should match in order');
  console.log('PASS live official toto HTML: round 1668, deadline, and all 13 fixtures');

  // Explicitly choose commodityId=02 (totoGOAL3) while allowing the official page
  // to supply its current round; do not rely on the page's default commodityId=01.
  const goal3Url = 'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1';
  const goal3Response = await fetch(goal3Url, options);
  assert.equal(goal3Response.ok, true, 'official GOAL3 selection page should return HTTP success');
  const goal3Html = await goal3Response.text();
  assert.ok(goal3Html.length > 1000, 'official GOAL3 HTML should be non-empty');
  const goal3 = parseToto(goal3Html, new Date('2026-10-28T00:00:00+09:00'));
  const expectedGoal3Round = selectedRoundFromHtml(goal3Html);
  assert.ok(Number.isInteger(expectedGoal3Round) && expectedGoal3Round > 0,
    'official GOAL3 page should expose a selected round');
  assert.equal(goal3.round, expectedGoal3Round,
    'current official GOAL3 round should be parsed from the selected form, not stale navigation text');
  assert.equal(goal3.marketType, 'totoGOAL3', 'commodityId=02 should be identified as totoGOAL3');
  assert.ok(goal3.deadline === null || (/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$/.test(goal3.deadline) && Number.isFinite(Date.parse(goal3.deadline))),
    'selected GOAL3 deadline must be absent or a valid JST ISO-8601 timestamp');
  if (goal3.deadline === null) {
    console.log('WARN official GOAL3 page omitted a parsable deadline; pipeline integrity gate must suppress forecast publication');
  }
  assert.equal(goal3.fixtures.length, 3, 'GOAL3 should normalize six team rows to three matches');
  assert.deepEqual(goal3.fixtures.map(f => f.number), [1, 2, 3], 'GOAL3 fixture numbers should be sequential');
  assert.ok(goal3.fixtures.every(f => f.date && f.home && f.away && f.home !== f.away),
    'all selected GOAL3 fixtures should have valid dates and distinct home/away teams');
  assert.equal(new Set(goal3.fixtures.flatMap(f => [f.home, f.away])).size, 6,
    'the selected GOAL3 page should identify six distinct team-side targets');
  console.log('PASS live official totoGOAL3 HTML: selected round ' + goal3.round + ', deadline, and 3 fixtures');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
