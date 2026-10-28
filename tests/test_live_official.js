'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

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
  assert.equal(goal3.round, 1668, 'current official GOAL3 round should be parsed from the selected form');
  assert.equal(goal3.marketType, 'totoGOAL3', 'commodityId=02 should be identified as totoGOAL3');
  assert.equal(goal3.deadline, '2026-10-31T13:50:00+09:00', 'official GOAL3 online deadline should be parsed');
  assert.equal(goal3.fixtures.length, 3, 'GOAL3 should normalize six team rows to three matches');
  assert.deepEqual(goal3.fixtures.map(f => [f.home, f.away]), [
    ['岡山', 'FC東京'], ['神戸', '名古屋'], ['柏', '浦和']
  ], 'official GOAL3 home/away pairs should match in order');
  console.log('PASS live official totoGOAL3 HTML: round 1668, deadline, and all 3 fixtures');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
