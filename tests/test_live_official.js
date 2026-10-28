'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

// Regression: the currently selected round must win over a newer round listed elsewhere.
{
  const html = '<label>開催回</label><select name="holdCntId"><option value="1668">第1668回</option><option value="1666" selected="selected">第1666回</option></select><div>第1668回 totoくじ情報</div><div>〖くじ〗 totoGOAL3</div>';
  assert.equal(parseToto(html, new Date('2026-10-28T00:00:00+09:00')).round, 1666,
    'selected official round should take precedence over unselected navigation round');
}

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

  // Freeze the parser's reference date so month/year normalization is deterministic.
  const parsed = parseToto(html, new Date('2026-10-28T00:00:00+09:00'));
  const expected = [
    ['岡山', 'FC東京'], ['水戸', '広島'], ['神戸', '名古屋'],
    ['鹿島', 'Ｃ大阪'], ['柏', '浦和'], ['京都', '清水'],
    ['千葉', '福岡'], ['長崎', '東京Ｖ'], ['町田', 'Ｇ大阪'],
    ['横浜FM', '川崎Ｆ'], ['仙台', '大宮'], ['新潟', '富山'],
    ['宮崎', '大分']
  ];
  assert.equal(parsed.round, 1668, 'official page round should be parsed');
  assert.equal(parsed.marketType, 'toto', 'official page should be identified as standard toto');
  assert.equal(parsed.deadline, '2026-10-31T13:50:00+09:00', 'official online deadline should be parsed');
  assert.equal(parsed.fixtures.length, 13, 'all 13 official matches should be parsed');

  assert.deepEqual(
    parsed.fixtures.map(f => [f.home, f.away]),
    expected,
    'all official home/away pairs should match in order'
  );
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
  const goal3 = parseToto(goal3Html, new Date('2026-10-28T00:00:00+09:00'));
  const roundControlTags = [...goal3Html.matchAll(/<(?:select|option|input|a)\b[^>]*>/gi)]
    .map(m => m[0])
    .filter(tag => /hold|kai|round|1666|1668|selected/i.test(tag))
    .slice(0, 60);
  const roundTextContexts = [...goal3Html.matchAll(/開催回|第1666回|第1668回/g)]
    .slice(0, 20)
    .map(m => goal3Html.slice(Math.max(0, m.index - 90), Math.min(goal3Html.length, m.index + 150)).replace(/\s+/g, ' '));
  console.log('DEBUG GOAL3 round control tags', JSON.stringify(roundControlTags));
  console.log('DEBUG GOAL3 round text contexts', JSON.stringify(roundTextContexts));
  assert.equal(goal3.round, 1666, 'official selected GOAL3 round should be parsed');
  assert.equal(goal3.marketType, 'totoGOAL3', 'selected official product should be identified as totoGOAL3');
  assert.equal(goal3.deadline, '2026-10-28T18:50:00+09:00', 'official GOAL3 deadline should be parsed');
  assert.equal(goal3.fixtures.length, 3, 'official GOAL3 page should normalize six team rows to three matches');
  assert.deepEqual(
    goal3.fixtures.map(f => [f.home, f.away]),
    [['水戸', 'Ｃ大阪'], ['横浜FM', '広島'], ['名古屋', '浦和']],
    'official GOAL3 home/away pairs should match in order'
  );
  console.log('PASS live official totoGOAL3 HTML: round 1666, deadline, and all 3 fixtures');

})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
