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
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
