'use strict';
const assert = require('node:assert/strict');
const { parseToto } = require('../scripts/auto_pipeline.js');

// Regression: the selected round must win over a newer round listed in navigation.
{
  const html = '<label>開催回</label><select name="holdCntId"><option value="1668">第1668回</option><option value="1666" selected="selected">第1666回</option></select><div>第1668回 totoくじ情報</div><div>〖くじ〗 totoGOAL3</div>';
  assert.equal(parseToto(html, new Date('2026-10-28T00:00:00+09:00')).round, 1666,
    'selected official round should take precedence over unselected navigation round');
}


function officialPlainText(html) {
  return String(html)
    .replace(/<script\b[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style\b[\s\S]*?<\/style>/gi, ' ')
    .replace(/<br\s*\/?>/gi, ' ')
    .replace(/<\/(?:tr|td|th|div|p|li|option|h[1-6])\s*>/gi, ' ')
    .replace(/<[^>]*>/g, ' ')
    .replace(/&nbsp;|&#160;/gi, ' ')
    .replace(/&amp;/gi, '&')
    .replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>')
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/[\u3000\s]+/g, ' ')
    .trim();
}

function extractGoal3DeadlineFromSalesTermsHtml(html, round) {
  const plain = officialPlainText(html);
  const heading = new RegExp('第\\s*' + Number(round) + '\\s*回\\s*toto\\s*GOAL\\s*3\\s*くじ情報', 'i');
  const matches = [...plain.matchAll(new RegExp(heading.source, 'gi'))];
  if (matches.length !== 1) {
    throw new Error('Expected exactly one official totoGOAL3 sales block for round ' + round + '; found ' + matches.length);
  }
  let section = plain.slice(matches[0].index + matches[0][0].length);
  const nextProduct = section.search(/第\s*\d+\s*回\s*(?:toto|mini\s*toto)/i);
  if (nextProduct >= 0) section = section.slice(0, nextProduct);
  const dateMatch = section.match(/販売終了日\s*(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日/);
  if (!dateMatch) throw new Error('Official totoGOAL3 sales block has no sale-end date for round ' + round);
  const afterDate = section.slice(dateMatch.index + dateMatch[0].length, dateMatch.index + dateMatch[0].length + 240);
  const timeMatch = afterDate.match(/当サイト\s*\(\s*ネット決済\s*\)\s*(\d{1,2})\s*[:：]\s*(\d{2})/);
  if (!timeMatch) throw new Error('Official totoGOAL3 sales block has no network deadline for round ' + round);
  const [, year, month, day] = dateMatch;
  const [, hour, minute] = timeMatch;
  return year + '-' + month.padStart(2, '0') + '-' + day.padStart(2, '0') +
    'T' + hour.padStart(2, '0') + ':' + minute + ':00+09:00';
}

// Regression: the GOAL3 deadline must come from its own product block,
// not the first regular-toto deadline on a multi-product sales page.
{
  const fixture = '<section><h2>第1670回 totoくじ情報</h2>販売終了日 2026年11月7日（当サイト(ネット決済) 12:50）</section>' +
    '<section><h2>第1670回 mini toto-A組 くじ情報</h2>販売終了日 2026年11月7日（当サイト(ネット決済) 12:50）</section>' +
    '<section><h2>第1670回 mini toto-B組 くじ情報</h2>販売終了日 2026年11月7日（当サイト(ネット決済) 12:50）</section>' +
    '<section><h2>第1670回 totoGOAL3 くじ情報</h2>販売終了日 2026年11月7日（当サイト(ネット決済) 13:50）</section>';
  assert.equal(extractGoal3DeadlineFromSalesTermsHtml(fixture, 1670), '2026-11-07T13:50:00+09:00');
}

(async () => {
  const headers = {
    Accept: 'text/html,application/xhtml+xml,*/*',
    'User-Agent': 'JLeagueTotoPredictor/0.1 (official-page regression test)'
  };
  // The official host intermittently closes TLS sockets between sequential requests.
  // Retry only transport errors and transient HTTP statuses; parsing/assertion failures
  // remain hard failures and are never converted into a pass.
  async function fetchOfficialHtml(url, label) {
    const maxAttempts = 3;
    for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
      try {
        const response = await fetch(url, { headers, signal: AbortSignal.timeout(20000) });
        if (!response.ok) {
          const transient = response.status === 429 || response.status >= 500;
          if (!transient) throw new Error(label + ' returned HTTP ' + response.status);
          if (attempt === maxAttempts) {
            const unavailable = new Error(label + ' unavailable after ' + maxAttempts + ' attempts (HTTP ' + response.status + ')');
            unavailable.code = 'LIVE_OFFICIAL_UNAVAILABLE';
            throw unavailable;
          }
        } else {
          const html = await response.text();
          if (html.length > 1000) return html;
          if (attempt === maxAttempts) {
            throw new Error(label + ' returned unexpectedly short HTML (' + html.length + ' bytes)');
          }
        }
      } catch (error) {
        const code = error && (error.cause?.code || error.code);
        const retryable = error.name === 'TypeError' ||
          ['UND_ERR_SOCKET', 'ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN'].includes(code) ||
          /terminated|other side closed|fetch failed/i.test(String(error.message));
        if (!retryable) throw error;
        if (attempt === maxAttempts) {
          const unavailable = new Error(label + ' unavailable after ' + maxAttempts + ' attempts: ' + error.message);
          unavailable.code = 'LIVE_OFFICIAL_UNAVAILABLE';
          throw unavailable;
        }
      }
      await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
    }
    throw new Error(label + ' failed after retries');
  }

  // Standard toto remains supported and must be distinguished from GOAL3.
  const totoUrl = 'https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardSalesTermtotoSP.form?holdCntId=1668';
  const totoHtml = await fetchOfficialHtml(totoUrl, 'official standard-toto page');
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
  // to supply its current round; enforce a known lower bound so a stale old form
  // cannot pass merely because it contains a positive integer.
  const goal3Url = 'https://sp.toto-dream.com/dcs/subos/screen/ss01/sssl014/PGSSSL01401InittotoSP.form?commodityId=02&gameAssortment=1';
  const goal3Html = await fetchOfficialHtml(goal3Url, 'official GOAL3 selection page');
  const goal3AsOf = new Date();
  const goal3 = parseToto(goal3Html, goal3AsOf);
  assert.ok(Number.isInteger(goal3.round) && goal3.round >= 1670,
    'current official GOAL3 round should be at least the latest verified round 1670');
  assert.equal(goal3.marketType, 'totoGOAL3', 'commodityId=02 should be identified as totoGOAL3');
  assert.ok(typeof goal3.deadline === 'string' && Number.isFinite(Date.parse(goal3.deadline)),
    'current official GOAL3 online deadline should be parsed as a valid timestamp');
  assert.ok(Date.parse(goal3.deadline) > goal3AsOf.getTime(),
    'current official GOAL3 sales deadline must still be in the future');
  assert.equal(goal3.fixtures.length, 3, 'GOAL3 should normalize six team rows to three matches');
  assert.ok(goal3.fixtures.every(f => f.home && f.away && f.home !== f.away),
    'all current GOAL3 fixtures should have distinct non-empty home/away teams');
  assert.equal(new Set(goal3.fixtures.map(f => f.home + '|' + f.away)).size, 3,
    'current GOAL3 fixtures should be unique');
  const salesTermsUrl = 'https://sp.toto-dream.com/dcs/subos/screen/si01/ssin025/PGSSIN02501ForwardSalesTermtotoSP.form?holdCntId=' + goal3.round;
  const salesTermsHtml = await fetchOfficialHtml(salesTermsUrl, 'official product sales-terms page');
  const independentlyParsedGoal3Deadline = extractGoal3DeadlineFromSalesTermsHtml(salesTermsHtml, goal3.round);
  assert.equal(
    goal3.deadline,
    independentlyParsedGoal3Deadline,
    'prediction parser deadline must match the independently parsed deadline inside the selected round totoGOAL3 product block'
  );
  console.log('PASS live official totoGOAL3 HTML: current round ' + goal3.round +
    ', product-specific deadline ' + independentlyParsedGoal3Deadline + ', and all 3 fixtures');
})().catch(error => {
  if (error && error.code === 'LIVE_OFFICIAL_UNAVAILABLE') {
    console.warn('SKIP live official HTML assertions: external source transport unavailable; deterministic parser assertions remain active.');
    console.warn(error.message);
    return;
  }
  console.error(error);
  process.exitCode = 1;
});
