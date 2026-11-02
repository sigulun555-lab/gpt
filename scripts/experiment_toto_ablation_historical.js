'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { parseSchedule } = require('./auto_pipeline.js');
const { canonical } = require('../model.js');
const { runAblation } = require('./experiment_toto_ablation.js');

const ROOT = path.resolve(__dirname, '..');
const HISTORICAL_URL = process.env.JLEAGUE_HISTORICAL_CSV_URL ||
  'https://raw.githubusercontent.com/k-addtimes/jleague_open_data/main/processed/2025_schedule_with_ids.csv';

function normalizeCompetition(value) {
  const raw = String(value || '').trim().toUpperCase();
  return /^[123]$/.test(raw) ? 'J' + raw : raw;
}

async function fetchHistoricalCsv() {
  const response = await fetch(HISTORICAL_URL, {
    headers: {
      'User-Agent': 'JLeagueTotoPredictor-accuracy-audit/0.1',
      Accept: 'text/csv,text/plain,*/*'
    },
    signal: AbortSignal.timeout(30000)
  });
  if (!response.ok) throw new Error('Historical CSV fetch failed: HTTP ' + response.status);
  const csv = await response.text();
  const normalized = csv.replace(/^\\uFEFF/, '');
  if (normalized.length < 100 || !/^year,category,term,date,/i.test(normalized)) {
    throw new Error('Historical CSV schema did not match the expected 2025 source');
  }
  return csv;
}

async function main() {
  const current = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'history.json'), 'utf8')).history
    .filter(m => m.date && m.competition && Number.isFinite(m.home_goals) && Number.isFinite(m.away_goals));
  const targetStartDate = current.map(m => m.date).sort()[0];
  if (!targetStartDate) throw new Error('Current-season target history is empty');

  const historicalParsed = parseSchedule(await fetchHistoricalCsv());
  const historical = historicalParsed
    .filter(m => m.date && m.date.startsWith('2025-') &&
      Number.isInteger(m.home_goals) && Number.isInteger(m.away_goals) &&
      m.home_goals >= 0 && m.away_goals >= 0)
    .map(m => ({ ...m, competition: normalizeCompetition(m.competition), season: 2025,
      source: 'community_jleague_historical_csv', sourceUrl: HISTORICAL_URL }));

  const currentNormalized = current.map(m => ({
    ...m, competition: normalizeCompetition(m.competition), season: Number(m.date.slice(0, 4))
  }));
  const combined = new Map();
  for (const row of [...historical, ...currentNormalized]) {
    if (!['J1', 'J2', 'J3'].includes(row.competition)) continue;
    const key = [row.date, row.competition, canonical(row.home), canonical(row.away)].join('|');
    combined.set(key, row);
  }
  const history = [...combined.values()].sort((a, b) => a.date.localeCompare(b.date));
  const historicalRows = history.filter(m => m.date.startsWith('2025-'));
  const targetRows = history.filter(m => m.date >= targetStartDate);
  const competitionCounts = {};
  for (const row of history) competitionCounts[row.competition] = (competitionCounts[row.competition] || 0) + 1;

  if (historicalRows.length < 1000) {
    throw new Error('Expected at least 1,000 verified-schema 2025 history rows; found ' + historicalRows.length);
  }
  if (targetRows.length < 100) {
    throw new Error('Expected at least 100 current-season target rows; found ' + targetRows.length);
  }
  for (const league of ['J1', 'J2', 'J3']) {
    if (!competitionCounts[league]) throw new Error('Missing competition history: ' + league);
  }

  const report = runAblation(history, {
    targetStartDate,
    source: '2025 historical CSV plus checked-in 2026 history; community J.League data mirror, not official results feed'
  });
  report.historySources = {
    historical2025: { url: HISTORICAL_URL, rows: historicalRows.length },
    current2026: { path: 'data/history.json', rows: currentNormalized.length }
  };
  report.competitionCounts = competitionCounts;
  report.evaluationScope = '2026 target matches only; 2025 and strictly earlier 2026 matches may be used as training history';
  process.stdout.write(JSON.stringify(report, null, 2) + '\\n');
}

main().catch(error => {
  console.error('HISTORICAL_ABLATION_FAILED:', error.message);
  process.exitCode = 1;
});
