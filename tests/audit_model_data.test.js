'use strict';
const assert=require('node:assert/strict');
const {run}=require('../scripts/audit_model_data.js');
const r=run();
assert.ok(r.dataQuality.historyRows>0);
assert.equal(r.dataQuality.invalidRows,0);
assert.equal(r.dataQuality.duplicateFixtureKeys,0);
assert.deepEqual(r.dataQuality.missingTeamAliases,[]);
assert.ok(r.evaluation.scoredFixtures>0);
assert.equal(r.evaluation.metrics.length,4);
for(const m of r.evaluation.metrics){assert.ok(Number.isFinite(m.accuracy));assert.ok(Number.isFinite(m.brierSum3));assert.ok(Number.isFinite(m.logLoss));}
console.log('PASS data-quality audit and walk-forward metric sanity checks');
