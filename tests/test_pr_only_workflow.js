'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const workflow = fs.readFileSync(path.join(__dirname, '..', '.github', 'workflows', 'update-predictions.yml'), 'utf8');

assert.ok(workflow.includes('pull-requests: write'), 'workflow needs permission to create a data-refresh PR');
assert.ok(workflow.includes("if: steps.pipeline.outcome == 'success' && github.ref == 'refs/heads/main'"), 'data refresh PR must only be created after a successful pipeline on main');
assert.ok(workflow.includes('branch: automation/toto-data-refresh'), 'generated data must be proposed on a dedicated branch');
assert.ok(!/git\s+push\s+origin\s+HEAD:main/.test(workflow), 'workflow must not push directly to main');
assert.ok(!/git\s+pull\s+--rebase[^\n]*origin\s+main/.test(workflow), 'workflow must not rebase and publish generated data to main');
assert.ok(!/^\s{2}deploy:\s*$/m.test(workflow), 'scheduled refresh workflow must not define a Pages deploy job');
assert.ok(!workflow.includes('actions/deploy-pages@'), 'scheduled refresh workflow must not deploy Pages');
assert.ok(!workflow.includes('actions/configure-pages@'), 'scheduled refresh workflow must not configure Pages');
assert.ok(workflow.includes('if: always()'), 'diagnostic artifact must be retained even after a failed pipeline');

console.log('PASS PR-only workflow guard: no main push, no Pages deployment, PR only after successful pipeline');
