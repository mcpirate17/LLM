#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');
const maxLines = 1250;

const suites = {
  dashboard: {
    label: 'research/dashboard',
    root: path.join(repoRoot, 'research/dashboard/src'),
    allowlist: new Set(['App.js']),
  },
  designer: {
    label: 'aria_designer/ui',
    root: path.join(repoRoot, 'aria_designer/ui/src'),
    allowlist: new Set(['App.legacy.jsx']),
  },
};

function walk(dir, out = []) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      walk(fullPath, out);
      continue;
    }
    if (/\.(js|jsx)$/.test(entry.name)) out.push(fullPath);
  }
  return out;
}

function countLines(filePath) {
  return fs.readFileSync(filePath, 'utf8').split('\n').length;
}

function checkSuite(key) {
  const suite = suites[key];
  if (!suite) {
    throw new Error(`Unknown UI suite "${key}"`);
  }

  const files = walk(suite.root);
  const offenders = [];
  const allowlisted = [];

  for (const filePath of files) {
    const lines = countLines(filePath);
    if (lines <= maxLines) continue;
    const relPath = path.relative(repoRoot, filePath);
    if (suite.allowlist.has(path.basename(filePath))) {
      allowlisted.push({ relPath, lines });
      continue;
    }
    offenders.push({ relPath, lines });
  }

  console.log(`\n[${suite.label}] UI size audit`);
  if (offenders.length === 0) {
    console.log(`  No non-allowlisted modules exceed ${maxLines} lines.`);
  } else {
    console.log(`  Offenders over ${maxLines} lines:`);
    for (const offender of offenders) {
      console.log(`  - ${offender.relPath}: ${offender.lines}`);
    }
  }

  if (allowlisted.length > 0) {
    console.log('  Temporary allowlist still present:');
    for (const entry of allowlisted) {
      console.log(`  - ${entry.relPath}: ${entry.lines}`);
    }
  }

  if (key === 'designer') {
    const entryPath = path.join(repoRoot, 'aria_designer/ui/src/App.jsx');
    const entryContents = fs.readFileSync(entryPath, 'utf8');
    if (entryContents.includes('App.legacy')) {
      console.log('  Warning: designer entrypoint still routes through App.legacy.jsx');
    }
  }

  return offenders.length === 0;
}

const requested = process.argv[2] ? [process.argv[2]] : Object.keys(suites);
let ok = true;

for (const key of requested) {
  ok = checkSuite(key) && ok;
}

if (!ok) {
  process.exit(1);
}
