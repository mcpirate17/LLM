#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SRC_DIR = path.join(ROOT, 'src');

const JS_EXTENSIONS = new Set(['.js', '.jsx']);
const HOOK_TOKENS = [
  'useState(',
  'useEffect(',
  'useMemo(',
  'useCallback(',
  'useRef(',
  'useReducer(',
  'useLayoutEffect(',
  'useImperativeHandle(',
  'useTransition(',
  'useDeferredValue(',
  'useId(',
  'useSyncExternalStore(',
  'useInsertionEffect(',
];

function walkFiles(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...walkFiles(fullPath));
      continue;
    }
    if (JS_EXTENSIONS.has(path.extname(entry.name))) {
      files.push(fullPath);
    }
  }
  return files;
}

function isLikelyComponentName(name) {
  return Boolean(name) && /^[A-Z]/.test(name);
}

function detectComponentStarts(lines) {
  const starts = [];
  const patterns = [
    /export\s+default\s+function\s+([A-Za-z0-9_]+)\s*\(/,
    /function\s+([A-Za-z0-9_]+)\s*\(/,
    /const\s+([A-Za-z0-9_]+)\s*=\s*\([^)]*\)\s*=>\s*\{/,
  ];

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    for (const pattern of patterns) {
      const match = line.match(pattern);
      if (match && isLikelyComponentName(match[1])) {
        starts.push({ line: index + 1, name: match[1], index });
        break;
      }
    }
  }

  return starts;
}

function isConditionalEarlyReturn(line) {
  return /^\s*if\s*\(.*\)\s*return\b/.test(line);
}

function hasHookToken(line) {
  return HOOK_TOKENS.some((token) => line.includes(token));
}

function analyzeComponent(lines, startIndex) {
  let depth = 0;
  let seenOpenBrace = false;
  let firstHookLine = null;
  let firstEarlyReturnLine = null;

  for (let index = startIndex; index < lines.length; index += 1) {
    const line = lines[index];

    if (line.includes('{')) {
      seenOpenBrace = true;
    }

    if (seenOpenBrace && depth === 1) {
      if (firstHookLine === null && hasHookToken(line)) {
        firstHookLine = index + 1;
      }
      if (firstEarlyReturnLine === null && isConditionalEarlyReturn(line)) {
        firstEarlyReturnLine = index + 1;
      }
    }

    const opens = (line.match(/\{/g) || []).length;
    const closes = (line.match(/\}/g) || []).length;
    depth += opens - closes;

    if (seenOpenBrace && depth === 0 && index > startIndex) {
      break;
    }
  }

  if (firstHookLine !== null && firstEarlyReturnLine !== null && firstEarlyReturnLine < firstHookLine) {
    return {
      firstHookLine,
      firstEarlyReturnLine,
    };
  }

  return null;
}

function main() {
  const files = walkFiles(SRC_DIR);
  const violations = [];

  for (const filePath of files) {
    const relPath = path.relative(ROOT, filePath);
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split(/\r?\n/);
    const starts = detectComponentStarts(lines);

    for (const start of starts) {
      const result = analyzeComponent(lines, start.index);
      if (result) {
        violations.push({
          file: relPath,
          component: start.name,
          returnLine: result.firstEarlyReturnLine,
          hookLine: result.firstHookLine,
        });
      }
    }
  }

  if (violations.length === 0) {
    console.log('Hook-order check passed: no early-return-before-hooks violations found.');
    process.exit(0);
  }

  console.error('Hook-order check failed. Found potential React hook-order violations:\n');
  for (const violation of violations) {
    console.error(`- ${violation.file} :: ${violation.component} (early return line ${violation.returnLine}, first hook line ${violation.hookLine})`);
  }
  console.error('\nFix by moving hook calls above conditional early returns.');
  process.exit(1);
}

main();
